"""
main.py — Entry point for the Slack Connector Agent.

Slack credentials (SLACK_BOT_TOKEN, SLACK_APP_TOKEN, SLACK_SIGNING_SECRET) are
NOT required at startup. They can be supplied via:

  1. Environment variables / .env file  (checked first)
  2. Orchestrator dashboard settings    (pushed at registration or via settings_push)

Start order:
  1. Load optional config from environment / .env
  2. Register with the orchestrator (with retry) — receives any saved settings
  3. Start the Slack Socket Mode handler if credentials are already available,
     otherwise log a warning and wait for a settings_push from the dashboard
  4. Connect the orchestrator WebSocket (blocks until shutdown)
  5. On settings_push: start (or restart) the Slack connection with new credentials
  6. On SIGINT / SIGTERM: deregister, close Slack handler, exit cleanly
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal

from contextlib import suppress
from dotenv import load_dotenv
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.errors import SlackApiError

from orchestrator_client import OrchestratorClient
from router import AgentRouter, RouterMode
from slack_handler import SlackConnectorApp, ConversationBuffer, format_for_slack

load_dotenv()

logging.basicConfig(
    level=logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO").upper()),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("slack-connector")


# ---------------------------------------------------------------------------
# Slack lifecycle manager
# ---------------------------------------------------------------------------

class _SlackLifecycle:
    """Manages deferred startup and hot-restart of the Slack connection."""

    def __init__(
        self,
        orchestrator: OrchestratorClient,
        router: AgentRouter,
        task_timeout_ms: float,
        conv_buffer: ConversationBuffer,
    ) -> None:
        self._orchestrator = orchestrator
        self._router = router
        self._timeout_ms = task_timeout_ms
        self._conv_buffer = conv_buffer

        self.app: SlackConnectorApp | None = None
        self.client: AsyncWebClient | None = None
        self._task: asyncio.Task | None = None

    @property
    def is_ready(self) -> bool:
        return self.client is not None

    async def start(self, bot_token: str, app_token: str, signing_secret: str) -> None:
        """(Re)start the Slack connection with the given credentials."""
        await self.stop()

        logger.info("Starting Slack connection ...")
        self.client = AsyncWebClient(token=bot_token)
        self.app = SlackConnectorApp(
            bot_token=bot_token,
            app_token=app_token,
            signing_secret=signing_secret,
            orchestrator=self._orchestrator,
            router=self._router,
            task_timeout_ms=self._timeout_ms,
            conv_buffer=self._conv_buffer,
        )
        self._task = asyncio.create_task(self.app.start(), name="slack_socket_mode")
        logger.info("Slack Socket Mode started")

    async def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        if self.app:
            await self.app.stop()
        self.app = None
        self.client = None
        self._task = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _optional(key: str, default: str = "") -> str:
    return os.getenv(key, default)


def _extract_credentials(settings: dict) -> tuple[str, str, str]:
    """Pull the three Slack credential keys from a settings dict."""
    return (
        settings.get("slack_bot_token", ""),
        settings.get("slack_app_token", ""),
        settings.get("slack_signing_secret", ""),
    )


def _credentials_complete(bot: str, app: str, secret: str) -> bool:
    return bool(bot and app and secret)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    orchestrator_url = _optional("ORCHESTRATOR_URL", "http://localhost:8000")

    # Read from env — all optional; dashboard settings take precedence if set
    env_bot_token      = _optional("SLACK_BOT_TOKEN")
    env_app_token      = _optional("SLACK_APP_TOKEN")
    env_signing_secret = _optional("SLACK_SIGNING_SECRET")
    task_timeout_ms    = float(_optional("TASK_TIMEOUT_MS", "120000"))

    # ------------------------------------------------------------------
    # Routing config
    # ------------------------------------------------------------------
    routing_mode_str = _optional("ROUTING_MODE", "hybrid").lower()
    valid_modes = {m.value for m in RouterMode}
    if routing_mode_str not in valid_modes:
        logger.warning(
            "Invalid ROUTING_MODE=%r (valid: %s). Defaulting to 'hybrid'.",
            routing_mode_str, ", ".join(sorted(valid_modes)),
        )
        routing_mode_str = "hybrid"
    routing_mode = RouterMode(routing_mode_str)

    try:
        routing_threshold = float(_optional("ROUTING_CONFIDENCE_THRESHOLD", "0.3"))
        if not (0.0 <= routing_threshold <= 1.0):
            raise ValueError("out of range")
    except ValueError:
        routing_threshold = 0.3

    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if routing_mode == RouterMode.LLM and not anthropic_api_key:
        logger.warning(
            "ROUTING_MODE=llm requires ANTHROPIC_API_KEY. Falling back to 'hybrid'."
        )
        routing_mode = RouterMode("hybrid")

    agent_router = AgentRouter(
        mode=routing_mode,
        confidence_threshold=routing_threshold,
        anthropic_api_key=anthropic_api_key,
    )
    logger.info(
        "AgentRouter configured: mode=%s, threshold=%.2f",
        routing_mode.value, routing_threshold,
    )

    # ------------------------------------------------------------------
    # 1. Register with orchestrator (retry up to 5 times)
    # ------------------------------------------------------------------
    orchestrator = OrchestratorClient(base_url=orchestrator_url)
    logger.info("Registering with orchestrator at %s ...", orchestrator_url)

    for attempt in range(1, 6):
        try:
            await orchestrator.register()
            break
        except Exception as exc:
            if attempt == 5:
                logger.error("Could not register after 5 attempts: %s", exc)
                raise
            wait = 2 ** attempt
            logger.warning(
                "Registration attempt %d failed (%s) — retrying in %d s",
                attempt, exc, wait,
            )
            await asyncio.sleep(wait)

    # ------------------------------------------------------------------
    # 2. Resolve credentials: env vars take priority, then dashboard settings
    # ------------------------------------------------------------------
    settings = orchestrator.common_settings
    bot_token      = env_bot_token      or settings.get("slack_bot_token", "")
    app_token      = env_app_token      or settings.get("slack_app_token", "")
    signing_secret = env_signing_secret or settings.get("slack_signing_secret", "")

    # ------------------------------------------------------------------
    # 3. Slack lifecycle manager + task handler
    # ------------------------------------------------------------------
    conv_buffer = ConversationBuffer()
    lifecycle = _SlackLifecycle(orchestrator, agent_router, task_timeout_ms, conv_buffer)

    async def handle_send_slack_message(msg: dict) -> dict | None:
        payload = msg.get("payload", {})
        if payload.get("capability") != "send_slack_message":
            return None

        if not lifecycle.is_ready:
            raise RuntimeError(
                "Slack credentials not configured. "
                "Set slack_bot_token, slack_app_token, and slack_signing_secret "
                "in the orchestrator dashboard settings for this agent."
            )

        input_data = payload.get("input_data", {})
        # Accept both "channel" and "channel_id" — planners emit channel_id
        channel    = input_data.get("channel") or input_data.get("channel_id")
        user_id    = input_data.get("user_id")
        text       = input_data.get("text")
        thread_ts  = input_data.get("thread_ts")

        if not text:
            raise ValueError("send_slack_message requires 'text'")
        if not channel and not user_id:
            raise ValueError("send_slack_message requires 'channel' or 'user_id'")

        text = format_for_slack(text)

        client = lifecycle.client
        if not channel and user_id:
            dm = await client.conversations_open(users=user_id)
            channel = dm.get("channel", {}).get("id")

        kwargs: dict = {"channel": channel, "text": text}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts

        try:
            resp = await client.chat_postMessage(**kwargs)
        except SlackApiError as exc:
            err = (exc.response or {}).get("error")
            if err == "channel_not_found" and user_id:
                dm = await client.conversations_open(users=user_id)
                dm_channel = dm.get("channel", {}).get("id")
                if not dm_channel:
                    raise
                kwargs["channel"] = dm_channel
                kwargs.pop("thread_ts", None)
                resp = await client.chat_postMessage(**kwargs)
            else:
                raise
        if resp.get("ok"):
            conv_buffer.record(kwargs["channel"], thread_ts, "assistant", text)
        return {"ok": bool(resp.get("ok")), "ts": resp.get("ts", "")}

    orchestrator.on_task_request(handle_send_slack_message)

    # ------------------------------------------------------------------
    # 4. Settings push handler — (re)starts Slack when credentials arrive
    # ------------------------------------------------------------------
    async def on_settings(new_settings: dict) -> None:
        nonlocal bot_token, app_token, signing_secret

        new_bot    = env_bot_token      or new_settings.get("slack_bot_token", "")
        new_app    = env_app_token      or new_settings.get("slack_app_token", "")
        new_secret = env_signing_secret or new_settings.get("slack_signing_secret", "")

        if not _credentials_complete(new_bot, new_app, new_secret):
            logger.debug("Settings push received but Slack credentials still incomplete")
            return

        # Only restart if the tokens actually changed
        if new_bot == bot_token and new_app == app_token and new_secret == signing_secret and lifecycle.is_ready:
            return

        bot_token      = new_bot
        app_token      = new_app
        signing_secret = new_secret
        await lifecycle.start(bot_token, app_token, signing_secret)

    orchestrator.on_settings_push(on_settings)

    # ------------------------------------------------------------------
    # 5. Start Slack immediately if credentials are already available
    # ------------------------------------------------------------------
    if _credentials_complete(bot_token, app_token, signing_secret):
        await lifecycle.start(bot_token, app_token, signing_secret)
    else:
        logger.warning(
            "Slack credentials not set — agent will connect to the orchestrator "
            "but Slack integration is inactive until credentials are configured "
            "in the dashboard (Agent Settings: slack_bot_token, slack_app_token, "
            "slack_signing_secret)."
        )

    # ------------------------------------------------------------------
    # 6. Shutdown wiring
    # ------------------------------------------------------------------
    loop = asyncio.get_running_loop()
    shutdown_event = asyncio.Event()

    def _on_signal(*_) -> None:
        logger.info("Shutdown signal received")
        shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _on_signal)

    # ------------------------------------------------------------------
    # 7. Run orchestrator WS; Slack runs as a background task managed by
    #    lifecycle.  We stop on signal or orchestrator exit.
    # ------------------------------------------------------------------
    logger.info("Slack Connector Agent is running.")

    orchestrator_task = asyncio.create_task(orchestrator.connect_and_run())
    shutdown_task     = asyncio.create_task(shutdown_event.wait())

    done, pending = await asyncio.wait(
        [orchestrator_task, shutdown_task],
        return_when=asyncio.FIRST_COMPLETED,
    )
    for t in pending:
        t.cancel()
        with suppress(asyncio.CancelledError, Exception):
            await t

    # ------------------------------------------------------------------
    # 8. Cleanup
    # ------------------------------------------------------------------
    logger.info("Shutting down ...")
    await lifecycle.stop()
    await orchestrator.shutdown()
    logger.info("Slack Connector Agent stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
