"""
slack_handler.py — Slack Bolt async app + event routing.

Supported interactions:
  - DM the bot with any text  →  routed to best available agent
  - @mention the bot in a channel  →  routed to best available agent
  - /agents   →  list all connected agents in the orchestrator
  - /ask      →  explicitly route a free-text instruction
  - /agents-help  →  show usage instructions

Routing strategy (in _route_to_agents):
  1. Fetch all available/busy agents (excluding ourselves).
  2. Sort by status (available first) then by load score (lowest first).
  3. Try preferred generic capabilities first, then fall back to the agent's
     first declared capability.
  4. Send task_request with {"instruction": text, ...} and post the result.

Other agents can also call OUR send_slack_message capability to push
messages into Slack — handled in main.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import deque
from typing import Callable

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler

from orchestrator_client import OrchestratorClient, AGENT_NAME
from router import AgentRouter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON → Slack-readable formatter
# ---------------------------------------------------------------------------

def _scalar(value: object) -> str:
    """Format a leaf value as a plain string."""
    if value is None:
        return "_null_"
    if isinstance(value, bool):
        return "`true`" if value else "`false`"
    return str(value)


def _json_to_slack(obj: object, indent: int = 0) -> str:
    """Recursively render a parsed JSON object as an indented bullet list."""
    pad = "    " * indent   # 4 spaces per level
    lines: list[str] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict) and value:
                lines.append(f"{pad}• *{key}*:")
                lines.append(_json_to_slack(value, indent + 1))
            elif isinstance(value, list) and value:
                lines.append(f"{pad}• *{key}*:")
                lines.append(_json_to_slack(value, indent + 1))
            else:
                lines.append(f"{pad}• *{key}*: {_scalar(value)}")
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)) and item:
                lines.append(f"{pad}•")
                lines.append(_json_to_slack(item, indent + 1))
            else:
                lines.append(f"{pad}• {_scalar(item)}")
    else:
        lines.append(f"{pad}{_scalar(obj)}")

    return "\n".join(lines)


def format_for_slack(text: str) -> str:
    """
    If *text* is a JSON string (object or array), convert it to a
    human-readable bullet list with key: value formatting.
    Passes plain text through unchanged.
    """
    stripped = text.strip() if text else ""
    if not stripped or stripped[0] not in ("{", "["):
        return text
    try:
        parsed = json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return text
    if not isinstance(parsed, (dict, list)):
        return text
    return _json_to_slack(parsed)


_HISTORY_TURNS = 10  # max turns kept per Slack thread/channel


class ConversationBuffer:
    """
    Rolling per-thread message buffer for Slack conversations.
    Key: "{channel_id}:{thread_ts}" — DMs use channel_id as the thread root,
    channel threads are scoped to their thread_ts.
    """

    def __init__(self, max_turns: int = _HISTORY_TURNS) -> None:
        self._max = max_turns
        self._threads: dict[str, deque] = {}

    def _key(self, channel_id: str, thread_ts: str | None) -> str:
        return f"{channel_id}:{thread_ts or ''}"

    def record(self, channel_id: str, thread_ts: str | None, sender: str, text: str) -> None:
        key = self._key(channel_id, thread_ts)
        if key not in self._threads:
            self._threads[key] = deque(maxlen=self._max)
        self._threads[key].append({"sender": sender, "text": text[:600]})

    def history(self, channel_id: str, thread_ts: str | None) -> list[dict]:
        return list(self._threads.get(self._key(channel_id, thread_ts), []))


# Capabilities that general-purpose agents are expected to expose.
# The connector tries each in order when routing a free-text instruction.
PREFERRED_CAPABILITIES = [
    "execute_task",
    "handle_instruction",
    "handle_task",
    "process_request",
    "run_task",
]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_ISO_8601_UTC_OR_OFFSET_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?(?:Z|[+-]\d{2}:\d{2})\b"
)


def _generic_input_data(
    text: str,
    user_id: str | None = None,
    channel_id: str | None = None,
) -> dict:
    """Build free-text input_data variants used by generic capabilities."""
    input_data: dict = {
        "task": text,
        "instruction": text,
        "query": text,
        "prompt": text,
        "text": text,
    }
    if user_id:
        input_data["user_id"] = user_id
    if channel_id:
        input_data["channel_id"] = channel_id
    return input_data


def _extract_iso_datetime(text: str) -> str | None:
    match = _ISO_8601_UTC_OR_OFFSET_RE.search(text)
    return match.group(0) if match else None


def _normalise_schedule_task_input(
    text: str,
    extracted: dict | None,
    user_id: str | None = None,
    channel_id: str | None = None,
) -> tuple[dict | None, str | None]:
    """
    Ensure schedule_task payload includes required fields.

    Returns (payload, error_text). If error_text is set, caller should reply
    to user and abort dispatch.
    """
    payload: dict = dict(extracted) if isinstance(extracted, dict) else {}
    payload["capability"] = (payload.get("capability") or "execute_task").strip()

    nested = payload.get("input_data")
    if not isinstance(nested, dict) or not nested:
        nested = _generic_input_data(text, user_id=user_id, channel_id=channel_id)
    else:
        # Ensure wrapped task still has a textual instruction.
        if not any(k in nested for k in ("task", "instruction", "query", "prompt", "text")):
            nested["instruction"] = text
            nested["task"] = text
        if user_id and "user_id" not in nested:
            nested["user_id"] = user_id
        if channel_id and "channel_id" not in nested:
            nested["channel_id"] = channel_id
    payload["input_data"] = nested

    scheduled_at = payload.get("scheduled_at")
    if not (isinstance(scheduled_at, str) and scheduled_at.strip()):
        inferred = _extract_iso_datetime(text)
        if inferred:
            payload["scheduled_at"] = inferred
        else:
            return None, (
                ":warning: `schedule_task` needs a time. Include an ISO 8601 datetime, "
                "for example `2026-02-26T15:00:00Z`."
            )

    return payload, None

def _agent_list_text(agents: list[dict]) -> str:
    if not agents:
        return ":warning: No agents are currently connected."

    lines = ["*Connected agents:*\n"]
    for a in agents:
        status = a.get("status", "unknown")
        emoji = {
            "available": ":large_green_circle:",
            "busy": ":large_yellow_circle:",
            "draining": ":large_orange_circle:",
            "error": ":red_circle:",
        }.get(status, ":white_circle:")

        name = a.get("name", "unknown")
        caps = a.get("capabilities", [])
        caps_str = ", ".join(f"`{c}`" for c in caps) if caps else "_none_"
        load = a.get("current_load", 0.0)
        lines.append(f"{emoji} *{name}*  |  capabilities: {caps_str}  |  load: {load:.0%}")

    return "\n".join(lines)


def _strip_mention(text: str) -> str:
    """Remove <@USERID> mentions from the start of a message."""
    return re.sub(r"^\s*<@[A-Z0-9]+>\s*", "", text).strip()


def _pick_capability(agent: dict) -> str:
    """Choose which capability to invoke on the target agent."""
    declared = agent.get("capabilities", [])
    for preferred in PREFERRED_CAPABILITIES:
        if preferred in declared:
            return preferred
    return declared[0] if declared else "execute_task"


# ---------------------------------------------------------------------------
# Core routing logic
# ---------------------------------------------------------------------------

async def _route_to_agents(
    text: str,
    orchestrator: OrchestratorClient,
    reply: Callable,
    router: AgentRouter,
    thread_ts: str | None = None,
    user_id: str | None = None,
    channel_id: str | None = None,
    timeout_ms: float = 120_000,
    session_history: list[dict] | None = None,
) -> None:
    """Dispatch every Slack instruction to task-planner-agent via plan_task."""

    async def say(msg: str) -> None:
        kwargs: dict = {"text": msg}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        await reply(**kwargs)

    # 1. Gate on WS connectivity — if we can't send, no point routing
    if not orchestrator.is_connected:
        await say(
            ":hourglass_flowing_sand: Still connecting to the agent network. "
            "Please try again in a moment."
        )
        return

    # 2. Discover agents; exclude ourselves by agent_id AND by name to catch
    #    stale records from a previous run (different agent_id, same name) that
    #    the orchestrator hasn't expired yet.
    try:
        agents = await orchestrator.discover_agents()
        agents = [
            a for a in agents
            if a.get("agent_id") != orchestrator.agent_id
            and a.get("name") != AGENT_NAME
        ]
    except Exception as exc:
        await say(f":x: Could not reach the orchestrator: `{exc}`")
        return

    if not agents:
        await say(":warning: No agents are currently available. Please try again later.")
        return

    # 3. Always route to the task planner. It will generate a plan and
    #    auto-forward to task-executor-agent when available.
    planners = [
        a for a in agents
        if a.get("name") == "task-planner-agent"
        or "plan_task" in a.get("capabilities", [])
    ]
    if not planners:
        await say(
            ":warning: `task-planner-agent` is not available. "
            "Please start it and try again."
        )
        return
    planners.sort(
        key=lambda a: (
            {"available": 0, "busy": 1}.get(a.get("status", ""), 2),
            a.get("score", 1.0),
        )
    )
    target = planners[0]
    target_id = target["agent_id"]
    target_name = target.get("name", target_id)
    capability = "plan_task"

    await say(
        f":arrows_counterclockwise: Routing to *{target_name}* "
        f"(capability: `{capability}`) …"
    )

    # 4. Send a planner-oriented payload and include Slack context for
    #    downstream planner/executor logic.
    input_data = {
        "goal": text,
        "auto_execute": True,
        "user_id": user_id,
        # Planner reads these top-level fields directly.
        "channel_id": channel_id,
        "thread_id": thread_ts,
        "session_history": session_history or [],
        "payload": {
            "source": "slack",
            "text": text,
            "user_id": user_id,
            "channel_id": channel_id,
            "thread_ts": thread_ts,
        },
    }

    logger.info(
        "Dispatching task | agent=%s (id=%s) capability=%s input=%r",
        target_name, target_id, capability, input_data,
    )

    try:
        result = await orchestrator.send_task(
            target_agent_id=target_id,
            capability=capability,
            input_data=input_data,
            timeout_ms=timeout_ms,
        )
    except TimeoutError:
        await say(f":clock1: *{target_name}* did not respond within {timeout_ms / 1000:.0f} s.")
        return
    except RuntimeError as exc:
        err = str(exc)
        if "WebSocket is not connected" in err:
            await say(
                ":hourglass_flowing_sand: Lost connection to the agent network mid-request. "
                "Reconnecting — please try again in a moment."
            )
        else:
            await say(f":x: Orchestrator error: `{exc}`")
        return
    except Exception as exc:
        await say(f":x: Unexpected error while waiting for response: `{exc}`")
        return

    # 4. Format and post response
    if result.get("success"):
        output = result.get("output_data") or {}
        response_text = (
            output.get("result")
            or output.get("response")
            or output.get("message")
            or output.get("summary")
            or output.get("text")
            or output.get("echo")
            or str(output)
        )
        await say(f":white_check_mark: *{target_name}* responded:\n\n{format_for_slack(str(response_text))}")
    else:
        error = result.get("error", "unknown error")
        await say(f":x: *{target_name}* failed: {error}")


# ---------------------------------------------------------------------------
# Slack app setup
# ---------------------------------------------------------------------------

class SlackConnectorApp:
    """Wraps the Slack Bolt async app and the Socket Mode handler."""

    def __init__(
        self,
        bot_token: str,
        app_token: str,
        signing_secret: str,
        orchestrator: OrchestratorClient,
        router: AgentRouter,
        task_timeout_ms: float = 120_000,
        conv_buffer: ConversationBuffer | None = None,
    ) -> None:
        self.app = AsyncApp(token=bot_token, signing_secret=signing_secret)
        self._app_token = app_token
        self._orchestrator = orchestrator
        self._router = router
        self._timeout_ms = task_timeout_ms
        self._handler: AsyncSocketModeHandler | None = None
        self.conv_buffer = conv_buffer or ConversationBuffer()
        self._register_handlers()

    def _register_handlers(self) -> None:
        app = self.app
        orc = self._orchestrator
        router = self._router
        timeout_ms = self._timeout_ms
        buf = self.conv_buffer

        # ── @mention in a channel ──────────────────────────────────────────
        @app.event("app_mention")
        async def on_mention(body: dict, say, event: dict) -> None:
            text = _strip_mention(event.get("text", ""))
            if not text:
                return
            thread_ts = event.get("thread_ts") or event.get("ts")
            user_id = event.get("user")
            channel_id = event.get("channel")
            logger.info(
                "→ SLACK REQUEST [app_mention] user=%s channel=%s\n%s",
                user_id, channel_id, json.dumps(body, indent=2, default=str),
            )

            async def logged_say(**kwargs):
                logger.info("← SLACK RESPONSE [app_mention] %s", json.dumps(kwargs, default=str))
                await say(**kwargs)

            if await _handle_builtin(text, logged_say, orc, thread_ts):
                return

            prior_history = buf.history(channel_id, thread_ts)
            buf.record(channel_id, thread_ts, "user", text)

            await _route_to_agents(
                text=text,
                orchestrator=orc,
                reply=logged_say,
                router=router,
                thread_ts=thread_ts,
                user_id=user_id,
                channel_id=channel_id,
                timeout_ms=timeout_ms,
                session_history=prior_history,
            )

        # ── Direct messages ────────────────────────────────────────────────
        @app.event("message")
        async def on_dm(body: dict, say, event: dict) -> None:
            # Handle direct-message surfaces. Channel messages are covered by
            # app_mention to avoid duplicate routing.
            channel_type = event.get("channel_type")
            if channel_type not in {"im", "mpim", "app_home"}:
                return
            # Ignore bot's own messages
            if event.get("bot_id"):
                return

            text = (event.get("text") or "").strip()
            if not text:
                return
            thread_ts = event.get("thread_ts") or event.get("ts")
            user_id = event.get("user")
            channel_id = event.get("channel")
            logger.info(
                "→ SLACK REQUEST [dm] user=%s channel=%s\n%s",
                user_id, channel_id, json.dumps(body, indent=2, default=str),
            )

            async def logged_say(**kwargs):
                logger.info("← SLACK RESPONSE [dm] %s", json.dumps(kwargs, default=str))
                await say(**kwargs)

            if await _handle_builtin(text, logged_say, orc, thread_ts):
                return

            prior_history = buf.history(channel_id, thread_ts)
            buf.record(channel_id, thread_ts, "user", text)

            await _route_to_agents(
                text=text,
                orchestrator=orc,
                reply=logged_say,
                router=router,
                thread_ts=thread_ts,
                user_id=user_id,
                channel_id=channel_id,
                timeout_ms=timeout_ms,
                session_history=prior_history,
            )

        # ── /agents slash command ──────────────────────────────────────────
        @app.command("/agents")
        async def cmd_agents(ack, respond, command: dict) -> None:
            await ack()
            logger.info(
                "→ SLACK REQUEST [/agents] user=%s channel=%s\n%s",
                command.get("user_id"), command.get("channel_id"),
                json.dumps(command, indent=2, default=str),
            )
            try:
                agents = await orc.discover_agents()
                agents = [a for a in agents if a.get("agent_id") != orc.agent_id]
                reply_text = _agent_list_text(agents)
                logger.info("← SLACK RESPONSE [/agents] %s", json.dumps({"text": reply_text}, default=str))
                await respond(reply_text)
            except Exception as exc:
                reply_text = f":x: Failed to list agents: `{exc}`"
                logger.info("← SLACK RESPONSE [/agents] %s", json.dumps({"text": reply_text}, default=str))
                await respond(reply_text)

        # ── /ask slash command ─────────────────────────────────────────────
        @app.command("/ask")
        async def cmd_ask(ack, respond, command: dict) -> None:
            await ack()
            text = (command.get("text") or "").strip()
            logger.info(
                "→ SLACK REQUEST [/ask] user=%s channel=%s\n%s",
                command.get("user_id"), command.get("channel_id"),
                json.dumps(command, indent=2, default=str),
            )
            if not text:
                reply_text = "Usage: `/ask <your instruction>`\nExample: `/ask summarise the latest news`"
                logger.info("← SLACK RESPONSE [/ask] %s", json.dumps({"text": reply_text}, default=str))
                await respond(reply_text)
                return

            async def logged_respond(**kwargs):
                logger.info("← SLACK RESPONSE [/ask] %s", json.dumps(kwargs, default=str))
                await respond(**kwargs)

            await _route_to_agents(
                text=text,
                orchestrator=orc,
                reply=logged_respond,
                router=router,
                user_id=command.get("user_id"),
                channel_id=command.get("channel_id"),
                timeout_ms=timeout_ms,
            )

        # ── /agents-help slash command ─────────────────────────────────────
        @app.command("/agents-help")
        async def cmd_help(ack, respond, command: dict) -> None:
            await ack()
            logger.info(
                "→ SLACK REQUEST [/agents-help] user=%s channel=%s\n%s",
                command.get("user_id"), command.get("channel_id"),
                json.dumps(command, indent=2, default=str),
            )
            reply_text = (
                "*Slack Connector Agent — help*\n\n"
                "*How to interact:*\n"
                "• DM the bot with any instruction\n"
                "• Mention the bot in a channel: `@SlackConnector <instruction>`\n\n"
                "*Slash commands:*\n"
                "• `/agents` — list all connected agents and their capabilities\n"
                "• `/ask <instruction>` — route an instruction to the best available agent\n"
                "• `/agents-help` — show this help message\n\n"
                "*Examples:*\n"
                "> `Find me a laptop under $1000`\n"
                "> `Summarise today's news headlines`\n"
                "> `Run a web search for Python asyncio best practices`"
            )
            logger.info("← SLACK RESPONSE [/agents-help] %s", json.dumps({"text": reply_text}, default=str))
            await respond(reply_text)

    async def start(self) -> None:
        """Start the Slack Socket Mode handler (non-blocking — returns after connecting)."""
        self._handler = AsyncSocketModeHandler(self.app, self._app_token)
        await self._handler.start_async()
        logger.info("Slack Socket Mode handler started")

    async def stop(self) -> None:
        if self._handler:
            try:
                await self._handler.close_async()
            except Exception as exc:
                logger.warning("Error closing Slack handler: %s", exc)


# ---------------------------------------------------------------------------
# Built-in commands (parsed before routing to agents)
# ---------------------------------------------------------------------------

async def _handle_builtin(
    text: str,
    say,
    orchestrator: OrchestratorClient,
    thread_ts: str | None,
) -> bool:
    """Handle simple built-in commands. Returns True if the message was consumed."""

    cmd = text.lower().strip().rstrip("?")

    if cmd in ("help", "?", "h"):
        await say(
            "*Slack Connector Agent*\n\n"
            "I route your messages to agents in the orchestrator network.\n\n"
            "*Built-in commands (in DMs or @mentions):*\n"
            "• `help` — show this message\n"
            "• `list agents` — show connected agents\n\n"
            "*Slash commands:*\n"
            "• `/agents` — list connected agents\n"
            "• `/ask <instruction>` — route an instruction\n"
            "• `/agents-help` — full help\n\n"
            "Or just type your request and I'll route it automatically.",
            thread_ts=thread_ts,
        )
        return True

    if cmd in ("list agents", "agents", "show agents", "list"):
        try:
            agents = await orchestrator.discover_agents()
            agents = [a for a in agents if a.get("agent_id") != orchestrator.agent_id]
            await say(_agent_list_text(agents), thread_ts=thread_ts)
        except Exception as exc:
            await say(f":x: Failed to list agents: `{exc}`", thread_ts=thread_ts)
        return True

    return False
