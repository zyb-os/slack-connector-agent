"""
orchestrator_client.py — WebSocket + HTTP client for the agent orchestrator.

Implements the full protocol described in AGENT_MANIFEST.md:
  - Registration (POST /api/v1/agents/register)
  - WebSocket connection with automatic reconnection + re-registration
  - Heartbeat loop (every 15 s)
  - Task request dispatch and response correlation
  - Discovery (GET /api/v1/discover)
  - Graceful shutdown (DELETE /api/v1/agents/{id})
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from pathlib import Path

import httpx
import websockets
from websockets.exceptions import ConnectionClosed

logger = logging.getLogger(__name__)

AGENT_NAME = "slack-connector"
AGENT_VERSION = "1.0.0"
AGENT_DESCRIPTION = "Bridges Slack users with the agent orchestrator network"
HEARTBEAT_INTERVAL = 15  # seconds

_ID_FILE = Path(".agent_id")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_agent_id() -> str:
    """Return a persistent agent UUID, generating and saving one on first call."""
    if _ID_FILE.exists():
        return _ID_FILE.read_text().strip()
    agent_id = str(uuid.uuid4())
    _ID_FILE.write_text(agent_id)
    logger.info("Generated new stable agent ID: %s (saved to %s)", agent_id, _ID_FILE)
    return agent_id


class OrchestratorClient:
    """Manages the full lifecycle of an agent connected to the orchestrator."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self.agent_id: Optional[str] = None
        self.ws_url: Optional[str] = None
        self._ws: Optional[Any] = None
        self._running = False
        self._stop_event = asyncio.Event()

        # correlation_id -> Future: resolves when matching task_response arrives
        self._pending: dict[str, asyncio.Future] = {}

        # Handlers called for incoming task_request messages
        self._task_handlers: list[Callable] = []

        # Handlers called when the orchestrator pushes new settings
        self._settings_handlers: list[Callable] = []

        # Settings received at registration or via settings_push
        self.common_settings: dict = {}

        # Metrics
        self._active_tasks = 0
        self._tasks_completed = 0
        self._tasks_failed = 0
        self._start_time = time.monotonic()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        """True when the WebSocket connection to the orchestrator is active."""
        return self._ws is not None

    def on_settings_push(self, handler: Callable) -> None:
        """Register an async handler called whenever the orchestrator pushes settings.

        The handler receives the full settings dict and should return None.
        It is called for both settings_push WS messages and the initial
        common_settings/agent_settings received at registration time.
        """
        self._settings_handlers.append(handler)

    def on_task_request(self, handler: Callable) -> None:
        """Register an async handler for incoming task_request messages.

        The handler receives the full message envelope (dict) and should
        return a dict (output_data) on success, or raise an exception on
        failure.  Return None to pass to the next handler.
        """
        self._task_handlers.append(handler)

    async def register(self) -> str:
        """Register with the orchestrator and return the assigned agent_id."""
        payload = {
            "id": _stable_agent_id(),
            "name": AGENT_NAME,
            "description": AGENT_DESCRIPTION,
            "version": AGENT_VERSION,
            "capabilities": [
                {
                    "name": "send_slack_message",
                    "description": "Send a message to a Slack channel or thread.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "channel": {
                                "type": "string",
                                "description": "Channel ID or name e.g. #general",
                            },
                            "text": {
                                "type": "string",
                                "description": "Message body (plain text or mrkdwn)",
                            },
                            "thread_ts": {
                                "type": "string",
                                "description": "Parent timestamp for thread replies",
                            },
                        },
                        "required": ["channel", "text"],
                    },
                    "tags": ["slack", "messaging", "notification", "send", "post", "notify", "alert", "communicate", "chat", "message", "team", "channel", "announce"],
                    "cost": {"type": "free", "estimated_cost_usd": None},
                }
            ],
            "tags": ["slack", "connector", "gateway"],
            "required_settings": [
                {
                    "key": "slack_bot_token",
                    "label": "Slack Bot Token",
                    "type": "secret",
                    "required": True,
                    "description": "Slack Bot Token (xoxb-…).",
                },
                {
                    "key": "slack_app_token",
                    "label": "Slack App-Level Token",
                    "type": "secret",
                    "required": True,
                    "description": "App-level token for Socket Mode (xapp-…).",
                },
                {
                    "key": "slack_signing_secret",
                    "label": "Slack Signing Secret",
                    "type": "secret",
                    "required": True,
                    "description": "Signing secret for request verification.",
                },
                {
                    "key": "default_channel",
                    "label": "Default Channel",
                    "type": "string",
                    "required": False,
                    "description": "Fallback channel when none specified.",
                    "default": "general",
                },
                {
                    "key": "task_timeout_ms",
                    "label": "Task Timeout (ms)",
                    "type": "integer",
                    "required": False,
                    "description": "Agent response timeout.",
                    "default": 120000,
                    "min_value": 5000,
                    "max_value": 600000,
                },
            ],
        }

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.base_url}/api/v1/agents/register",
                json=payload,
                timeout=10.0,
            )
            resp.raise_for_status()
            data = resp.json()

        self.agent_id = data["agent_id"]
        self.ws_url = data["ws_url"]
        # Merge common settings and per-agent saved settings returned at registration
        self.common_settings = {
            **data.get("common_settings", {}),
            **data.get("agent_settings", {}),
        }
        logger.info("Registered as %s  ws_url=%s", self.agent_id, self.ws_url)
        if self.common_settings:
            logger.info("Received %d setting(s) from orchestrator at registration", len(self.common_settings))
        return self.agent_id

    async def discover_agents(
        self,
        capability: Optional[str] = None,
        tags: Optional[list[str]] = None,
        status: str = "available,busy",
    ) -> list[dict]:
        """Return agents from the orchestrator, optionally filtered."""
        params: dict[str, str] = {"status": status}
        if capability:
            params["capability"] = capability
        if tags:
            params["tags"] = ",".join(tags)

        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.base_url}/api/v1/discover",
                params=params,
                timeout=10.0,
            )
            resp.raise_for_status()
            return resp.json()

    async def send_task(
        self,
        target_agent_id: str,
        capability: str,
        input_data: dict,
        timeout_ms: float = 120_000,
    ) -> dict:
        """Send a task_request and await the matching task_response payload.

        Returns the task_response payload dict:
          {"success": bool, "output_data": {...}, "duration_ms": float}

        Raises TimeoutError if no response within timeout_ms.
        Raises RuntimeError on orchestrator-level errors (AGENT_UNAVAILABLE etc.).
        """
        if not self._ws:
            raise RuntimeError("WebSocket is not connected")

        req_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut

        msg = {
            "id": req_id,
            "type": "task_request",
            "sender_id": self.agent_id,
            "recipient_id": target_agent_id,
            "payload": {
                "capability": capability,
                "input_data": input_data,
                "timeout_ms": timeout_ms,
            },
            "timestamp": _now_iso(),
            "correlation_id": None,
        }

        await self._ws.send(json.dumps(msg))
        logger.debug("Sent task_request id=%s to %s", req_id, target_agent_id)

        try:
            return await asyncio.wait_for(fut, timeout=timeout_ms / 1000)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            raise TimeoutError(f"task_request {req_id} timed out after {timeout_ms} ms")

    async def connect_and_run(self) -> None:
        """Connect to the orchestrator WebSocket and run until stopped.

        Automatically reconnects (with exponential back-off) on disconnection.
        Re-registers if the orchestrator returns close code 4004.
        """
        self._running = True
        retry_delay = 1.0

        while self._running:
            try:
                logger.info("Connecting to %s", self.ws_url)
                async with websockets.connect(self.ws_url) as ws:
                    self._ws = ws
                    retry_delay = 1.0  # reset on successful connect
                    logger.info("WebSocket connected")

                    sender_task = asyncio.create_task(self._heartbeat_loop(), name="heartbeat")
                    receiver_task = asyncio.create_task(self._recv_loop(), name="recv_loop")
                    stop_task = asyncio.create_task(self._stop_event.wait(), name="stop")

                    done, pending = await asyncio.wait(
                        [sender_task, receiver_task, stop_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for t in pending:
                        t.cancel()

                    # Diagnose which task(s) finished and why
                    for t in done:
                        task_name = t.get_name()
                        if t.cancelled():
                            logger.debug("Task %r was cancelled", task_name)
                        elif t.exception():
                            logger.warning("Task %r raised: %s", task_name, t.exception())
                        else:
                            logger.info("Task %r finished normally", task_name)

                    # Re-raise any exception from the completed tasks
                    for t in done:
                        if t != stop_task and not t.cancelled():
                            exc = t.exception()
                            if exc:
                                raise exc

            except ConnectionClosed as exc:
                if exc.code == 4004:
                    logger.warning("Orchestrator does not recognise our agent_id (4004) — re-registering")
                    await self.register()
                elif exc.code == 4003:
                    logger.info("Agent is disabled by orchestrator (4003) — will retry so dashboard enable can restore connection")
                    retry_delay = max(retry_delay, 10.0)
                else:
                    logger.warning("WebSocket closed (code=%s): %s", exc.code, exc.reason)
            except Exception as exc:
                logger.warning("WebSocket error: %s", exc)
            finally:
                self._ws = None

            if not self._running:
                break

            logger.info("Reconnecting in %.1f s …", retry_delay)
            await asyncio.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60.0)

    async def shutdown(self) -> None:
        """Drain, deregister, and stop the run loop."""
        self._running = False
        self._stop_event.set()

        # Tell peers we are draining
        if self._ws:
            try:
                drain_msg = self._make_envelope("status_update", {"status": "draining"})
                await self._ws.send(json.dumps(drain_msg))
            except Exception:
                pass

        # Deregister from orchestrator
        if self.agent_id:
            async with httpx.AsyncClient() as client:
                try:
                    await client.delete(
                        f"{self.base_url}/api/v1/agents/{self.agent_id}",
                        timeout=5.0,
                    )
                    logger.info("Deregistered agent %s", self.agent_id)
                except Exception as exc:
                    logger.warning("Deregister failed: %s", exc)

        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_envelope(
        self,
        msg_type: str,
        payload: dict,
        recipient_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
    ) -> dict:
        return {
            "id": str(uuid.uuid4()),
            "type": msg_type,
            "sender_id": self.agent_id,
            "recipient_id": recipient_id,
            "payload": payload,
            "timestamp": _now_iso(),
            "correlation_id": correlation_id,
        }

    async def _heartbeat_loop(self) -> None:
        while True:
            uptime = time.monotonic() - self._start_time
            hb = self._make_envelope(
                "heartbeat",
                {
                    "status": "busy" if self._active_tasks > 0 else "available",
                    "current_load": min(self._active_tasks / 10.0, 1.0),
                    "active_tasks": self._active_tasks,
                    "metrics": {
                        "tasks_completed": self._tasks_completed,
                        "tasks_failed": self._tasks_failed,
                        "uptime_seconds": round(uptime, 1),
                    },
                },
            )
            await self._ws.send(json.dumps(hb))
            logger.debug("Heartbeat sent (load=%.2f)", hb["payload"]["current_load"])
            await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _recv_loop(self) -> None:
        logger.debug("_recv_loop started")
        msg_count = 0
        async for raw in self._ws:
            msg_count += 1
            logger.debug("_recv_loop: received frame #%d (%d bytes)", msg_count, len(raw))
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Received non-JSON frame, ignoring")
                continue

            msg_type = msg.get("type")

            if msg_type == "task_response":
                corr_id = msg.get("correlation_id")
                if corr_id and corr_id in self._pending:
                    fut = self._pending.pop(corr_id)
                    if not fut.done():
                        fut.set_result(msg["payload"])
                else:
                    logger.debug("Received task_response with no pending future (correlation_id=%s)", corr_id)

            elif msg_type == "task_request":
                asyncio.create_task(self._dispatch_task(msg))

            elif msg_type == "error":
                err = msg.get("payload", {})
                original_id = err.get("original_message_id")
                code = err.get("code", "UNKNOWN")
                detail = err.get("detail", "")
                logger.error("Orchestrator error [%s]: %s (original_message_id=%s)", code, detail, original_id)
                # Resolve any pending future that was waiting on this message
                if original_id and original_id in self._pending:
                    fut = self._pending.pop(original_id)
                    if not fut.done():
                        fut.set_exception(RuntimeError(f"{code}: {detail}"))

            elif msg_type == "discovery_response":
                corr_id = msg.get("correlation_id")
                if corr_id and corr_id in self._pending:
                    fut = self._pending.pop(corr_id)
                    if not fut.done():
                        fut.set_result(msg["payload"])

            elif msg_type == "settings_push":
                pushed = msg.get("payload", {}).get("settings", {})
                if pushed:
                    self.common_settings.update(pushed)
                    logger.info("Settings push received: %s", list(pushed.keys()))
                    for handler in self._settings_handlers:
                        asyncio.create_task(handler(self.common_settings.copy()))

            elif msg_type == "agent_registered":
                logger.info("New agent joined: %s", msg.get("payload", {}).get("agent_id"))

            elif msg_type == "agent_offline":
                agent_id = msg.get("payload", {}).get("agent_id")
                reason = msg.get("payload", {}).get("reason", "unknown")
                logger.info("Agent went offline: %s (%s)", agent_id, reason)
                # Fail any pending tasks targeted at the offline agent (best-effort)
                for req_id, fut in list(self._pending.items()):
                    if not fut.done():
                        fut.set_exception(RuntimeError(f"Agent {agent_id} went offline ({reason})"))
                        self._pending.pop(req_id, None)

            elif msg_type == "agent_restart":
                logger.info("Restart requested by orchestrator — shutting down for restart")
                import asyncio as _asyncio, sys
                _asyncio.get_event_loop().call_later(1.0, lambda: sys.exit(0))

            else:
                logger.debug("Unhandled message type: %s", msg_type)

        # The async-for loop ends when the server closes the connection
        ws_state = getattr(self._ws, "state", None)
        close_code = getattr(self._ws, "close_code", None)
        close_reason = getattr(self._ws, "close_reason", None)
        logger.info(
            "_recv_loop exited after %d message(s) — ws.state=%s close_code=%s close_reason=%r",
            msg_count, ws_state, close_code, close_reason,
        )

    async def _dispatch_task(self, msg: dict) -> None:
        """Call registered task handlers and send a task_response."""
        self._active_tasks += 1
        start = time.monotonic()
        capability = msg.get("payload", {}).get("capability")

        try:
            for handler in self._task_handlers:
                result = await handler(msg)
                if result is not None:
                    duration_ms = (time.monotonic() - start) * 1000
                    response = self._make_envelope(
                        "task_response",
                        {
                            "success": True,
                            "output_data": result,
                            "duration_ms": round(duration_ms, 2),
                        },
                        recipient_id=msg.get("sender_id"),
                        correlation_id=msg.get("id"),
                    )
                    await self._ws.send(json.dumps(response))
                    self._tasks_completed += 1
                    logger.info("Handled task (capability=%s, %.1f ms)", capability, duration_ms)
                    return

            # No handler matched — respond with a failure
            raise ValueError(f"No handler registered for capability '{capability}'")

        except Exception as exc:
            duration_ms = (time.monotonic() - start) * 1000
            logger.error("Task handler error (capability=%s): %s", capability, exc)
            error_response = self._make_envelope(
                "task_response",
                {
                    "success": False,
                    "error": str(exc),
                    "duration_ms": round(duration_ms, 2),
                },
                recipient_id=msg.get("sender_id"),
                correlation_id=msg.get("id"),
            )
            try:
                await self._ws.send(json.dumps(error_response))
            except Exception:
                pass
            self._tasks_failed += 1

        finally:
            self._active_tasks = max(0, self._active_tasks - 1)
