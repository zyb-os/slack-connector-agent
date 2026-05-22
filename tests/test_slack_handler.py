"""
tests/test_slack_handler.py — Tests for slack_handler.py helpers and routing.

Coverage:
  _strip_mention      — pure string helper
  _agent_list_text    — formatting helper
  _pick_capability    — capability selection helper
  _route_to_agents    — async routing function (orchestrator + router mocked)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator_client import AGENT_NAME
from router import AgentRouter, RouterMode
from slack_handler import (
    _agent_list_text,
    _pick_capability,
    _route_to_agents,
    _strip_mention,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent(
    agent_id: str = "a1",
    name: str = "test-agent",
    capabilities: list[str] | None = None,
    status: str = "available",
    current_load: float = 0.1,
) -> dict:
    return {
        "agent_id": agent_id,
        "name": name,
        "capabilities": capabilities if capabilities is not None else ["execute_task"],
        "status": status,
        "current_load": current_load,
        "score": current_load,
    }


def make_router(mode: RouterMode = RouterMode.HYBRID) -> MagicMock:
    """Return a mock AgentRouter (kept for API compatibility)."""
    router = MagicMock(spec=AgentRouter)
    router.mode = mode
    router.pick_agent = AsyncMock()
    return router


def make_orchestrator(
    agents: list[dict],
    self_id: str = "self-id",
    is_connected: bool = True,
) -> MagicMock:
    orc = AsyncMock()
    orc.agent_id = self_id
    orc.is_connected = is_connected
    orc.discover_agents = AsyncMock(return_value=agents)
    return orc


async def collect_replies(*args, **kwargs) -> list[str]:
    """Helper that captures reply() calls; returns message list."""
    messages: list[str] = []

    async def reply(**kw):
        messages.append(kw.get("text", ""))

    return messages, reply


# ---------------------------------------------------------------------------
# _strip_mention
# ---------------------------------------------------------------------------

class TestStripMention:
    def test_removes_leading_mention(self):
        assert _strip_mention("<@U012AB3CD> hello bot") == "hello bot"

    def test_preserves_text_without_mention(self):
        assert _strip_mention("just a message") == "just a message"

    def test_handles_mention_with_extra_spaces(self):
        assert _strip_mention("  <@U012AB3CD>   do something  ") == "do something"

    def test_empty_string(self):
        assert _strip_mention("") == ""

    def test_mention_only_returns_empty(self):
        assert _strip_mention("<@U012AB3CD>") == ""


# ---------------------------------------------------------------------------
# _agent_list_text
# ---------------------------------------------------------------------------

class TestAgentListText:
    def test_no_agents_returns_warning(self):
        result = _agent_list_text([])
        assert ":warning:" in result

    def test_lists_agent_names(self):
        agents = [make_agent("a1", "my-bot")]
        result = _agent_list_text(agents)
        assert "my-bot" in result

    def test_shows_capabilities(self):
        agents = [make_agent("a1", capabilities=["web_search", "summarise"])]
        result = _agent_list_text(agents)
        assert "web_search" in result
        assert "summarise" in result

    def test_shows_multiple_agents(self):
        agents = [make_agent("a1", "bot-one"), make_agent("a2", "bot-two")]
        result = _agent_list_text(agents)
        assert "bot-one" in result
        assert "bot-two" in result

    def test_available_status_shows_green_circle(self):
        agents = [make_agent("a1", status="available")]
        result = _agent_list_text(agents)
        assert ":large_green_circle:" in result

    def test_busy_status_shows_yellow_circle(self):
        agents = [make_agent("a1", status="busy")]
        result = _agent_list_text(agents)
        assert ":large_yellow_circle:" in result


# ---------------------------------------------------------------------------
# _pick_capability
# ---------------------------------------------------------------------------

class TestPickCapability:
    def test_returns_preferred_execute_task(self):
        a = make_agent(capabilities=["execute_task", "custom"])
        assert _pick_capability(a) == "execute_task"

    def test_falls_back_to_first_declared(self):
        a = make_agent(capabilities=["my_special_cap"])
        assert _pick_capability(a) == "my_special_cap"

    def test_no_capabilities_returns_default(self):
        a = make_agent(capabilities=[])
        assert _pick_capability(a) == "execute_task"


# ---------------------------------------------------------------------------
# _route_to_agents
# ---------------------------------------------------------------------------

class TestRouteToAgents:
    async def _run(
        self,
        *,
        text: str = "do something",
        agents: list[dict] | None = None,
        self_id: str = "self-id",
        is_connected: bool = True,
        router_result: tuple | None = None,
        send_task_result: dict | None = None,
        send_task_exc: Exception | None = None,
        discover_exc: Exception | None = None,
        thread_ts: str | None = None,
    ) -> list[str]:
        """Run _route_to_agents and return all reply messages."""
        if agents is None:
            agents = [make_agent("planner-1", "task-planner-agent", capabilities=["plan_task"])]

        orc = make_orchestrator(agents, self_id=self_id, is_connected=is_connected)
        if discover_exc:
            orc.discover_agents = AsyncMock(side_effect=discover_exc)

        router = make_router()
        if router_result is None:
            # Default: pick the first non-self agent
            non_self = [a for a in agents if a["agent_id"] != self_id]
            if non_self:
                router.pick_agent.return_value = (non_self[0], "execute_task")
        else:
            router.pick_agent.return_value = router_result

        if send_task_exc:
            orc.send_task = AsyncMock(side_effect=send_task_exc)
        elif send_task_result is not None:
            orc.send_task = AsyncMock(return_value=send_task_result)
        else:
            orc.send_task = AsyncMock(return_value={"success": True, "output_data": {"result": "done"}})

        messages: list[str] = []

        async def reply(**kwargs):
            messages.append(kwargs.get("text", ""))

        await _route_to_agents(
            text=text,
            orchestrator=orc,
            reply=reply,
            router=router,
            thread_ts=thread_ts,
        )
        return messages

    async def test_discover_failure_posts_error(self):
        messages = await self._run(discover_exc=RuntimeError("network down"))
        assert any(":x:" in m for m in messages)
        assert any("orchestrator" in m.lower() for m in messages)

    async def test_no_agents_posts_warning(self):
        # All agents are self — after filtering, list is empty
        messages = await self._run(
            agents=[make_agent("self-id")],
            self_id="self-id",
        )
        assert any(":warning:" in m for m in messages)

    async def test_routing_announcement_targets_task_planner(self):
        messages = await self._run(
            agents=[
                make_agent("a1", "my-bot"),
                make_agent("planner-1", "task-planner-agent", capabilities=["plan_task"]),
            ],
        )
        routing_msg = next(m for m in messages if ":arrows_counterclockwise:" in m)
        assert "task-planner-agent" in routing_msg
        assert "plan_task" in routing_msg

    async def test_success_result_posted(self):
        messages = await self._run(
            send_task_result={"success": True, "output_data": {"result": "the answer"}}
        )
        assert any("the answer" in m for m in messages)
        assert any(":white_check_mark:" in m for m in messages)

    async def test_task_failure_posted(self):
        messages = await self._run(
            send_task_result={"success": False, "error": "something went wrong"}
        )
        assert any("something went wrong" in m for m in messages)
        assert any(":x:" in m for m in messages)

    async def test_timeout_posts_timeout_message(self):
        messages = await self._run(send_task_exc=TimeoutError())
        assert any(":clock1:" in m for m in messages)

    async def test_runtime_error_posts_orchestrator_error(self):
        messages = await self._run(send_task_exc=RuntimeError("AGENT_UNAVAILABLE"))
        assert any("AGENT_UNAVAILABLE" in m for m in messages)

    async def test_missing_planner_posts_warning(self):
        messages = await self._run(agents=[make_agent("a1", "bot")])
        assert any("task-planner-agent" in m for m in messages)
        assert any(":warning:" in m for m in messages)

    async def test_input_data_is_planner_payload_with_slack_context(self):
        agents = [make_agent("planner-1", "task-planner-agent", capabilities=["plan_task"])]
        orc = make_orchestrator(agents, self_id="self")
        router = make_router()
        orc.send_task = AsyncMock(
            return_value={"success": True, "output_data": {"result": "ok"}}
        )

        async def reply(**kwargs):
            pass

        await _route_to_agents(
            text="search for cats",
            orchestrator=orc,
            reply=reply,
            router=router,
        )

        _, call_kwargs = orc.send_task.call_args
        sent = call_kwargs["input_data"]
        assert call_kwargs["capability"] == "plan_task"
        assert sent["goal"] == "search for cats"
        assert sent["auto_execute"] is True
        assert sent["payload"]["source"] == "slack"
        assert sent["payload"]["text"] == "search for cats"

    async def test_prefers_available_task_planner_when_multiple_exist(self):
        agents = [
            make_agent("planner-busy", "task-planner-agent", capabilities=["plan_task"], status="busy", current_load=0.9),
            make_agent("planner-avail", "task-planner-agent", capabilities=["plan_task"], status="available", current_load=0.2),
        ]
        orc = make_orchestrator(agents, self_id="self-id")
        router = make_router()
        orc.send_task = AsyncMock(
            return_value={"success": True, "output_data": {"result": "ok"}}
        )

        async def reply(**kwargs):
            pass

        await _route_to_agents(
            text="anything",
            orchestrator=orc,
            reply=reply,
            router=router,
        )

        _, call_kwargs = orc.send_task.call_args
        assert call_kwargs["target_agent_id"] == "planner-avail"

    async def test_output_data_fields_tried_in_order(self):
        """_route_to_agents should surface 'result', 'response', etc. from output_data."""
        for key, value in [
            ("result", "result-value"),
            ("response", "response-value"),
            ("message", "message-value"),
            ("summary", "summary-value"),
        ]:
            messages = await self._run(
                send_task_result={"success": True, "output_data": {key: value}}
            )
            assert any(value in m for m in messages), f"Expected {key!r} value in reply"

    # ------------------------------------------------------------------
    # Connectivity gate (Bug fix: "WebSocket is not connected")
    # ------------------------------------------------------------------

    async def test_not_connected_posts_friendly_message_and_skips_routing(self):
        """When WS is down, _route_to_agents must not attempt routing."""
        agents = [make_agent("a1", "agent-one")]
        orc = make_orchestrator(agents, is_connected=False)
        router = make_router()

        messages: list[str] = []

        async def reply(**kwargs):
            messages.append(kwargs.get("text", ""))

        await _route_to_agents(
            text="do something",
            orchestrator=orc,
            reply=reply,
            router=router,
        )

        assert any(":hourglass_flowing_sand:" in m for m in messages)
        # discover_agents must NOT have been called
        orc.discover_agents.assert_not_called()

    async def test_ws_disconnected_mid_request_shows_reconnecting_message(self):
        """RuntimeError('WebSocket is not connected') should give a specific message."""
        messages = await self._run(
            send_task_exc=RuntimeError("WebSocket is not connected")
        )
        assert any(":hourglass_flowing_sand:" in m for m in messages)
        # Must NOT show the generic "Orchestrator error" label
        assert not any("Orchestrator error" in m for m in messages)

    async def test_other_runtime_error_still_shows_orchestrator_error(self):
        """Non-WS RuntimeErrors should still show the 'Orchestrator error' label."""
        messages = await self._run(
            send_task_exc=RuntimeError("AGENT_UNAVAILABLE: peer offline")
        )
        assert any("Orchestrator error" in m for m in messages)
        assert any("AGENT_UNAVAILABLE" in m for m in messages)

    # ------------------------------------------------------------------
    # Stale-record filter (Bug fix: routing to own stale record)
    # ------------------------------------------------------------------

    async def test_stale_self_record_excluded_by_name(self):
        """An agent named AGENT_NAME with a *different* agent_id (stale record from a
        previous run) must be filtered out even though the id comparison would keep it."""
        stale = make_agent("stale-uuid", AGENT_NAME, capabilities=["send_slack_message"])
        other = make_agent("other-id", "task-planner-agent", capabilities=["plan_task"])
        # orchestrator.agent_id = "current-uuid" (different from stale)
        messages = await self._run(
            agents=[stale, other],
            self_id="current-uuid",
        )
        # Routing should go to task-planner-agent, not to AGENT_NAME
        routing_msg = next((m for m in messages if ":arrows_counterclockwise:" in m), "")
        assert "task-planner-agent" in routing_msg
        assert AGENT_NAME not in routing_msg

    async def test_stale_self_record_alone_shows_no_agents_warning(self):
        """If the only agent in discover is a stale self-record, show the warning."""
        stale = make_agent("stale-uuid", AGENT_NAME, capabilities=["send_slack_message"])
        messages = await self._run(
            agents=[stale],
            self_id="current-uuid",  # different from stale, so id filter keeps it
        )
        # Name filter should catch it; result: empty list → warning
        assert any(":warning:" in m for m in messages)
