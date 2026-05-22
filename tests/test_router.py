"""
tests/test_router.py — Unit tests for router.py (AgentRouter).

Coverage:
  RouterMode enum
  AgentRouter.__init__ validation
  _tokenise / _jaccard helpers
  _fallback_capability
  _load_based_pick
  _score_agent
  _keyword_pick
  _llm_pick  (Anthropic client mocked)
  pick_agent  (all three modes + fallback invariant)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from router import AgentRouter, RouterMode, _fallback_capability, PREFERRED_CAPABILITIES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_agent(
    agent_id: str = "a1",
    name: str = "agent-1",
    capabilities: list[str] | None = None,
    tags: list[str] | None = None,
    description: str = "",
    status: str = "available",
    score: float = 0.1,
) -> dict:
    return {
        "agent_id": agent_id,
        "name": name,
        "capabilities": capabilities if capabilities is not None else ["execute_task"],
        "tags": tags or [],
        "description": description,
        "status": status,
        "score": score,
    }


def make_llm_client(json_text: str) -> MagicMock:
    """Return a mock AsyncAnthropic client whose messages.create returns *json_text*."""
    content = MagicMock()
    content.text = json_text
    response = MagicMock()
    response.content = [content]
    client = MagicMock()
    client.messages.create = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# RouterMode
# ---------------------------------------------------------------------------

class TestRouterMode:
    def test_values_are_strings(self):
        assert RouterMode.KEYWORD.value == "keyword"
        assert RouterMode.LLM.value == "llm"
        assert RouterMode.HYBRID.value == "hybrid"

    def test_is_str_subclass(self):
        # Enum inherits from str, so equality with plain strings works
        assert RouterMode.KEYWORD == "keyword"
        assert RouterMode.LLM == "llm"
        assert RouterMode.HYBRID == "hybrid"


# ---------------------------------------------------------------------------
# AgentRouter.__init__
# ---------------------------------------------------------------------------

class TestAgentRouterInit:
    def test_llm_mode_without_key_raises(self):
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AgentRouter(mode=RouterMode.LLM, anthropic_api_key=None)

    def test_keyword_mode_without_key_is_fine(self):
        r = AgentRouter(mode=RouterMode.KEYWORD, anthropic_api_key=None)
        assert r._client is None

    def test_hybrid_mode_without_key_is_fine(self):
        r = AgentRouter(mode=RouterMode.HYBRID, anthropic_api_key=None)
        assert r._client is None

    def test_mode_and_threshold_stored(self):
        r = AgentRouter(mode=RouterMode.KEYWORD, confidence_threshold=0.75)
        assert r.mode == RouterMode.KEYWORD
        assert r.confidence_threshold == 0.75

    def test_with_api_key_creates_anthropic_client(self):
        mock_instance = MagicMock()
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = mock_instance

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            r = AgentRouter(mode=RouterMode.HYBRID, anthropic_api_key="sk-ant-test")

        assert r._client is mock_instance
        mock_anthropic.AsyncAnthropic.assert_called_once_with(api_key="sk-ant-test")

    def test_llm_mode_with_key_does_not_raise(self):
        mock_anthropic = MagicMock()
        mock_anthropic.AsyncAnthropic.return_value = MagicMock()

        with patch.dict("sys.modules", {"anthropic": mock_anthropic}):
            r = AgentRouter(mode=RouterMode.LLM, anthropic_api_key="sk-ant-test")

        assert r._client is not None


# ---------------------------------------------------------------------------
# _tokenise
# ---------------------------------------------------------------------------

class TestTokenise:
    def test_lowercases_input(self):
        tokens = AgentRouter._tokenise("Web Search")
        assert "web" in tokens
        assert "search" in tokens

    def test_drops_tokens_shorter_than_three_chars(self):
        # "do" (2), "a" (1), "go" (2) should be dropped
        tokens = AgentRouter._tokenise("do a go run task")
        assert "do" not in tokens
        assert "a" not in tokens
        assert "go" not in tokens
        assert "run" in tokens

    def test_removes_stop_words(self):
        tokens = AgentRouter._tokenise("search the web for information")
        assert "the" not in tokens
        assert "for" not in tokens
        assert "search" in tokens
        assert "web" in tokens
        assert "information" in tokens

    def test_splits_on_underscores_and_hyphens(self):
        tokens = AgentRouter._tokenise("web_search-agent")
        assert "web" in tokens
        assert "search" in tokens
        assert "agent" in tokens

    def test_returns_frozenset(self):
        result = AgentRouter._tokenise("hello world")
        assert isinstance(result, frozenset)

    def test_empty_string_returns_empty_frozenset(self):
        assert AgentRouter._tokenise("") == frozenset()

    def test_deduplicates(self):
        tokens = AgentRouter._tokenise("search search search")
        assert len(tokens) == 1


# ---------------------------------------------------------------------------
# _jaccard
# ---------------------------------------------------------------------------

class TestJaccard:
    def test_both_empty_returns_zero(self):
        assert AgentRouter._jaccard(frozenset(), frozenset()) == 0.0

    def test_identical_sets_return_one(self):
        s = frozenset({"alpha", "beta", "gamma"})
        assert AgentRouter._jaccard(s, s) == 1.0

    def test_disjoint_sets_return_zero(self):
        a = frozenset({"foo", "bar"})
        b = frozenset({"baz", "qux"})
        assert AgentRouter._jaccard(a, b) == 0.0

    def test_partial_overlap(self):
        a = frozenset({"a", "b", "c"})
        b = frozenset({"b", "c", "d"})
        # intersection=2, union=4 → 0.5
        assert AgentRouter._jaccard(a, b) == pytest.approx(0.5)

    def test_one_empty_returns_zero(self):
        assert AgentRouter._jaccard(frozenset({"x"}), frozenset()) == 0.0


# ---------------------------------------------------------------------------
# _fallback_capability
# ---------------------------------------------------------------------------

class TestFallbackCapability:
    def test_prefers_execute_task(self):
        a = make_agent(capabilities=["custom_cap", "execute_task", "run_task"])
        assert _fallback_capability(a) == "execute_task"

    def test_respects_preferred_order(self):
        # handle_instruction is second in PREFERRED_CAPABILITIES
        a = make_agent(capabilities=["handle_instruction", "handle_task"])
        assert _fallback_capability(a) == "handle_instruction"

    def test_falls_back_to_first_declared(self):
        a = make_agent(capabilities=["my_unique_capability"])
        assert _fallback_capability(a) == "my_unique_capability"

    def test_no_capabilities_returns_execute_task(self):
        a = make_agent(capabilities=[])
        assert _fallback_capability(a) == "execute_task"

    def test_all_preferred_capabilities_are_tried(self):
        # Verify every entry in PREFERRED_CAPABILITIES is handled
        for cap in PREFERRED_CAPABILITIES:
            a = make_agent(capabilities=[cap])
            assert _fallback_capability(a) == cap


# ---------------------------------------------------------------------------
# _load_based_pick
# ---------------------------------------------------------------------------

class TestLoadBasedPick:
    def test_available_beats_busy(self):
        agents = [
            make_agent("busy", status="busy", score=0.05),
            make_agent("avail", status="available", score=0.95),
        ]
        picked, _ = AgentRouter._load_based_pick(agents)
        assert picked["agent_id"] == "avail"

    def test_lower_score_wins_within_same_status(self):
        agents = [
            make_agent("heavy", status="available", score=0.8),
            make_agent("light", status="available", score=0.2),
        ]
        picked, _ = AgentRouter._load_based_pick(agents)
        assert picked["agent_id"] == "light"

    def test_unknown_status_treated_as_worst(self):
        agents = [
            make_agent("starting", status="starting", score=0.0),
            make_agent("avail", status="available", score=0.9),
        ]
        picked, _ = AgentRouter._load_based_pick(agents)
        assert picked["agent_id"] == "avail"

    def test_uses_fallback_capability(self):
        agents = [make_agent("a1", capabilities=["execute_task", "custom"])]
        _, cap = AgentRouter._load_based_pick(agents)
        assert cap == "execute_task"

    def test_single_agent_returned(self):
        agents = [make_agent("only")]
        picked, _ = AgentRouter._load_based_pick(agents)
        assert picked["agent_id"] == "only"

    def test_does_not_mutate_input_list(self):
        agents = [
            make_agent("b", status="busy"),
            make_agent("a", status="available"),
        ]
        original_order = [ag["agent_id"] for ag in agents]
        AgentRouter._load_based_pick(agents)
        assert [ag["agent_id"] for ag in agents] == original_order


# ---------------------------------------------------------------------------
# _score_agent
# ---------------------------------------------------------------------------

class TestScoreAgent:
    def setup_method(self):
        self.router = AgentRouter(mode=RouterMode.KEYWORD)

    def test_matches_capability_name_tokens(self):
        a = make_agent(capabilities=["web_search"])
        tokens = AgentRouter._tokenise("search the web")
        score, cap = self.router._score_agent(tokens, a)
        assert score > 0.0
        assert cap == "web_search"

    def test_matches_agent_tags(self):
        a = make_agent(capabilities=["do_stuff"], tags=["web", "search"])
        tokens = AgentRouter._tokenise("search the web")
        score, _ = self.router._score_agent(tokens, a)
        assert score > 0.0

    def test_matches_description(self):
        a = make_agent(capabilities=["process"], description="Summarise news articles")
        tokens = AgentRouter._tokenise("summarise news")
        score, _ = self.router._score_agent(tokens, a)
        assert score > 0.0

    def test_no_match_returns_zero(self):
        a = make_agent(capabilities=["image_resize"], tags=["image"])
        tokens = AgentRouter._tokenise("write a poem")
        score, _ = self.router._score_agent(tokens, a)
        assert score == 0.0

    def test_returns_best_capability_when_multiple(self):
        a = make_agent(capabilities=["image_resize", "web_search"])
        tokens = AgentRouter._tokenise("search the web")
        score, cap = self.router._score_agent(tokens, a)
        assert cap == "web_search"
        assert score > 0.0

    def test_returns_fallback_capability_on_zero_score(self):
        a = make_agent(capabilities=["execute_task"])
        tokens = AgentRouter._tokenise("zzzzz bbbbb")
        _, cap = self.router._score_agent(tokens, a)
        assert cap == "execute_task"


# ---------------------------------------------------------------------------
# _keyword_pick
# ---------------------------------------------------------------------------

class TestKeywordPick:
    def setup_method(self):
        self.router = AgentRouter(mode=RouterMode.KEYWORD)

    def test_picks_agent_with_matching_capability(self):
        agents = [
            make_agent("search", capabilities=["web_search"], tags=["web", "search"]),
            make_agent("image", capabilities=["image_resize"], tags=["image"]),
        ]
        picked, cap = self.router._keyword_pick("search the web for news", agents)
        assert picked["agent_id"] == "search"
        assert cap == "web_search"

    def test_all_zero_scores_falls_back_to_load_based(self):
        agents = [
            make_agent("a1", capabilities=["xyzzy_cap"], status="busy", score=0.9),
            make_agent("a2", capabilities=["execute_task"], status="available", score=0.1),
        ]
        picked, _ = self.router._keyword_pick("make me a coffee", agents)
        # Load-based: available beats busy
        assert picked["agent_id"] == "a2"

    def test_tiebreak_by_availability(self):
        # Both agents score identically; the available one should win
        agents = [
            make_agent("busy-one", capabilities=["execute_task"], tags=["task"],
                       status="busy", score=0.1),
            make_agent("free-one", capabilities=["execute_task"], tags=["task"],
                       status="available", score=0.9),
        ]
        picked, _ = self.router._keyword_pick("execute a task", agents)
        assert picked["agent_id"] == "free-one"

    def test_tiebreak_by_score_within_same_status(self):
        agents = [
            make_agent("heavy", capabilities=["execute_task"], tags=["task"],
                       status="available", score=0.8),
            make_agent("light", capabilities=["execute_task"], tags=["task"],
                       status="available", score=0.2),
        ]
        picked, _ = self.router._keyword_pick("execute a task", agents)
        assert picked["agent_id"] == "light"


# ---------------------------------------------------------------------------
# _llm_pick  (async — client is mocked directly on self.router)
# ---------------------------------------------------------------------------

class TestLlmPick:
    def setup_method(self):
        # Build a router without any real API key; we inject the client manually
        self.router = AgentRouter(mode=RouterMode.KEYWORD)

    async def test_raises_when_no_client(self):
        self.router._client = None
        with pytest.raises(RuntimeError, match="not initialised"):
            await self.router._llm_pick("do something", [make_agent()])

    async def test_success_path(self):
        agents = [make_agent("a1", name="bot", capabilities=["web_search"])]
        self.router._client = make_llm_client('{"agent_id": "a1", "capability": "web_search"}')
        picked, cap = await self.router._llm_pick("search the web", agents)
        assert picked["agent_id"] == "a1"
        assert cap == "web_search"

    async def test_strips_markdown_fences(self):
        agents = [make_agent("a1", capabilities=["execute_task"])]
        self.router._client = make_llm_client(
            "```json\n"
            '{"agent_id": "a1", "capability": "execute_task"}\n'
            "```"
        )
        picked, cap = await self.router._llm_pick("run a task", agents)
        assert picked["agent_id"] == "a1"
        assert cap == "execute_task"

    async def test_unknown_agent_id_raises(self):
        agents = [make_agent("a1")]
        self.router._client = make_llm_client(
            '{"agent_id": "does-not-exist", "capability": "execute_task"}'
        )
        with pytest.raises(ValueError, match="unknown agent_id"):
            await self.router._llm_pick("do something", agents)

    async def test_hallucinated_capability_uses_fallback(self):
        agents = [make_agent("a1", capabilities=["execute_task"])]
        self.router._client = make_llm_client(
            '{"agent_id": "a1", "capability": "hallucinated_cap_xyz"}'
        )
        picked, cap = await self.router._llm_pick("do something", agents)
        assert picked["agent_id"] == "a1"
        assert cap == "execute_task"  # _fallback_capability result

    async def test_invalid_json_raises(self):
        agents = [make_agent("a1")]
        self.router._client = make_llm_client("not valid json at all")
        with pytest.raises(Exception):
            await self.router._llm_pick("something", agents)

    async def test_passes_all_agents_to_model(self):
        agents = [
            make_agent("a1", name="bot-1", capabilities=["cap_a"]),
            make_agent("a2", name="bot-2", capabilities=["cap_b"]),
        ]
        self.router._client = make_llm_client('{"agent_id": "a1", "capability": "cap_a"}')
        await self.router._llm_pick("do cap_a work", agents)

        call_kwargs = self.router._client.messages.create.call_args
        prompt_text = call_kwargs.kwargs["messages"][0]["content"]
        assert "a1" in prompt_text
        assert "a2" in prompt_text


# ---------------------------------------------------------------------------
# pick_agent — public entry point
# ---------------------------------------------------------------------------

class TestPickAgent:
    async def test_keyword_mode_routes_correctly(self):
        r = AgentRouter(mode=RouterMode.KEYWORD)
        agents = [
            make_agent("search", capabilities=["web_search"], tags=["web", "search"]),
            make_agent("other", capabilities=["image_resize"], tags=["image"]),
        ]
        picked, cap = await r.pick_agent("search the web", agents)
        assert picked["agent_id"] == "search"
        assert cap == "web_search"

    async def test_llm_mode_routes_via_client(self):
        r = AgentRouter(mode=RouterMode.KEYWORD)
        r.mode = RouterMode.LLM  # bypass init validation
        agents = [make_agent("a1", capabilities=["execute_task"])]
        r._client = make_llm_client('{"agent_id": "a1", "capability": "execute_task"}')

        picked, cap = await r.pick_agent("run something", agents)
        assert picked["agent_id"] == "a1"
        assert cap == "execute_task"

    async def test_hybrid_above_threshold_skips_llm(self):
        r = AgentRouter(mode=RouterMode.HYBRID, confidence_threshold=0.1)
        agents = [make_agent("search", capabilities=["web_search"], tags=["web", "search"])]
        # Inject a mock client that must NOT be called
        spy = AsyncMock()
        r._client = MagicMock()
        r._client.messages.create = spy

        await r.pick_agent("web search", agents)

        spy.assert_not_called()

    async def test_hybrid_below_threshold_calls_llm(self):
        r = AgentRouter(mode=RouterMode.HYBRID, confidence_threshold=0.99)
        agents = [make_agent("a1", capabilities=["execute_task"])]
        r._client = make_llm_client('{"agent_id": "a1", "capability": "execute_task"}')

        picked, _ = await r.pick_agent("do something vague", agents)

        assert picked["agent_id"] == "a1"
        r._client.messages.create.assert_called_once()

    async def test_hybrid_no_client_falls_back_to_load_based(self):
        r = AgentRouter(mode=RouterMode.HYBRID, confidence_threshold=0.99)
        r._client = None
        agents = [
            make_agent("heavy", capabilities=["xyzzy"], status="busy", score=0.9),
            make_agent("light", capabilities=["execute_task"], status="available", score=0.1),
        ]
        # Score will be 0 (instruction has no token overlap); no client → load-based
        picked, _ = await r.pick_agent("completely unrelated instruction here", agents)
        assert picked["agent_id"] == "light"

    async def test_hybrid_llm_failure_falls_back_to_keyword(self):
        r = AgentRouter(mode=RouterMode.HYBRID, confidence_threshold=0.99)
        bad_client = MagicMock()
        bad_client.messages.create = AsyncMock(side_effect=RuntimeError("API unavailable"))
        r._client = bad_client

        agents = [make_agent("search", capabilities=["web_search"], tags=["web", "search"])]
        picked, cap = await r.pick_agent("search the web", agents)

        # Keyword result used as fallback after LLM failure
        assert picked["agent_id"] == "search"
        assert cap == "web_search"

    async def test_never_raises_on_unexpected_internal_error(self):
        """pick_agent must catch any exception and return a load-based result."""
        r = AgentRouter(mode=RouterMode.KEYWORD)
        agents = [make_agent("safe", capabilities=["execute_task"], status="available")]

        # Force _keyword_pick to blow up
        r._keyword_pick = MagicMock(side_effect=RuntimeError("something exploded"))

        picked, _ = await r.pick_agent("anything at all", agents)
        assert picked["agent_id"] == "safe"
