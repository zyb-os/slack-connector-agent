"""
router.py — Hybrid capability-based & intent-based agent router.

Routing modes (set via ROUTING_MODE env var):
  keyword  — Jaccard-similarity token matching against agent capabilities/tags
  llm      — Claude-based semantic routing (requires ANTHROPIC_API_KEY)
  hybrid   — keyword first; fall back to LLM when confidence is below threshold

Public entry point:
  AgentRouter.pick_agent(instruction, agents) -> (agent_dict, capability_name)

Never raises — all failures fall back to load-based selection.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# Stop-words excluded from keyword tokenisation
_STOP_WORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "do",
        "for", "from", "has", "have", "he", "her", "his", "how", "i",
        "if", "in", "is", "it", "its", "me", "my", "no", "not", "of",
        "on", "or", "our", "she", "so", "that", "the", "their", "them",
        "then", "there", "they", "this", "to", "up", "us", "was", "we",
        "what", "when", "who", "will", "with", "you", "your",
    }
)

# Capabilities tried (in order) when no specific capability is matched.
# Mirrors PREFERRED_CAPABILITIES in slack_handler.py.
PREFERRED_CAPABILITIES = [
    "execute_task",
    "handle_instruction",
    "handle_task",
    "process_request",
    "run_task",
]


# ---------------------------------------------------------------------------
# RouterMode
# ---------------------------------------------------------------------------

class RouterMode(str, Enum):
    KEYWORD = "keyword"
    LLM = "llm"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# AgentRouter
# ---------------------------------------------------------------------------

class AgentRouter:
    """Routes an instruction to the best agent using keyword, LLM, or hybrid logic."""

    def __init__(
        self,
        mode: RouterMode = RouterMode.HYBRID,
        confidence_threshold: float = 0.3,
        anthropic_api_key: str | None = None,
    ) -> None:
        if mode == RouterMode.LLM and not anthropic_api_key:
            raise ValueError(
                "ROUTING_MODE=llm requires ANTHROPIC_API_KEY to be set."
            )

        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self._client: Any = None  # anthropic.AsyncAnthropic, lazily imported

        if anthropic_api_key:
            try:
                import anthropic  # noqa: PLC0415
                self._client = anthropic.AsyncAnthropic(api_key=anthropic_api_key)
                logger.info("AgentRouter: Anthropic client initialised (mode=%s)", mode.value)
            except ImportError:
                logger.error(
                    "AgentRouter: 'anthropic' package not installed. "
                    "Run: pip install 'anthropic>=0.26.0'"
                )
                if mode == RouterMode.LLM:
                    raise

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def pick_agent(
        self, instruction: str, agents: list[dict]
    ) -> tuple[dict, str]:
        """
        Select the best agent and capability for *instruction*.

        Returns (agent_dict, capability_name).
        Never raises — falls back to load-based on any error.
        """
        if not agents:
            raise ValueError("No agents provided to pick_agent()")

        try:
            if self.mode == RouterMode.KEYWORD:
                return self._keyword_pick(instruction, agents)

            if self.mode == RouterMode.LLM:
                return await self._llm_pick(instruction, agents)

            # HYBRID
            return await self._hybrid_pick(instruction, agents)

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "AgentRouter.pick_agent failed (%s); falling back to load-based. Error: %s",
                self.mode.value,
                exc,
            )
            return self._load_based_pick(agents)

    # ------------------------------------------------------------------
    # Keyword scoring
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenise(text: str) -> frozenset[str]:
        tokens = re.split(r"[^a-z0-9]+", text.lower())
        return frozenset(t for t in tokens if len(t) >= 3 and t not in _STOP_WORDS)

    @staticmethod
    def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
        if not a and not b:
            return 0.0
        union = a | b
        return len(a & b) / len(union)

    def _score_agent(
        self, instruction_tokens: frozenset[str], agent: dict
    ) -> tuple[float, str]:
        """
        Return (best_score, best_capability) for this agent.

        Each capability is represented by tokens from:
          cap_name + agent tags + agent description
        """
        tags_tokens = self._tokenise(" ".join(agent.get("tags", [])))
        desc_tokens = self._tokenise(agent.get("description", ""))
        base_tokens = tags_tokens | desc_tokens

        best_score = 0.0
        best_cap = _fallback_capability(agent)

        for cap_name in agent.get("capabilities", []):
            cap_tokens = self._tokenise(cap_name) | base_tokens
            score = self._jaccard(instruction_tokens, cap_tokens)
            if score > best_score:
                best_score = score
                best_cap = cap_name

        return best_score, best_cap

    def _keyword_pick(
        self, instruction: str, agents: list[dict]
    ) -> tuple[dict, str]:
        tokens = self._tokenise(instruction)
        scored: list[tuple[float, dict, str]] = []

        for agent in agents:
            score, cap = self._score_agent(tokens, agent)
            scored.append((score, agent, cap))

        best_score = max(s for s, _, _ in scored)

        if best_score == 0.0:
            logger.debug("AgentRouter keyword: all scores=0, load-based fallback")
            return self._load_based_pick(agents)

        # Among agents tied at best_score, pick the least loaded
        candidates = [(a, cap) for s, a, cap in scored if s == best_score]
        candidates.sort(
            key=lambda ac: (
                {"available": 0, "busy": 1}.get(ac[0].get("status", ""), 2),
                ac[0].get("score", 1.0),
            )
        )
        agent, cap = candidates[0]
        logger.info(
            "AgentRouter keyword: picked %s (cap=%s, score=%.3f)",
            agent.get("name", agent.get("agent_id")),
            cap,
            best_score,
        )
        return agent, cap

    # ------------------------------------------------------------------
    # LLM routing
    # ------------------------------------------------------------------

    async def _llm_pick(
        self, instruction: str, agents: list[dict]
    ) -> tuple[dict, str]:
        if self._client is None:
            raise RuntimeError("Anthropic client not initialised (no API key).")

        # Build a compact catalogue for the prompt
        catalogue: list[dict] = []
        id_map: dict[str, dict] = {}
        for a in agents:
            entry = {
                "agent_id": a["agent_id"],
                "name": a.get("name", a["agent_id"]),
                "capabilities": a.get("capabilities", []),
                "tags": a.get("tags", []),
            }
            if a.get("description"):
                entry["description"] = a["description"]
            catalogue.append(entry)
            id_map[a["agent_id"]] = a

        prompt = (
            "You are an agent router. Given the user instruction and agent catalogue, "
            "choose the BEST agent and capability.\n\n"
            f"Instruction: {instruction}\n\n"
            f"Agents:\n{json.dumps(catalogue, indent=2)}\n\n"
            'Respond with ONLY a JSON object: {"agent_id": "...", "capability": "..."}'
        )

        response = await self._client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()
        # Strip markdown fences if present
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        parsed = json.loads(raw)
        chosen_id = parsed["agent_id"]
        chosen_cap = parsed["capability"]

        if chosen_id not in id_map:
            raise ValueError(f"LLM returned unknown agent_id={chosen_id!r}")

        agent = id_map[chosen_id]
        declared = agent.get("capabilities", [])
        if chosen_cap not in declared:
            logger.warning(
                "AgentRouter LLM: capability %r not declared by %s; using fallback",
                chosen_cap,
                chosen_id,
            )
            chosen_cap = _fallback_capability(agent)

        logger.info(
            "AgentRouter LLM: picked %s (cap=%s)",
            agent.get("name", chosen_id),
            chosen_cap,
        )
        return agent, chosen_cap

    # ------------------------------------------------------------------
    # Hybrid
    # ------------------------------------------------------------------

    async def _hybrid_pick(
        self, instruction: str, agents: list[dict]
    ) -> tuple[dict, str]:
        # Step 1: keyword score
        tokens = self._tokenise(instruction)
        scored = [(self._score_agent(tokens, a), a) for a in agents]
        best_score = max(s for (s, _), _ in scored)

        if best_score >= self.confidence_threshold:
            logger.debug(
                "AgentRouter hybrid: keyword score %.3f >= threshold %.3f, using keyword result",
                best_score,
                self.confidence_threshold,
            )
            return self._keyword_pick(instruction, agents)

        # Step 2: try LLM
        if self._client is None:
            logger.warning(
                "AgentRouter hybrid: keyword score %.3f < threshold %.3f "
                "but ANTHROPIC_API_KEY absent — using keyword/load-based fallback",
                best_score,
                self.confidence_threshold,
            )
            if best_score > 0.0:
                return self._keyword_pick(instruction, agents)
            return self._load_based_pick(agents)

        try:
            logger.debug(
                "AgentRouter hybrid: keyword score %.3f < threshold %.3f, trying LLM",
                best_score,
                self.confidence_threshold,
            )
            return await self._llm_pick(instruction, agents)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "AgentRouter hybrid: LLM fallback failed (%s); using keyword/load-based", exc
            )
            if best_score > 0.0:
                return self._keyword_pick(instruction, agents)
            return self._load_based_pick(agents)

    # ------------------------------------------------------------------
    # Structured input extraction
    # ------------------------------------------------------------------

    async def extract_input_data(
        self,
        instruction: str,
        agent: dict,
        capability: str,
    ) -> dict | None:
        """
        Use the LLM to extract structured input_data for the chosen capability.

        When the discover response includes full capability objects the schema
        is passed to the model for accuracy.  Returns a dict on success, None
        when the LLM is unavailable or parsing fails (caller should fall back
        to generic free-text fields).
        """
        if self._client is None:
            return None

        # Pull input_schema if the agent record carries full capability objects
        schema: dict | None = None
        for cap in agent.get("capabilities", []):
            if isinstance(cap, dict) and cap.get("name") == capability:
                schema = cap.get("input_schema")
                break

        schema_hint = (
            f"\nCapability input schema:\n{json.dumps(schema, indent=2)}"
            if schema
            else ""
        )

        prompt = (
            "Extract structured input_data for an agent capability call.\n\n"
            f"User instruction: {instruction}\n"
            f"Target agent: {agent.get('name', 'unknown')}\n"
            f"Target capability: {capability}"
            f"{schema_hint}\n\n"
            "Rules:\n"
            "- Respond with ONLY a JSON object — no explanation, no markdown fences.\n"
            "- Use ISO 8601 UTC for any datetime fields (e.g. '2026-02-24T17:00:00Z').\n"
            "- Omit fields you cannot determine from the instruction.\n"
            "- If the capability wraps another capability (e.g. schedule_task), "
            "set the inner 'capability' field to the most appropriate capability "
            "name for the described action."
        )

        try:
            response = await self._client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
            logger.info(
                "extract_input_data: capability=%r  extracted=%s",
                capability, json.dumps(result)[:200],
            )
            return result
        except Exception as exc:
            logger.warning(
                "extract_input_data failed for capability %r: %s", capability, exc
            )
            return None

    # ------------------------------------------------------------------
    # Load-based fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _load_based_pick(agents: list[dict]) -> tuple[dict, str]:
        sorted_agents = sorted(
            agents,
            key=lambda a: (
                {"available": 0, "busy": 1}.get(a.get("status", ""), 2),
                a.get("score", 1.0),
            ),
        )
        agent = sorted_agents[0]
        cap = _fallback_capability(agent)
        logger.info(
            "AgentRouter load-based: picked %s (cap=%s)",
            agent.get("name", agent.get("agent_id")),
            cap,
        )
        return agent, cap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fallback_capability(agent: dict) -> str:
    """Pick the best generic capability from the agent's declared list."""
    declared = agent.get("capabilities", [])
    for preferred in PREFERRED_CAPABILITIES:
        if preferred in declared:
            return preferred
    return declared[0] if declared else "execute_task"
