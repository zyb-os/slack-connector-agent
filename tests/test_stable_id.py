"""
tests/test_stable_id.py — Tests for stable agent identity in orchestrator_client.py.

Coverage:
  _stable_agent_id() — creates file on first call, reads on subsequent calls
  OrchestratorClient.register() — includes stable 'id' in the POST payload
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import orchestrator_client as oc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fake_http_client(agent_id: str) -> MagicMock:
    """Return an async context-manager mock for httpx.AsyncClient.

    The mock's .post() returns a response whose .json() yields a minimal
    registration response.
    """
    response = MagicMock()
    response.raise_for_status = MagicMock()
    response.json.return_value = {
        "agent_id": agent_id,
        "ws_url": f"ws://localhost/ws/{agent_id}",
    }

    client = AsyncMock()
    client.__aenter__.return_value = client
    client.__aexit__.return_value = False
    client.post = AsyncMock(return_value=response)
    return client


# ---------------------------------------------------------------------------
# _stable_agent_id
# ---------------------------------------------------------------------------

class TestStableAgentId:
    def test_creates_file_on_first_call(self, tmp_path, monkeypatch):
        id_file = tmp_path / ".agent_id"
        monkeypatch.setattr(oc, "_ID_FILE", id_file)

        oc._stable_agent_id()

        assert id_file.exists()

    def test_written_value_is_valid_uuid(self, tmp_path, monkeypatch):
        id_file = tmp_path / ".agent_id"
        monkeypatch.setattr(oc, "_ID_FILE", id_file)

        result = oc._stable_agent_id()

        # Must not raise
        uuid.UUID(result)

    def test_returns_same_id_on_repeat_calls(self, tmp_path, monkeypatch):
        id_file = tmp_path / ".agent_id"
        monkeypatch.setattr(oc, "_ID_FILE", id_file)

        first = oc._stable_agent_id()
        second = oc._stable_agent_id()

        assert first == second

    def test_reads_pre_existing_file(self, tmp_path, monkeypatch):
        preset = str(uuid.uuid4())
        id_file = tmp_path / ".agent_id"
        id_file.write_text(preset)
        monkeypatch.setattr(oc, "_ID_FILE", id_file)

        result = oc._stable_agent_id()

        assert result == preset

    def test_does_not_overwrite_pre_existing_file(self, tmp_path, monkeypatch):
        preset = str(uuid.uuid4())
        id_file = tmp_path / ".agent_id"
        id_file.write_text(preset)
        monkeypatch.setattr(oc, "_ID_FILE", id_file)

        oc._stable_agent_id()

        assert id_file.read_text().strip() == preset

    def test_strips_trailing_whitespace(self, tmp_path, monkeypatch):
        preset = str(uuid.uuid4())
        id_file = tmp_path / ".agent_id"
        id_file.write_text(preset + "\n")
        monkeypatch.setattr(oc, "_ID_FILE", id_file)

        result = oc._stable_agent_id()

        assert result == preset


# ---------------------------------------------------------------------------
# register() includes stable id
# ---------------------------------------------------------------------------

class TestRegisterPayload:
    async def test_id_field_is_present_in_post_payload(self, tmp_path, monkeypatch):
        preset = str(uuid.uuid4())
        id_file = tmp_path / ".agent_id"
        id_file.write_text(preset)
        monkeypatch.setattr(oc, "_ID_FILE", id_file)

        http_mock = fake_http_client(preset)
        with patch("orchestrator_client.httpx.AsyncClient", return_value=http_mock):
            client = oc.OrchestratorClient()
            await client.register()

        _, call_kwargs = http_mock.post.call_args
        assert call_kwargs["json"]["id"] == preset

    async def test_register_sets_agent_id_from_response(self, tmp_path, monkeypatch):
        preset = str(uuid.uuid4())
        id_file = tmp_path / ".agent_id"
        id_file.write_text(preset)
        monkeypatch.setattr(oc, "_ID_FILE", id_file)

        http_mock = fake_http_client(preset)
        with patch("orchestrator_client.httpx.AsyncClient", return_value=http_mock):
            client = oc.OrchestratorClient()
            returned_id = await client.register()

        assert returned_id == preset
        assert client.agent_id == preset

    async def test_register_stores_ws_url(self, tmp_path, monkeypatch):
        preset = str(uuid.uuid4())
        id_file = tmp_path / ".agent_id"
        id_file.write_text(preset)
        monkeypatch.setattr(oc, "_ID_FILE", id_file)

        http_mock = fake_http_client(preset)
        with patch("orchestrator_client.httpx.AsyncClient", return_value=http_mock):
            client = oc.OrchestratorClient()
            await client.register()

        assert client.ws_url == f"ws://localhost/ws/{preset}"

    async def test_stable_id_reused_across_two_registrations(self, tmp_path, monkeypatch):
        """Calling register() twice must send the same 'id' both times."""
        preset = str(uuid.uuid4())
        id_file = tmp_path / ".agent_id"
        id_file.write_text(preset)
        monkeypatch.setattr(oc, "_ID_FILE", id_file)

        http_mock = fake_http_client(preset)
        with patch("orchestrator_client.httpx.AsyncClient", return_value=http_mock):
            client = oc.OrchestratorClient()
            await client.register()
            await client.register()

        calls = http_mock.post.call_args_list
        assert len(calls) == 2
        id_first = calls[0][1]["json"]["id"]
        id_second = calls[1][1]["json"]["id"]
        assert id_first == id_second == preset
