from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from websockets.exceptions import ConnectionClosedError
from websockets.frames import Close

import orchestrator_client as oc


class _FailingConnect:
    async def __aenter__(self):
        raise ConnectionClosedError(Close(4003, "disabled"), None, None)

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.mark.asyncio
async def test_4003_disabled_retries_instead_of_stopping(monkeypatch):
    client = oc.OrchestratorClient()
    client.ws_url = "ws://localhost:8000/ws/fake-agent"

    register_mock = AsyncMock()
    monkeypatch.setattr(client, "register", register_mock)
    monkeypatch.setattr(oc.websockets, "connect", lambda *_a, **_k: _FailingConnect())

    sleep_calls: list[float] = []

    async def fake_sleep(seconds: float):
        sleep_calls.append(seconds)
        client._running = False  # stop after first retry cycle

    monkeypatch.setattr(oc.asyncio, "sleep", fake_sleep)

    await client.connect_and_run()

    assert register_mock.await_count == 0
    assert sleep_calls
    assert sleep_calls[0] >= 10.0
