"""Fail closed: tests cannot use the actual HTTP transport or real API key."""

import httpx
import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "offline-test-key")

    def denied(*args, **kwargs):
        raise AssertionError("Live network access is forbidden in tests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", denied)
