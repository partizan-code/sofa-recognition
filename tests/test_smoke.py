"""Verify the starter application without contacting external services."""

import main


def test_main(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(main, "load_dotenv", lambda *a, **kw: calls.append("dotenv"))
    monkeypatch.setattr(main, "configure_logging", lambda: calls.append("logging"))
    monkeypatch.setattr("ui.render", lambda: calls.append("ui"))
    assert main.main() == 0
    assert calls == ["dotenv", "logging", "ui"]
