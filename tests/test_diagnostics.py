"""Keep actionable diagnostics while excluding keys, images and raw validation input."""

from functools import partial

import httpx
import pytest
from loguru import logger
from PIL import Image
from pydantic import ValidationError
from streamlit.testing.v1 import AppTest

import config
from log_config import configure_logging
from schemas import Prediction
from storage import write_json
from ui import error_message, show_error
from vision_client import OpenRouterVision


def test_error_details_redact_key_and_image(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "private-test-key")
    message = error_message(
        ValueError(
            "Vision: finish_reason='length' private-test-key "
            "Bearer another-secret data:image/jpeg;base64,YWJjZA=="
        )
    )
    assert "finish_reason='length'" in message
    assert not any(value in message for value in ["private-test-key", "another-secret", "YWJjZA=="])


def test_validation_details_omit_input():
    with pytest.raises(ValidationError) as caught:
        Prediction.model_validate(
            {
                "model": None,
                "color": None,
                "configuration": "private-invalid-value",
                "orientation": None,
            }
        )
    message = error_message(caught.value)
    assert "configuration: literal_error" in message
    assert "private-invalid-value" not in message


def test_http_details_omit_body_and_request_headers():
    request = httpx.Request(
        "POST",
        "https://example.test/embeddings",
        headers={"Authorization": "Bearer private-header"},
    )
    response = httpx.Response(400, request=request, json={"error": "private-body"})
    message = error_message(
        httpx.HTTPStatusError("private-message", request=request, response=response)
    )
    assert "Embeddings: HTTP 400" in message
    assert "private" not in message


def test_warning_is_saved_with_context(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "ROOT", tmp_path)
    displayed = []
    monkeypatch.setattr("ui.st.error", displayed.append)
    try:
        configure_logging()
        show_error(ValueError("Индекс Voyage отсутствует"), "P0001.webp: Voyage Embeddings")
    finally:
        logger.remove()
    log = (tmp_path / "logs/errors.log").read_text(encoding="utf-8")
    assert "P0001.webp: Voyage Embeddings" in log
    assert displayed[0] in log


def test_unfinished_vision_reports_reason_without_caching(monkeypatch, tmp_path):
    def handler(request):
        return httpx.Response(200, json={"choices": [{"finish_reason": "length"}]})

    monkeypatch.setattr(
        httpx, "Client", partial(httpx.Client, transport=httpx.MockTransport(handler))
    )
    image = tmp_path / "sofa.png"
    Image.new("RGB", (4, 4)).save(image)
    with OpenRouterVision(tmp_path / "cache") as client:
        with pytest.raises(ValueError, match="finish_reason='length'"):
            client.analyze(image, {}, "test/vision", "test prompt")
    assert not list((tmp_path / "cache").glob("*.json"))


def test_batch_displays_and_logs_failure_reasons(monkeypatch, tmp_path):
    Image.new("RGB", (4, 4)).save(tmp_path / "P0001.png")
    manifest = tmp_path / "catalog.json"
    write_json(manifest, [{"name_file": "P0001.png", "model": "A"}])
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(config, "CATALOG_FILE", manifest)
    monkeypatch.setattr(config, "TEST_FILE", manifest)
    monkeypatch.setattr(config, "INDEX_DIR", tmp_path / "indexes")
    monkeypatch.setattr("ui.load_runs", lambda: [])

    def fail(*args, **kwargs):
        raise ValueError("Vision не завершил ответ: finish_reason='length'")

    monkeypatch.setattr("ui.recognize", fail)
    try:
        configure_logging()
        app = AppTest.from_string("from ui import render\nrender()")
        app.run(timeout=20)
        next(b for b in app.button if b.label == "Запустить сравнение").click().run()
        assert not app.exception
        assert "ошибок: 5" in app.success[0].value
        assert "P0001.png: Vision Only: ValueError:" in app.warning[0].value
        assert "finish_reason='length'" in app.warning[0].value
    finally:
        logger.remove()
    log = (tmp_path / "logs/errors.log").read_text(encoding="utf-8")
    assert "Пакетное распознавание: P0001.png: Vision Only" in log
    assert "finish_reason='length'" in log
