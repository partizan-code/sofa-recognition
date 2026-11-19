"""Exercise the embedding client with an in-memory HTTP transport."""

import json
from functools import partial
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

import config
from embedding_client import OpenRouterEmbeddings
from images import image_url


def mock_response(monkeypatch: pytest.MonkeyPatch, vector: list[float]) -> list[httpx.Request]:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"data": [{"embedding": vector}]})

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(
        httpx, "Client", partial(httpx.Client, transport=httpx.MockTransport(handler))
    )
    return requests


def test_embed_normalizes_and_closes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    requests = mock_response(monkeypatch, [3.0, 4.0])
    monkeypatch.setattr(config, "OPENROUTER_MODEL", "test-model")
    monkeypatch.setattr(config, "OPENROUTER_EMBEDDINGS_URL", "https://example.test/embeddings")
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "red").save(image)

    with OpenRouterEmbeddings() as client:
        vector = client.embed(image)

    np.testing.assert_allclose(vector, [0.6, 0.8])
    assert vector.dtype == np.float32
    assert client.client.is_closed
    request = requests[0]
    assert str(request.url) == "https://example.test/embeddings"
    assert request.headers["Authorization"] == "Bearer test-key"
    payload = json.loads(request.content)
    assert payload["model"] == "test-model"
    assert payload["encoding_format"] == "float"
    assert payload["input"][0]["content"][0]["image_url"]["url"] == image_url(image)
    assert payload["input_type"] == "search_document"


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterEmbeddings("test-model")


def test_non_image_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = mock_response(monkeypatch, [3.0, 4.0])
    with OpenRouterEmbeddings("test-model") as client:
        with pytest.raises(ValueError, match="MIME type"):
            client.embed(Path("document.txt"))
    assert not requests


def test_zero_embedding_is_rejected(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    mock_response(monkeypatch, [0.0, 0.0])
    image = tmp_path / "image.png"
    Image.new("RGB", (8, 8), "red").save(image)
    with OpenRouterEmbeddings("test-model") as client:
        with pytest.raises(ValueError, match="non-zero norm"):
            client.embed(image)
