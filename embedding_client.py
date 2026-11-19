"""Synchronous OpenRouter client for image embeddings."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import config
from api_client import OpenRouterClient
from images import image_url
from schemas import CallStats


def normalize(vector: object) -> NDArray[np.float32]:
    array = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(array)
    if array.ndim != 1 or array.size == 0 or not np.isfinite(norm) or norm == 0:
        raise ValueError("Embedding must be a one-dimensional vector with a finite, non-zero norm")
    return array / norm


class OpenRouterEmbeddings(OpenRouterClient):
    def __init__(
        self, model: str | None = None, cache_dir: Path | None = None, *, force: bool = False
    ) -> None:
        self.model = model or config.OPENROUTER_MODEL
        if not self.model.strip():
            raise ValueError("Set OPENROUTER_MODEL in config.py or pass model explicitly")
        super().__init__(cache_dir, force=force)

    def embed(self, image_path: Path) -> NDArray[np.float32]:
        return self.embed_details(image_path)[0]

    def embed_details(
        self, image_path: Path, *, query: bool = False
    ) -> tuple[NDArray[np.float32], CallStats]:
        payload = {
            "model": self.model,
            "input": [
                {"content": [{"type": "image_url", "image_url": {"url": image_url(image_path)}}]}
            ],
            "input_type": "search_query" if query else "search_document",
            "encoding_format": "float",
        }
        data, stats = self.post(config.OPENROUTER_EMBEDDINGS_URL, payload)
        vector = normalize(data["data"][0]["embedding"])
        self.save_response(config.OPENROUTER_EMBEDDINGS_URL, payload, data)
        return vector, stats
