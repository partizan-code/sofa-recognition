"""OpenRouter transport with bounded timeout and content-addressed response cache."""

from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import Self

import httpx

import config
from schemas import CallStats
from storage import digest, read_json, write_json


class OpenRouterClient:
    def __init__(self, cache_dir: Path | None = None, *, force: bool = False) -> None:
        self.cache_dir = cache_dir
        self.force = force
        self.client = httpx.Client(
            headers={"Authorization": f"Bearer {config.get_openrouter_api_key()}"},
            timeout=config.HTTP_TIMEOUT_SECONDS,
        )

    def post(self, url: str, payload: dict) -> tuple[dict, CallStats]:
        started = perf_counter()
        key = digest({"version": 1, "url": url, "payload": payload})
        path = self.cache_dir / f"{key}.json" if self.cache_dir is not None else None
        if path is not None and path.exists() and not self.force:
            data = read_json(path)
            return data, CallStats(
                seconds=perf_counter() - started,
                cached=True,
                cost_usd=0.0,
                usage=data.get("usage", {}),
            )
        response = self.client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            raise ValueError("OpenRouter returned an API error; response was not cached")
        usage = data.get("usage") or {}
        stats = CallStats(seconds=perf_counter() - started, cost_usd=usage.get("cost"), usage=usage)
        return data, stats

    def save_response(self, url: str, payload: dict, data: dict) -> None:
        # Only validated API responses are cached by the caller.
        if self.cache_dir is not None:
            key = digest({"version": 1, "url": url, "payload": payload})
            write_json(self.cache_dir / f"{key}.json", data)

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
