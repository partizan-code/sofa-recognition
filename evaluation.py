"""Persist runs and calculate accuracy, abstention, recall and measured API cost."""

from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import config
from recognition import Settings
from schemas import Photo, Recognition
from storage import file_digest, read_json, write_json

FIELDS = ("model", "color", "configuration", "orientation")


def equivalent(first: str | None, second: str | None) -> bool:
    return (
        first is not None
        and second is not None
        and first.strip().casefold() == second.strip().casefold()
    )


def comparisons(photo: Photo, result: Recognition) -> list[dict]:
    predicted = result.prediction.model_dump()
    return [
        {
            "Параметр": field,
            "Ожидается": truth,
            "Результат": predicted[field],
            "Совпадение": None if truth is None else equivalent(truth, predicted[field]),
        }
        for field, truth in photo.truth().items()
    ]


def save_run(
    photo: Photo,
    source: Path,
    settings: Settings,
    result: Recognition,
    directory: Path = config.RESULTS_DIR,
) -> None:
    write_json(
        directory / f"{uuid4().hex}.json",
        {
            "created": datetime.now(UTC).isoformat(),
            "image_hash": file_digest(source),
            "photo": photo.model_dump(),
            "settings": asdict(settings),
            "result": result.model_dump(),
            "cost_usd": result.cost_usd,
        },
    )


def load_runs(directory: Path = config.RESULTS_DIR) -> list[dict]:
    return sorted(
        (read_json(path) for path in directory.glob("*.json")), key=lambda row: row["created"]
    )


def summarize(runs: list[dict], *, common_only: bool = False) -> list[dict]:
    grouped = defaultdict(list)
    for run in runs:
        grouped[run["result"]["experiment"]].append(run)
    common = (
        set.intersection(*({r["image_hash"] for r in history} for history in grouped.values()))
        if grouped
        else set()
    )
    rows = []
    for experiment, history in grouped.items():
        # One latest observation per image: browsing/reruns do not inflate accuracy.
        latest = {r["image_hash"]: r for r in history}
        samples = [r for r in latest.values() if not common_only or r["image_hash"] in common]
        if not samples:
            continue
        settings = samples[-1]["settings"]
        row = {
            "Эксперимент": experiment[:12],
            "Режим": settings["mode"],
            "Embedding": settings["embedding_model"] if settings["mode"] != "Vision Only" else None,
            "Vision": settings["vision_model"] if "Vision" in settings["mode"] else None,
            "Фото": len(samples),
        }
        for field in FIELDS:
            known = [r for r in samples if r["photo"].get(field) is not None]
            answered = [r for r in known if r["result"]["prediction"][field] is not None]
            correct = sum(
                equivalent(r["photo"][field], r["result"]["prediction"][field]) for r in known
            )
            row[f"{field} N"] = len(known)
            row[f"{field} accuracy"] = correct / len(known) if known else None
            row[f"{field} coverage"] = len(answered) / len(known) if known else None
            row[f"{field} answered accuracy"] = correct / len(answered) if answered else None
        known_models = [r for r in samples if r["photo"].get("model") is not None]
        row["TOP-5 recall"] = (
            sum(
                any(equivalent(r["photo"]["model"], c["model"]) for c in r["result"]["candidates"])
                for r in known_models
            )
            / len(known_models)
            if known_models
            else None
        )
        measured = [r for r in history if not common_only or r["image_hash"] in common]
        fresh = [r for r in measured if all(not c["cached"] for c in r["result"]["calls"])]
        row["API запусков"] = len(fresh)
        row["Среднее время полностью без кэша, с"] = (
            sum(r["result"]["seconds"] for r in fresh) / len(fresh) if fresh else None
        )
        costs = [r["cost_usd"] for r in measured]
        row["Известные расходы, USD"] = sum(c for c in costs if c is not None)
        row["Запусков с неизвестной стоимостью"] = sum(c is None for c in costs)
        row["Средняя стоимость API запуска, USD"] = (
            sum(r["cost_usd"] for r in fresh) / len(fresh)
            if fresh and all(r["cost_usd"] is not None for r in fresh)
            else None
        )
        rows.append(row)
    return rows
