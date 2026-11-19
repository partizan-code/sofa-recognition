"""Recognition pipeline shared by the interactive UI and batch experiments."""

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import config
from catalog import fingerprint
from embedding_client import OpenRouterEmbeddings
from images import IMAGE_VERSION
from schemas import MODES, Candidate, Photo, Prediction, Recognition
from storage import digest, file_digest
from vector_index import load_index
from vision_client import OpenRouterVision


@dataclass(frozen=True)
class Settings:
    mode: str = "Vision Only"
    embedding_model: str = config.VOYAGE_MODEL
    vision_model: str = config.VISION_MODELS[0]
    prompt: str = config.DEFAULT_PROMPT
    threshold: float = 0.0

    @property
    def provider(self) -> str | None:
        return next((p for p in config.EMBEDDING_MODELS if self.mode.startswith(p)), None)

    @property
    def vision(self) -> bool:
        return self.mode == "Vision Only" or self.mode.endswith("+ Vision")


def experiment_key(settings: Settings, photos: list[Photo], root: Path) -> str:
    options = asdict(settings)
    if settings.provider is None:
        options.pop("embedding_model")
    if settings.vision:
        options.pop("threshold")
    if not settings.vision:
        options.pop("vision_model")
        options.pop("prompt")
    return digest(
        {
            "settings": options,
            "catalog": fingerprint(photos, root),
            "image_version": IMAGE_VERSION,
            "side": config.IMAGE_MAX_SIDE,
            "views": config.REFERENCE_VIEWS,
            "tokens": config.MAX_OUTPUT_TOKENS,
        }
    )


def recognize(
    source: Path,
    photos: list[Photo],
    root: Path,
    settings: Settings,
    *,
    force: bool = False,
    cache_dir: Path = config.CACHE_DIR,
    index_dir: Path = config.INDEX_DIR,
) -> Recognition:
    if settings.mode not in MODES:
        raise ValueError("Неизвестный режим распознавания")
    started = perf_counter()
    source_hash = file_digest(source)
    eligible = [p for p in photos if file_digest(root / p.name_file) != source_hash]
    calls, candidates = [], []
    prediction = Prediction(model=None, color=None, configuration=None, orientation=None)
    if settings.provider:
        index = load_index(photos, root, settings.provider, settings.embedding_model, index_dir)
        with OpenRouterEmbeddings(settings.embedding_model, cache_dir, force=force) as client:
            vector, stats = client.embed_details(source, query=True)
        calls.append(stats)
        candidates = index.search(vector, exclude_hash=source_hash)
        if candidates and candidates[0].similarity >= settings.threshold:
            prediction.model = candidates[0].model
    if settings.vision:
        allowed = (
            {c.model for c in candidates} if settings.provider else {p.model for p in eligible}
        )
        references: dict[str, list[Path]] = {}
        for name in sorted(allowed):
            # Put the highest-scoring photo first for hybrid comparisons.
            ranked_file = next((c.name_file for c in candidates if c.model == name), None)
            views = sorted(
                (p for p in eligible if p.model == name), key=lambda p: p.name_file != ranked_file
            )
            references[name] = [root / p.name_file for p in views[: config.REFERENCE_VIEWS]]
        with OpenRouterVision(cache_dir, force=force) as client:
            decision, stats = client.analyze(
                source, references, settings.vision_model, settings.prompt
            )
        calls.append(stats)
        prediction = decision.prediction
        if settings.provider is None:
            names = list(decision.ranked_models)
            if prediction.model:
                names = [prediction.model] + [n for n in names if n != prediction.model]
            candidates = [
                Candidate(model=name, name_file=references[name][0].name) for name in names[:5]
            ]
    return Recognition(
        prediction=prediction,
        candidates=candidates,
        calls=calls,
        seconds=perf_counter() - started,
        experiment=experiment_key(settings, photos, root),
    )
