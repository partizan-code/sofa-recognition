"""Independent NumPy indexes, unique-model cosine search, and duplicate exclusion."""

import json
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

import config
from catalog import fingerprint
from embedding_client import OpenRouterEmbeddings, normalize
from images import IMAGE_VERSION
from schemas import CallStats, Candidate, Photo
from storage import digest, file_digest


def index_path(provider: str, model: str, directory: Path = config.INDEX_DIR) -> Path:
    return directory / provider.lower() / f"{digest(model)[:20]}.npz"


def signature(photos: list[Photo], root: Path, model: str) -> str:
    return digest(
        {
            "catalog": fingerprint(photos, root),
            "model": model,
            "preprocess": IMAGE_VERSION,
            "side": config.IMAGE_MAX_SIDE,
            "input_type": "search_document",
        }
    )


class VectorIndex:
    def __init__(self, photos: list[Photo], vectors: np.ndarray, hashes: list[str]) -> None:
        if vectors.ndim != 2 or len(vectors) != len(photos) or len(hashes) != len(photos):
            raise ValueError("Неверная форма индекса; пересоздайте индекс")
        self.photos = photos
        self.vectors = np.stack([normalize(v) for v in vectors])
        self.hashes = hashes

    def search(
        self, vector: np.ndarray, *, exclude_hash: str | None = None, top_k: int = 5
    ) -> list[Candidate]:
        query = normalize(vector)
        if query.size != self.vectors.shape[1]:
            raise ValueError("Размерность embedding изменилась; пересоздайте индекс")
        scores = self.vectors @ query
        found: list[Candidate] = []
        seen: set[str] = set()
        for i in np.argsort(-scores, kind="stable"):
            photo = self.photos[i]
            if self.hashes[i] == exclude_hash or photo.model in seen:
                continue
            seen.add(photo.model)
            found.append(
                Candidate(
                    model=photo.model,
                    name_file=photo.name_file,
                    similarity=float(np.clip(scores[i], -1, 1)),
                )
            )
            if len(found) >= top_k:
                break
        return found


def build_index(
    photos: list[Photo],
    root: Path,
    provider: str,
    model: str,
    *,
    directory: Path = config.INDEX_DIR,
    cache_dir: Path = config.CACHE_DIR,
    progress: Callable[[int, int], None] | None = None,
) -> list[CallStats]:
    if not photos:
        raise ValueError("Каталог пуст")
    current_signature = signature(photos, root, model)
    vectors, calls, hashes = [], [], []
    with OpenRouterEmbeddings(model, cache_dir) as client:
        for i, photo in enumerate(photos):
            path = root / photo.name_file
            hashes.append(file_digest(path))
            vector, stats = client.embed_details(path)
            vectors.append(vector)
            calls.append(stats)
            if progress:
                progress(i + 1, len(photos))
    if current_signature != signature(photos, root, model):
        raise ValueError("Каталог изменился во время индексации; повторите запуск")
    matrix = np.stack(vectors)
    path = index_path(provider, model, directory)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(dir=path.parent, delete=False) as f:
            temporary = Path(f.name)
            np.savez_compressed(
                f,
                vectors=matrix,
                hashes=np.array(hashes),
                signature=np.array(current_signature),
                photos=np.array(json.dumps([p.model_dump() for p in photos])),
            )
        os.replace(temporary, path)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return calls


def load_index(
    photos: list[Photo], root: Path, provider: str, model: str, directory: Path = config.INDEX_DIR
) -> VectorIndex:
    path = index_path(provider, model, directory)
    if not path.exists():
        raise ValueError(f"Индекс {provider} отсутствует. Нажмите «Построить индекс».")
    with np.load(path, allow_pickle=False) as data:
        if str(data["signature"]) != signature(photos, root, model):
            raise ValueError("Каталог или настройки изменились. Пересоздайте индекс.")
        stored = [Photo.model_validate(p) for p in json.loads(str(data["photos"]))]
        return VectorIndex(stored, data["vectors"], data["hashes"].tolist())
