"""Load reference catalogs and test photos without inventing ground truth."""

from pathlib import Path

from images import EXTENSIONS
from schemas import Photo
from storage import digest, file_digest, read_json


def load_photos(manifest: Path, *, references: bool = False) -> list[Photo]:
    if manifest.exists():
        photos = [Photo.model_validate(row) for row in read_json(manifest)]
    elif references:
        raise ValueError(f"Каталог не найден: {manifest}")
    else:
        photos = [
            Photo(name_file=p.name)
            for p in sorted(manifest.parent.glob("*"))
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        ]
    if len({p.name_file for p in photos}) != len(photos):
        raise ValueError("Повторяющиеся name_file в метаданных")
    for photo in photos:
        if not (manifest.parent / photo.name_file).is_file():
            raise ValueError(f"Фото не найдено: {photo.name_file}")
        if references and not photo.model:
            raise ValueError(f"У эталона отсутствует модель: {photo.name_file}")
    return photos


def fingerprint(photos: list[Photo], root: Path) -> str:
    return digest([(p.model_dump(), file_digest(root / p.name_file)) for p in photos])
