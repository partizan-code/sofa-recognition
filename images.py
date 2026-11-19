"""Deterministic Pillow preprocessing shared by embeddings and vision."""

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageOps

import config

IMAGE_VERSION = 1
EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        source.seek(0)
        image = ImageOps.exif_transpose(source).convert("RGBA")
        background = Image.new("RGBA", image.size, "white")
        background.alpha_composite(image)
        return background.convert("RGB")


def data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def image_url(path: Path) -> str:
    if path.suffix.lower() not in EXTENSIONS:
        raise ValueError(f"Unsupported image MIME type: {path.suffix}")
    with open_rgb(path) as image:
        image.thumbnail((config.IMAGE_MAX_SIDE, config.IMAGE_MAX_SIDE))
        return data_url(image)


def reference_url(paths: list[Path]) -> str:
    """A model card shows several views without mixing labels or metadata."""
    size = 512
    with Image.new("RGB", (size * len(paths), size), "white") as sheet:
        for index, path in enumerate(paths):
            with open_rgb(path) as image:
                image.thumbnail((size, size))
                sheet.paste(
                    image, (index * size + (size - image.width) // 2, (size - image.height) // 2)
                )
        return data_url(sheet)
