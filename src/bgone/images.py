"""Image discovery, decoding, and lossless PNG writing."""

from __future__ import annotations

import os
import re
import secrets
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

try:  # Optional: HEIC/HEIF/AVIF support from phone cameras.
    import pillow_heif

    pillow_heif.register_heif_opener()
    try:
        pillow_heif.register_avif_opener()
    except AttributeError:
        pass
    HEIF_AVAILABLE = True
except Exception:  # pragma: no cover - depends on optional wheels
    HEIF_AVAILABLE = False

BASE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}
HEIF_EXTENSIONS = {".heic", ".heif", ".avif"}
SUPPORTED_EXTENSIONS = BASE_EXTENSIONS | (HEIF_EXTENSIONS if HEIF_AVAILABLE else set())


@dataclass
class LoadedImage:
    image: Image.Image
    icc_profile: bytes | None
    dpi: tuple[float, float] | None


def is_supported(path: Path) -> bool:
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def _natural_key(path: Path) -> list[object]:
    parts = re.split(r"(\d+)", str(path).lower())
    return [int(part) if part.isdigit() else part for part in parts]


def find_images(
    source: Path, *, recursive: bool = False, exclude: Path | None = None
) -> list[Path]:
    """Return supported image files in a deterministic order.

    `exclude` lets a destination directory nested inside the source be skipped.
    """
    source = source.resolve()
    if source.is_file():
        return [source] if is_supported(source) else []

    if recursive:
        candidates = source.rglob("*")
    else:
        candidates = source.iterdir()

    excluded = exclude.resolve() if exclude is not None else None
    images: list[Path] = []
    for path in candidates:
        if not path.is_file():
            continue
        if recursive:
            relative_parts = path.relative_to(source).parts
        else:
            relative_parts = (path.name,)
        if any(part.startswith(".") for part in relative_parts):
            continue
        if not is_supported(path):
            continue
        if excluded is not None and path.resolve().is_relative_to(excluded):
            continue
        images.append(path)
    return sorted(images, key=_natural_key)


def output_path_for(
    image_path: Path, source: Path, destination: Path, *, recursive: bool
) -> Path:
    """Map an input file to `<destination>/<same stem>.png`."""
    source = source.resolve()
    if source.is_file() or not recursive:
        return destination / f"{image_path.stem}.png"
    relative = image_path.resolve().relative_to(source)
    return destination / relative.with_suffix(".png")


def load_image(path: Path) -> LoadedImage:
    """Decode an image fully and normalize EXIF orientation."""
    with Image.open(path) as opened:
        opened.load()
        image = ImageOps.exif_transpose(opened)

    if image.mode in ("I;16", "I;16L", "I;16B") or image.mode == "I":
        # Pillow's direct 16-bit -> RGB conversion truncates instead of scaling.
        values = np.asarray(image)
        if values.dtype == np.uint16:
            scaled = values.astype(np.float32) / 257.0
        else:
            scaled = values.astype(np.float32)
            if float(scaled.max(initial=0)) > 255.0:
                scaled = scaled / 257.0
        image = Image.fromarray(
            np.clip(np.rint(scaled), 0, 255).astype(np.uint8), mode="L"
        )

    icc_profile = image.info.get("icc_profile")
    if image.mode not in ("RGB", "RGBA", "P"):
        # A CMYK/gray profile must not be attached to an RGB PNG.
        icc_profile = None

    dpi = image.info.get("dpi")
    if not (
        isinstance(dpi, (tuple, list))
        and len(dpi) == 2
        and all(isinstance(value, (int, float)) and value > 0 for value in dpi)
    ):
        dpi = None
    else:
        dpi = (float(dpi[0]), float(dpi[1]))

    return LoadedImage(image=image, icc_profile=icc_profile, dpi=dpi)


def save_png_atomic(
    image: Image.Image,
    destination: Path,
    *,
    icc_profile: bytes | None = None,
    dpi: tuple[float, float] | None = None,
) -> None:
    """Write a PNG atomically so an interrupted run never leaves broken files."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{secrets.token_hex(4)}.tmp"
    )
    options: dict[str, object] = {
        "format": "PNG",
        "compress_level": 6,
    }
    if icc_profile:
        options["icc_profile"] = icc_profile
    if dpi is not None:
        options["dpi"] = dpi

    try:
        image.save(temporary, **options)
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
