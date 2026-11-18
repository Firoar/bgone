from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from bgone.images import (
    find_images,
    load_image,
    output_path_for,
    save_png_atomic,
)
from bgone.pipeline import find_refinement_crop, upscale_mask


def test_upscale_mask_returns_full_resolution_float() -> None:
    mask = np.linspace(0.0, 1.0, 512 * 512, dtype=np.float32).reshape(512, 512)
    result = upscale_mask(mask, 640, 480, "quality")
    assert result.shape == (480, 640)
    assert result.dtype == np.float32
    assert 0.0 <= float(result.min()) <= float(result.max()) <= 1.0


def test_find_refinement_crop_skips_full_frame_subject() -> None:
    mask = np.full((512, 512), 0.9, dtype=np.float32)
    assert find_refinement_crop(mask, 512, 512, 4000, 3000) is None


def test_find_refinement_crop_skips_empty_mask() -> None:
    mask = np.zeros((512, 512), dtype=np.float32)
    assert find_refinement_crop(mask, 512, 512, 4000, 3000) is None


def test_find_refinement_crop_returns_padded_subject_box() -> None:
    mask = np.zeros((512, 512), dtype=np.float32)
    mask[180:330, 200:310] = 0.95
    crop = find_refinement_crop(mask, 512, 512, 2048, 2048)
    assert crop is not None
    left, top, right, bottom = crop
    assert 0 <= left < 800
    assert 0 <= top < 720
    assert 1240 <= right <= 2048
    assert 1320 <= bottom <= 2048
    assert right - left >= 440
    assert bottom - top >= 600


def test_output_paths_flat_and_recursive(tmp_path) -> None:
    source = tmp_path / "in"
    (source / "sub").mkdir(parents=True)
    flat = source / "Photo.JPG"
    nested = source / "sub" / "other.jpeg"
    flat.write_bytes(b"x")
    nested.write_bytes(b"x")

    assert output_path_for(flat, source, tmp_path / "out", recursive=False) == (
        tmp_path / "out" / "Photo.png"
    )
    assert output_path_for(nested, source, tmp_path / "out", recursive=True) == (
        tmp_path / "out" / "sub" / "other.png"
    )


def test_find_images_excludes_destination_and_hidden(tmp_path) -> None:
    source = tmp_path / "in"
    nested_dest = source / "out"
    nested_dest.mkdir(parents=True)
    (source / "a.jpg").write_bytes(b"x")
    (source / ".hidden.png").write_bytes(b"x")
    (source / "notes.txt").write_bytes(b"x")
    (nested_dest / "old.png").write_bytes(b"x")

    found = find_images(source, recursive=True, exclude=nested_dest)
    assert [path.name for path in found] == ["a.jpg"]


def test_save_png_is_atomic_and_lossless(tmp_path) -> None:
    image = Image.new("RGBA", (4, 3), (10, 20, 30, 128))
    destination = tmp_path / "nested" / "image.png"
    save_png_atomic(image, destination)
    assert destination.exists()
    with Image.open(destination) as loaded:
        assert loaded.mode == "RGBA"
        assert loaded.size == (4, 3)
        assert loaded.getpixel((0, 0)) == (10, 20, 30, 128)
    assert not list(destination.parent.glob("*.tmp"))


def test_load_16bit_grayscale_scales_to_8bit(tmp_path) -> None:
    values = np.array([[0, 32896], [65535, 25700]], dtype=np.uint16)
    path = tmp_path / "gray16.png"
    Image.fromarray(values).save(path)
    loaded = load_image(path)
    assert loaded.image.mode == "L"
    pixels = np.asarray(loaded.image)
    assert pixels.tolist() == [[0, 128], [255, 100]]


@pytest.mark.parametrize("quality", ["quality", "fast"])
def test_upscale_preserves_uniform_mask(quality: str) -> None:
    mask = np.full((512, 512), 0.5, dtype=np.float32)
    result = upscale_mask(mask, 300, 200, quality)
    assert np.allclose(result, 0.5, atol=1e-5)
