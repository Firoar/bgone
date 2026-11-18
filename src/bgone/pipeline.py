"""BiRefNet inference, mask refinement, and lossless alpha compositing.

The preprocessing and refinement heuristics mirror bg0's browser pipeline so
the CLI produces comparable masks while keeping output at full resolution.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

MODEL_INPUT_SIZE = 512
FOREGROUND_THRESHOLD = 0.1
REFINEMENT_PADDING_RATIO = 0.12
MAX_REFINEMENT_AREA_RATIO = 0.85
LOW_ZOOM_AREA_RATIO = 0.8
MIN_LOW_ZOOM_FOREGROUND_DENSITY = 0.4
REFINEMENT_SOURCE_MAX_SIDE = 1024

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


@dataclass(frozen=True)
class Refinement:
    """A refined mask and the full-resolution rectangle it replaces."""

    mask: np.ndarray
    left: int
    top: int
    right: int
    bottom: int


class BackgroundRemover:
    """Wraps one ONNX Runtime session and applies bg0-style postprocessing."""

    def __init__(
        self, model_path: Path, threads: int = 0, low_memory: bool = False
    ) -> None:
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = max(0, threads)
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        # The arena speeds up repeated runs but retains large scratch buffers.
        # Disabling it keeps the quality refinement pass near one-model peak RSS.
        options.enable_cpu_mem_arena = not low_memory

        started = time.monotonic()
        try:
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
        except Exception as exc:  # noqa: BLE001 - surface a clear CLI error
            raise RuntimeError(
                f"ONNX Runtime could not load {model_path.name}: {exc}"
            ) from exc

        self.load_seconds = time.monotonic() - started
        self.input_name = self.session.get_inputs()[0].name
        input_type = self.session.get_inputs()[0].type
        self.input_dtype = np.float16 if "float16" in input_type else np.float32

        outputs = self.session.get_outputs()
        preferred = next(
            (
                item
                for item in outputs
                if "logits" in item.name.lower() or "output_image" in item.name.lower()
            ),
            outputs[0],
        )
        self.output_name = preferred.name

    def infer_mask(self, image: Image.Image) -> np.ndarray:
        """Run the model on a PIL image and return a 512x512 float alpha mask."""
        tensor = self._preprocess(image)
        try:
            raw = self.session.run(
                [self.output_name], {self.input_name: tensor}
            )[0]
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Model inference failed: {exc}") from exc
        return _sigmoid_to_mask(raw)

    def _preprocess(self, image: Image.Image) -> np.ndarray:
        resized = image.convert("RGB").resize(
            (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), Image.Resampling.BILINEAR
        )
        pixels = np.asarray(resized, dtype=np.float32) / 255.0
        pixels = pixels.transpose(2, 0, 1)
        pixels = (pixels - _MEAN) / _STD
        tensor = pixels[None, ...]
        if self.input_dtype != np.float32:
            tensor = tensor.astype(self.input_dtype)
        return np.ascontiguousarray(tensor)

    def remove_background(
        self, image: Image.Image, quality: str = "quality"
    ) -> Image.Image:
        """Return a full-resolution RGBA image with the background removed."""
        width, height = image.size
        base = self.infer_mask(image)

        if quality == "fast":
            # bg0's fast path sharpens the 512px mask before upscaling.
            base = _smoothstep(0.08, 0.92, base)

        alpha = upscale_mask(base, width, height, quality)

        if quality == "quality":
            try:
                refinement = self._refine(image, base, width, height)
            except Exception:  # noqa: BLE001 - refinement is optional
                refinement = None
            if refinement is not None:
                alpha[
                    refinement.top : refinement.bottom,
                    refinement.left : refinement.right,
                ] = refinement.mask

        alpha = np.clip(alpha, 0.0, 1.0)
        alpha *= _source_alpha(image)

        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        alpha_u8 = np.rint(alpha * 255.0).astype(np.uint8)
        rgba = np.dstack((rgb, alpha_u8))
        return Image.fromarray(rgba, mode="RGBA")

    def _refine(
        self, image: Image.Image, base: np.ndarray, width: int, height: int
    ) -> Refinement | None:
        """Run a second pass on a bounded subject crop, like bg0 quality mode."""
        source = image
        max_side = max(image.size)
        if max_side > REFINEMENT_SOURCE_MAX_SIDE:
            scale = REFINEMENT_SOURCE_MAX_SIDE / max_side
            source = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                ),
                Image.Resampling.LANCZOS,
            )

        crop = find_refinement_crop(
            base, base.shape[1], base.shape[0], source.width, source.height
        )
        if crop is None:
            return None

        left, top, right, bottom = crop
        cropped = source.crop(crop)
        try:
            mask = self.infer_mask(cropped)
        except RuntimeError:
            return None

        if not _has_foreground_signal(mask):
            return None

        out_left = int(left * width / source.width)
        out_top = int(top * height / source.height)
        out_right = int(np.ceil(right * width / source.width))
        out_bottom = int(np.ceil(bottom * height / source.height))
        out_left = max(0, min(width, out_left))
        out_top = max(0, min(height, out_top))
        out_right = max(out_left, min(width, out_right))
        out_bottom = max(out_top, min(height, out_bottom))
        if out_right - out_left < 1 or out_bottom - out_top < 1:
            return None

        refined = upscale_mask(
            mask, out_right - out_left, out_bottom - out_top, "quality"
        )
        return Refinement(refined, out_left, out_top, out_right, out_bottom)


def _sigmoid_to_mask(raw: np.ndarray) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float32)
    if values.ndim >= 3:
        squeezed = np.squeeze(values)
    else:
        squeezed = values
    while squeezed.ndim > 2 and squeezed.shape[0] == 1:
        squeezed = squeezed[0]
    if squeezed.ndim == 3 and squeezed.shape[0] == 1:
        squeezed = squeezed[0]
    if squeezed.ndim != 2:
        raise RuntimeError(f"Unexpected model output shape: {values.shape}")

    clipped = np.clip(squeezed, -60.0, 60.0)
    alpha = 1.0 / (1.0 + np.exp(-clipped))
    return np.ascontiguousarray(alpha, dtype=np.float32)


def _has_foreground_signal(mask: np.ndarray) -> bool:
    if mask.size == 0 or not np.isfinite(mask).all():
        return False
    low = float(mask.min())
    high = float(mask.max())
    return high >= 0.05 and high - low >= 0.005


def _smoothstep(minimum: float, maximum: float, values: np.ndarray) -> np.ndarray:
    scaled = np.clip((values - minimum) / (maximum - minimum), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def upscale_mask(
    mask: np.ndarray, width: int, height: int, quality: str = "quality"
) -> np.ndarray:
    """Upscale a float mask to full output size without 8-bit quantization."""
    resample = (
        Image.Resampling.LANCZOS
        if quality == "quality"
        else Image.Resampling.BILINEAR
    )
    source = Image.fromarray(np.ascontiguousarray(mask, dtype=np.float32), mode="F")
    if source.size != (width, height):
        source = source.resize((width, height), resample)
    return np.clip(np.asarray(source, dtype=np.float32), 0.0, 1.0)


def _source_alpha(image: Image.Image) -> np.ndarray:
    """Return the source alpha as 0..1, or ones when the image is opaque."""
    has_alpha = image.mode in ("RGBA", "LA") or (
        image.mode == "P" and "transparency" in image.info
    )
    if not has_alpha:
        return 1.0
    alpha = np.asarray(image.convert("RGBA"), dtype=np.uint8)[..., 3]
    return alpha.astype(np.float32) / 255.0


def find_refinement_crop(
    mask: np.ndarray,
    mask_width: int,
    mask_height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    """Locate a padded subject crop worth a second inference pass.

    Mirrors bg0's heuristics: skip full-frame subjects, tiny crops, and large
    sparse subjects where a second pass would mostly lose edge pixels.
    """
    foreground = mask >= FOREGROUND_THRESHOLD
    if not foreground.any():
        return None

    ys, xs = np.nonzero(foreground)
    min_x = int(xs.min())
    max_x = int(xs.max())
    min_y = int(ys.min())
    max_y = int(ys.max())
    foreground_pixels = int(foreground.sum())

    scale_x = image_width / mask_width
    scale_y = image_height / mask_height
    left = int(np.floor(min_x * scale_x))
    top = int(np.floor(min_y * scale_y))
    right = int(np.ceil((max_x + 1) * scale_x))
    bottom = int(np.ceil((max_y + 1) * scale_y))

    padding = int(
        np.ceil(max(right - left, bottom - top) * REFINEMENT_PADDING_RATIO)
    )
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(image_width, right + padding)
    bottom = min(image_height, bottom + padding)

    crop_width = right - left
    crop_height = bottom - top
    if crop_width < 32 or crop_height < 32:
        return None

    crop_area = crop_width * crop_height
    image_area = image_width * image_height
    if crop_area >= image_area * MAX_REFINEMENT_AREA_RATIO:
        return None

    foreground_area = foreground_pixels * scale_x * scale_y
    if (
        crop_area >= image_area * LOW_ZOOM_AREA_RATIO
        and foreground_area / crop_area < MIN_LOW_ZOOM_FOREGROUND_DENSITY
    ):
        return None

    return left, top, right, bottom
