"""Pinned BiRefNet model registry and resumable, verified downloads."""

from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

HF_BASE = "https://huggingface.co"


class ModelError(RuntimeError):
    """A model could not be downloaded, verified, or loaded."""


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    repo: str
    revision: str
    filename: str
    size: int
    sha256: str
    note: str

    @property
    def url(self) -> str:
        return f"{HF_BASE}/{self.repo}/resolve/{self.revision}/{self.filename}"

    @property
    def cache_name(self) -> str:
        return f"{self.repo.replace('/', '__')}@{self.revision[:12]}__{Path(self.filename).name}"

    @property
    def size_label(self) -> str:
        return f"{self.size / (1024 * 1024):.0f} MiB"


# The lite weights are the fp32 export: ONNX Runtime's CPU provider is fastest
# and most reliable with float32, and the model is only 183 MiB.
LITE = ModelSpec(
    key="lite",
    label="BiRefNet Lite 512",
    repo="studioludens/birefnet-lite-512",
    revision="4a3c40c36c94093cc1e724d9ea428b8fa4b57dc7",
    filename="onnx/model.onnx",
    size=191_877_254,
    sha256="1cb0fb360dadd15af77c639085d77a9df67db0c64315560c3de005f676345ac2",
    note="Fast, ~183 MiB download. Best default for CPU-only machines.",
)

# The full Swin-L weights. This is the standard ONNX Community fp16 export; the
# CPU provider upcasts unsupported fp16 operators as needed.
FULL = ModelSpec(
    key="full",
    label="BiRefNet 512 (Swin-L)",
    repo="onnx-community/BiRefNet_512x512-ONNX",
    revision="b0b30aff33d009f6bcd7dacdc3cdcf2f8f42175b",
    filename="onnx/model_fp16.onnx",
    size=473_498_113,
    sha256="1b254749feb1edf83667a4c9da710cc1962a1980b8d0b2f4432100278c9ac0af",
    note="Highest quality, ~452 MiB download. Much slower and needs several GiB of RAM.",
)

MODELS: dict[str, ModelSpec] = {LITE.key: LITE, FULL.key: FULL}


def cache_dir(override: str | os.PathLike[str] | None = None) -> Path:
    """Return the model cache directory (`--model-dir`, else XDG cache)."""
    if override:
        return Path(override).expanduser().resolve()
    root = os.environ.get("XDG_CACHE_HOME")
    base = Path(root).expanduser() if root else Path.home() / ".cache"
    return base / "bgone" / "models"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _human_mib(value: int) -> str:
    return f"{value / (1024 * 1024):.1f} MiB"


def _default_progress(loaded: int, total: int) -> None:
    if not sys.stderr.isatty():
        return
    width = 28
    ratio = min(1.0, loaded / total) if total else 0.0
    filled = int(width * ratio)
    bar = "=" * filled + ">" + " " * max(0, width - filled - 1)
    sys.stderr.write(
        f"\r  {bar} {ratio * 100:5.1f}%  {_human_mib(loaded)} / {_human_mib(total)}"
    )
    sys.stderr.flush()


ProgressCallback = Callable[[int, int], None]


def download_model(
    spec: ModelSpec,
    model_dir: Path,
    *,
    quiet: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    """Download and verify a model, resuming a prior partial download.

    Returns the path to a verified cache file. Repeated runs do not hash the
    full file when a matching `.verified` stamp is present.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    final = model_dir / spec.cache_name
    stamp = model_dir / f"{spec.cache_name}.verified"
    part = model_dir / f"{spec.cache_name}.part"

    if final.exists():
        if final.stat().st_size == spec.size:
            if stamp.exists() and stamp.read_text(encoding="utf-8").strip() == spec.sha256:
                return final
            if not quiet:
                print(f"Verifying {spec.label}…", file=sys.stderr)
            if _sha256(final) == spec.sha256:
                stamp.write_text(spec.sha256, encoding="utf-8")
                return final
        # A wrong-sized or corrupt file is not trustworthy.
        final.unlink(missing_ok=True)
        stamp.unlink(missing_ok=True)

    if part.exists() and part.stat().st_size >= spec.size:
        part.unlink()
    if part.exists() and part.stat().st_size == spec.size:
        if _sha256(part) == spec.sha256:
            part.replace(final)
            stamp.write_text(spec.sha256, encoding="utf-8")
            return final
        part.unlink()

    pos = part.stat().st_size if part.exists() else 0
    if not quiet:
        print(
            f"Downloading {spec.label} ({spec.size_label})…",
            file=sys.stderr,
        )
        if pos:
            print(f"  Resuming at {_human_mib(pos)}", file=sys.stderr)

    headers = {"User-Agent": "bgone/0.1 (+https://github.com/Firoar/bgone)"}
    if pos:
        headers["Range"] = f"bytes={pos}-"
    request = urllib.request.Request(spec.url, headers=headers)

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            if response.status == 206 and pos:
                mode = "ab"
            else:
                mode = "wb"
                pos = 0
            total = spec.size
            last_update = 0.0
            with part.open(mode) as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    pos += len(chunk)
                    now = time.monotonic()
                    if now - last_update >= 0.1:
                        last_update = now
                        if progress is not None:
                            progress(pos, total)
                        elif not quiet:
                            _default_progress(pos, total)
    except urllib.error.HTTPError as exc:
        raise ModelError(
            f"Could not download {spec.label}: HTTP {exc.code} from {spec.url}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ModelError(
            f"Could not download {spec.label}: {exc.reason}. "
            f"Check your network and retry; partial downloads resume automatically."
        ) from exc
    except OSError as exc:
        raise ModelError(f"Could not write the model to {part}: {exc}") from exc

    if progress is not None and not quiet:
        progress(spec.size, spec.size)
    if not quiet and sys.stderr.isatty():
        sys.stderr.write("\n")
        sys.stderr.flush()

    if not part.exists() or part.stat().st_size != spec.size:
        actual = part.stat().st_size if part.exists() else 0
        raise ModelError(
            f"Download of {spec.label} is incomplete "
            f"({_human_mib(actual)} of {_human_mib(spec.size)}). Retry to resume."
        )

    if not quiet:
        print(f"Verifying {spec.label}…", file=sys.stderr)
    digest = _sha256(part)
    if digest != spec.sha256:
        part.unlink(missing_ok=True)
        raise ModelError(
            f"Checksum mismatch for {spec.label}. "
            f"Expected {spec.sha256}, got {digest}. The download was discarded; retry."
        )

    part.replace(final)
    stamp.write_text(spec.sha256, encoding="utf-8")
    return final


def ensure_model(
    spec: ModelSpec,
    model_dir: Path,
    *,
    quiet: bool = False,
    progress: ProgressCallback | None = None,
) -> Path:
    stamp = model_dir / f"{spec.cache_name}.verified"
    final = model_dir / spec.cache_name
    if (
        final.exists()
        and final.stat().st_size == spec.size
        and stamp.exists()
        and stamp.read_text(encoding="utf-8").strip() == spec.sha256
    ):
        return final
    return download_model(spec, model_dir, quiet=quiet, progress=progress)
