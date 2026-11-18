"""Command-line interface for local background removal."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .images import (
    HEIF_AVAILABLE,
    find_images,
    is_supported,
    load_image,
    output_path_for,
    save_png_atomic,
)
from .models import MODELS, ModelError, cache_dir, ensure_model
from .pipeline import BackgroundRemover

MODEL_CHOICES = ("lite", "full")
QUALITY_CHOICES = ("quality", "fast")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bgone",
        description=(
            "Remove the background from every image in a folder. "
            "Runs fully locally with a BiRefNet ONNX model and writes "
            "lossless full-resolution PNGs."
        ),
    )
    parser.add_argument("source", type=Path, help="Image file or folder to process")
    parser.add_argument(
        "destination",
        type=Path,
        help="Folder where transparent PNGs are written (same stem as input)",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_CHOICES,
        default="lite",
        help="lite: fast, 183 MiB download (default); full: max quality, 452 MiB, slow on CPU",
    )
    parser.add_argument(
        "--quality",
        choices=QUALITY_CHOICES,
        default="quality",
        help="quality: second pass that refines small subjects (default); fast: one inference per image",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Process subfolders and mirror their layout under the destination",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace outputs that already exist (default: skip them)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        metavar="N",
        help="ONNX Runtime CPU threads (default: automatic)",
    )
    parser.add_argument(
        "--low-memory",
        action="store_true",
        help="Trade some speed for a lower peak RAM footprint",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        metavar="PATH",
        help="Override the model cache directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be processed without downloading the model",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Only print errors and the summary"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except KeyboardInterrupt:
        print("\nInterrupted. No partial output files were left behind.", file=sys.stderr)
        return 130
    except ModelError as exc:
        print(f"bgone: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 - concise CLI failure
        print(f"bgone: {exc}", file=sys.stderr)
        return 2


def _run(args: argparse.Namespace) -> int:
    if args.threads < 0:
        raise ValueError("--threads must be zero or a positive integer")

    source: Path = args.source.expanduser()
    destination: Path = args.destination.expanduser()
    if not source.exists():
        raise ValueError(f"source does not exist: {source}")
    if source.is_file() and not is_supported(source):
        raise ValueError(f"unsupported image type: {source.suffix or '(none)'}")
    if destination.exists() and not destination.is_dir():
        raise ValueError(f"destination is not a folder: {destination}")

    resolved_source = source.resolve()
    resolved_destination = destination.resolve()
    images = find_images(
        resolved_source,
        recursive=args.recursive,
        exclude=resolved_destination if source.is_dir() else None,
    )
    if not images:
        print(
            f"bgone: no supported images found in {source}"
            + ("" if HEIF_AVAILABLE else " (.heic/.heif/.avif support unavailable)"),
            file=sys.stderr,
        )
        return 1

    model = MODELS[args.model]
    if not args.quiet:
        print(f"bgone — {model.label}", file=sys.stderr)
        print(f"Source: {source}", file=sys.stderr)
        print(f"Destination: {destination}", file=sys.stderr)
        formats = "JPEG, PNG, WebP, BMP, TIFF"
        if HEIF_AVAILABLE:
            formats += ", HEIC/HEIF/AVIF"
        print(f"Formats: {formats}", file=sys.stderr)
        print(
            f"Found {len(images)} image(s)"
            + (" (recursive)" if args.recursive else ""),
            file=sys.stderr,
        )

    planned: list[tuple[Path, Path]] = []
    claimed: dict[Path, Path] = {}
    skipped = 0
    for image_path in images:
        output = output_path_for(
            image_path,
            resolved_source,
            resolved_destination,
            recursive=args.recursive,
        )
        if output in claimed:
            print(
                f"bgone: warning: {image_path.name} and {claimed[output].name} "
                f"both map to {output.name}; skipping {image_path.name}",
                file=sys.stderr,
            )
            skipped += 1
            continue
        claimed[output] = image_path
        if output.exists() and not args.overwrite:
            skipped += 1
            continue
        planned.append((image_path, output))

    if skipped and not args.quiet:
        print(
            f"Skipping {skipped} file(s) with existing or duplicate outputs "
            f"(use --overwrite to replace).",
            file=sys.stderr,
        )

    if args.dry_run:
        for image_path, output in planned:
            print(f"{image_path} -> {output}", flush=True)
        print(
            f"Dry run: {len(planned)} to process, {skipped} skipped.",
            file=sys.stderr,
        )
        return 0

    if not planned:
        print("Nothing to do.", file=sys.stderr)
        return 0

    _warn_about_memory(args.model)
    progress = None if args.quiet else _DownloadProgress()
    model_path = ensure_model(
        model,
        cache_dir(args.model_dir),
        quiet=args.quiet,
        progress=progress,
    )
    if not args.quiet:
        print(f"Model: {model_path}", file=sys.stderr)
        print("Loading model…", file=sys.stderr)

    remover = BackgroundRemover(
        model_path, threads=args.threads, low_memory=args.low_memory
    )
    if not args.quiet:
        print(
            f"Model ready in {remover.load_seconds:.1f}s "
            f"(quality: {args.quality}, threads: {args.threads or 'auto'}"
            + (", low-memory" if args.low_memory else "")
            + ")",
            file=sys.stderr,
        )

    if not args.quiet:
        print(f"Processing {len(planned)} image(s)…", file=sys.stderr)
    failures = 0
    completed = 0
    batch_started = time.monotonic()

    for index, (image_path, output) in enumerate(planned, start=1):
        label = _display_name(image_path, resolved_source, args.recursive)
        output_label = _display_output(output, resolved_destination, args.recursive)
        prefix = f"[{index}/{len(planned)}] {label} -> {output_label}"
        if not args.quiet:
            print(prefix, end=" … ", file=sys.stderr, flush=True)
        started = time.monotonic()
        try:
            loaded = load_image(image_path)
            result = remover.remove_background(loaded.image, quality=args.quality)
            save_png_atomic(
                result,
                output,
                icc_profile=loaded.icc_profile,
                dpi=loaded.dpi,
            )
        except Exception as exc:  # noqa: BLE001 - keep the batch going
            failures += 1
            if args.quiet:
                print(f"bgone: {label}: {exc}", file=sys.stderr)
            else:
                print(f"failed: {exc}", file=sys.stderr)
            continue

        completed += 1
        elapsed = time.monotonic() - started
        if not args.quiet:
            print(f"ok ({elapsed:.1f}s)", file=sys.stderr)

    total = time.monotonic() - batch_started
    print(
        f"Done: {completed} processed, {skipped} skipped, {failures} failed "
        f"in {total:.1f}s",
        file=sys.stderr,
    )
    print(f"Output: {destination}", file=sys.stderr)
    return 1 if failures else 0


def _display_name(path: Path, source: Path, recursive: bool) -> str:
    if recursive and source.is_dir():
        try:
            return str(path.relative_to(source))
        except ValueError:
            pass
    return path.name


def _display_output(path: Path, destination: Path, recursive: bool) -> str:
    if recursive:
        try:
            return str(path.relative_to(destination))
        except ValueError:
            pass
    return path.name


def _download_progress(loaded: int, total: int) -> None:
    ratio = min(1.0, loaded / total) if total else 0.0
    width = 28
    filled = int(width * ratio)
    bar = "=" * filled + ">" + " " * max(0, width - filled - 1)
    sys.stderr.write(
        f"\r  {bar} {ratio * 100:5.1f}%  "
        f"{loaded / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MiB"
    )
    sys.stderr.flush()


class _DownloadProgress:
    """Show a live bar on a TTY, or one line per 10% when piped."""

    def __init__(self) -> None:
        self._tty = sys.stderr.isatty()
        self._bucket = -1

    def __call__(self, loaded: int, total: int) -> None:
        if self._tty:
            _download_progress(loaded, total)
            return
        ratio = min(1.0, loaded / total) if total else 0.0
        bucket = (int(ratio * 100) // 10) * 10
        if bucket != self._bucket:
            self._bucket = bucket
            print(
                f"  Downloading model… {bucket}% "
                f"({loaded / (1024 * 1024):.1f} / {total / (1024 * 1024):.1f} MiB)",
                file=sys.stderr,
            )


def _warn_about_memory(model_key: str) -> None:
    available = _available_memory_gib()
    if available is None:
        return
    if model_key == "full" and available < 2.5:
        print(
            f"bgone: warning: only {available:.1f} GiB RAM available. "
            "The full model may be slow or fail; use --model lite if it does.",
            file=sys.stderr,
        )
    elif model_key == "lite" and available < 0.7:
        print(
            f"bgone: warning: only {available:.1f} GiB RAM available. "
            "Close some applications if processing fails.",
            file=sys.stderr,
        )


def _available_memory_gib() -> float | None:
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError):
        return None
    return None


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
