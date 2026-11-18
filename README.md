# bgone

**Background → gone.** Local, offline background removal for image folders.

bgone is a terminal-focused, independent implementation inspired by
[bg0](https://github.com/opencoredev/bg0). It uses pinned BiRefNet ONNX weights
via ONNX Runtime on the CPU. Images never leave the machine, and there is no
account, API key, or usage limit.

The output is a **lossless, full-resolution RGBA PNG**: the original RGB pixels
are kept untouched, the model only supplies the alpha channel. No resizing and
no JPEG re-encoding.

## Requirements

- Linux/macOS/Windows
- [uv](https://docs.astral.sh/uv/) (recommended) or Python 3.11–3.13
- ~200 MB disk for the default model
- ~2 GiB free RAM (lite, fast), ~3.5 GiB (lite, quality), ~3.5–4 GiB (full);
  `--low-memory` lowers the quality-pass peak to about 2.1 GiB

No GPU is required. An NVIDIA GPU is not used; ONNX Runtime runs on the CPU.

## Install

```sh
git clone https://github.com/Firoar/bgone.git
cd bgone

# Install the `bgone` command into uv's isolated tool environment.
uv tool install --python 3.12 .
```

Make sure `~/.local/bin` is on your `PATH` (uv installs command shims there).
After changing the source, reinstall with `uv tool install --python 3.12 . --force`.

For development instead:

```sh
uv sync
uv run bgone --help
```

## Usage

```sh
bgone <folder_path> <destination_path>
```

Example:

```sh
bgone ~/Pictures/in ~/Pictures/out
```

The first run downloads the model (~183 MiB) to
`~/.cache/bgone/models/` and verifies its SHA-256. Later runs are fully
offline. Every input `name.jpg` becomes `name.png` in the destination folder;
PNG inputs keep the exact same filename. Existing outputs are skipped unless
`--overwrite` is given.

### Options

```
--model lite|full      lite (default): fast, 183 MiB
                       full: BiRefNet Swin-L, 452 MiB, highest quality,
                             much slower and needs ~2–4 GiB RAM
--quality quality|fast quality (default): second pass that refines small
                             subjects; fast: one inference per image
-r, --recursive        Process subfolders and mirror the folder layout
--overwrite            Replace existing output files
--threads N            ONNX Runtime CPU threads (default: automatic)
--low-memory           Trade ~15% speed for a lower peak RAM footprint
--model-dir PATH       Override the model cache directory
--dry-run              List source → output mappings without downloading
-q, --quiet            Only errors and the final summary
--version              Print the version
```

Examples:

```sh
# Most accurate, slow
bgone ~/Pictures/portraits ~/Pictures/cutouts --model full

# Quick pass over a large folder, including subfolders
bgone ~/Pictures/inbox ~/Pictures/done -r --quality fast

# Preview what will be written
bgone ~/Pictures/inbox ~/Pictures/done --dry-run
```

## How it works

1. Decode the image, apply EXIF orientation, and keep the full-resolution
   RGB pixels.
2. Resize a copy to 512×512, apply ImageNet normalization, and run BiRefNet to
   get a 512×512 alpha mask (sigmoid of the logits).
3. In `quality` mode, find the subject bounding box and, when it is small in
   frame, run a second inference on a padded crop (bounded to 1024 px) and
   paste the sharper mask back.
4. Upscale the mask to the original dimensions with a float (non-8-bit)
   resize, multiply by any alpha already present in the source, and attach it
   to the untouched RGB pixels.
5. Write an atomic PNG with the original ICC profile and DPI metadata.

## Performance on a CPU-only machine

Reference numbers from an Intel i5-8250U (4 cores / 8 threads) on a 768×512
image. Model loading adds about 5–11 seconds per run.

| Model | Quality | Per image | Peak RAM |
| --- | --- | --- | --- |
| lite | fast | ~9 s | ~2.1 GiB |
| lite | quality | ~18–21 s | ~3.3 GiB |
| lite | quality + `--low-memory` | ~21 s | ~2.1 GiB |
| full | fast | ~39 s | ~3.2 GiB |
| full | quality | ~48 s | ~3.5 GiB |

`--low-memory` is worth using on 8 GiB machines, especially with the quality
pass or the full model. Runtime scales with CPU performance; a modern desktop
CPU will be noticeably faster.

## Notes and limitations

- Output is 8-bit RGBA PNG. 16-bit sources are reduced to 8-bit.
- The alpha mask is model-estimated at 512×512; very fine hair edges can still
  need manual touch-up. `--model full` gives better edges at a large speed
  cost. If you want maximum quality on a CPU-only machine, expect tens of
  seconds per image.
- `--model full` needs roughly 2–4 GiB of free RAM. On low-memory machines the
  tool warns before loading it.
- HEIC/HEIF/AVIF support comes from `pillow-heif`; JPEG, PNG, WebP, BMP, and
  TIFF work out of the box.
- The destination folder is excluded when it is nested inside the source
  folder, so a recursive rerun does not process its own output.
- Processing is sequential on purpose: parallel images would compete for the
  same CPU cores and multiply peak memory.

## Troubleshooting

- **`bgone: command not found`** — add `~/.local/bin` to `PATH`, or run
  `uv tool update-shell`.
- **Download fails** — rerun; partial downloads resume automatically. For an
  offline machine, copy the model file into the cache directory shown by the
  error (`--model-dir` lets you choose another folder).
- **Full model is killed / swaps heavily** — use the default `--model lite`.
- **HEIC not recognized** — reinstall dependencies with
  `uv tool install --python 3.12 . --force`.

## Uninstall

```sh
uv tool uninstall bgone
rm -rf ~/.cache/bgone
```

## Acknowledgements

Inspired by [bg0](https://github.com/opencoredev/bg0), an Apache-2.0 project
that brought private, local BiRefNet background removal to the browser. bgone
is an independent Python terminal implementation; it is not a fork and is not
affiliated with or endorsed by the bg0 project.

## License

The CLI source is MIT licensed. BiRefNet weights and runtime dependencies keep
their own licenses; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
