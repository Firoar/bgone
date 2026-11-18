# Third-party notices

This project (bgone) downloads model weights at runtime; they are not bundled.

## BiRefNet

- Project: https://github.com/ZhengPeng7/BiRefNet
- Original weights: `ZhengPeng7/BiRefNet`, `ZhengPeng7/BiRefNet_lite`
- Lite ONNX export used by default: `studioludens/birefnet-lite-512`,
  revision `4a3c40c36c94093cc1e724d9ea428b8fa4b57dc7`
- Full ONNX export: `onnx-community/BiRefNet_512x512-ONNX`,
  revision `b0b30aff33d009f6bcd7dacdc3cdcf2f8f42175b`
- License: MIT (see the model cards on Hugging Face)

The model cards retain the authoritative license, citation, limitations, and
attribution for the weights.

## bg0

The CLI's processing stages (512 px BiRefNet input, sigmoid mask, optional
padded subject-crop refinement, transparent PNG output) are modeled on
[bg0](https://github.com/opencoredev/bg0), which is Apache-2.0 licensed. This
project is an independent Python implementation; no bg0 source code is bundled.

## Runtime dependencies

- **ONNX Runtime** — MIT License — https://github.com/microsoft/onnxruntime
- **NumPy** — BSD-3-Clause — https://numpy.org
- **Pillow** — HPND License — https://python-pillow.org
- **pillow-heif** — BSD-3-Clause; bundles libheif (LGPL-3.0-or-later) and
  libde265 (LGPL-3.0-or-later) — https://github.com/bigcat88/pillow_heif

The exact resolved versions and their licenses are in `uv.lock` and the
installed environment metadata.
