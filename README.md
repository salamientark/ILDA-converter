# ILDA Converter

[![CI](https://github.com/salamientark/ILDA-converter/actions/workflows/ci.yml/badge.svg)](https://github.com/salamientark/ILDA-converter/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.14%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

Convert ordinary bitmap images into [ILDA](http://www.laserist.org/StandardsDocs/ILDA_IDTF14_rev011.pdf) files ready to drive a laser projector. A single command runs the full pipeline: preprocessing, vectorization, path optimization, and ILDA encoding.

> [!NOTE]
> Every stage writes inspectable artifacts (PBM, SVG, ILD) under `data/<image>/`, so you can tune and debug each step.

## Features

- **End-to-end pipeline** — bitmap → binary mask → vector polylines → optimized laser path → `.ild`.
- **Multiple thresholding methods** — binary, adaptive mean, adaptive Gaussian, Otsu (or `all`).
- **Two vectorization backends** — [Potrace](https://potrace.sourceforge.net/) for quality, OpenCV contour approximation for speed.
- **Laser-aware post-processing** — vertex welding, Eulerian path stitching, resampling, corner dwell, blanking anchors, galvo/color signal shifting.
- **Debug tooling** — render `.ild` back to SVG, inspect point/frame counts.

## Pipeline

```
input.png ─▶ preprocessing ─▶ vectorization ─▶ weld ─▶ Eulerian path
                                                       │
                              ┌────────────────────────┘
                              ▼
                        resample ─▶ corner dwell ─▶ blanking anchors
                              │
                              ▼
                        color shift ─▶ output.ild
```

## Requirements

- Python **>= 3.14**
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

## Installation

```bash
git clone https://github.com/salamientark/ILDA-converter.git
cd ILDA-converter

# With uv
uv sync

# Or with pip
pip install -e .
```

## Usage

### CLI

```bash
python main.py --input path/to/image.png \
               --preprocessing binary \
               --vector-config fast
```

| Flag | Choices | Default | Description |
|------|---------|---------|-------------|
| `--input` | path | *required* | Source bitmap (PNG/JPG/BMP/...). |
| `--preprocessing`, `-p` | `binary`, `gaussian`, `mean`, `otsu`, `all` | `binary` | Thresholding method. `all` runs every method. |
| `--vector-config`, `-v` | `default`, `fast`, `high`, `smooth`, `all` | `fast` | Potrace vectorization preset. |
| `--output` | path | — | Reserved for future use. |

Outputs land in `data/<image_name>/`:

```
data/<name>/
├── preprocessing/    # binary masks (.pbm)
├── svg/              # vectorized output (.svg)
├── ilda/             # raw ILDA file (.ild)
└── optimizer/        # post-processed ILDA at each optimization stage
```

### Python API

```python
from src.pipeline.orchestrator import run_pipeline

run_pipeline(
    input="data/smiley.png",
    preprocessing="otsu",
    vectorization="high",
)
```

> [!TIP]
> Lower-level building blocks live in `src/preprocessing`, `src/vectorization`, `src/postprocessing`, and `src/ilda` — each is usable on its own.

### Debug scripts

```bash
# Render a generated .ild back to SVG and open it in the browser
python scripts/draw_ilda.py data/smiley/ilda/binary_smiley_fast.ild

# Print point/frame statistics for a file
python scripts/count_ilda_points.py data/smiley/optimizer/binary_smiley_fast_optimized.ild
```

## Project layout

```
src/
├── preprocessing/    # thresholding (binary, mean, gaussian, otsu)
├── vectorization/    # potrace + opencv backends, configs
├── postprocessing/   # weld, Eulerian path, resample, dwell, blanking, color shift
├── ilda/             # ILDA format encoder (2D / 3D, frame separation)
├── pipeline/         # orchestrator wiring the stages together
├── logger/           # structured logging + Timer helpers
└── debug/            # SVG renderers used by debug scripts

docs/                 # ILDA spec and reference papers
scripts/              # CLI helpers to inspect generated .ild files
```

## Dependencies

- [`opencv-python`](https://pypi.org/project/opencv-python/) — image processing and contour extraction
- [`potracer`](https://pypi.org/project/potracer/) — Potrace bindings for vectorization
- [`numpy`](https://numpy.org/) — array math
- [`matplotlib`](https://matplotlib.org/) — visualization (debug only)

> [!WARNING]
> Laser projection can cause permanent eye damage. Always validate generated `.ild` output on a low-power setup before driving real galvos.

## References

- [ILDA Image Data Transfer Format Specification](http://www.laserist.org/StandardsDocs/ILDA_IDTF14_rev011.pdf)
- [ILDA Test Pattern 95](./docs/ILDA_TestPattern95_rev002.pdf)
- *Accurate and Efficient Drawing Method for Laser Projection* (see [`docs/`](./docs))
- [OpenCV documentation](https://docs.opencv.org/)
- [Potrace algorithm](https://potrace.sourceforge.net/potrace.pdf)
