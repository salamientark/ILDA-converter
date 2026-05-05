# ILDA Converter

[![CI](https://github.com/salamientark/ILDA-converter/actions/workflows/ci.yml/badge.svg)](https://github.com/salamientark/ILDA-converter/actions/workflows/ci.yml)

A Python toolkit that turns ordinary bitmap images into [ILDA](http://www.laserist.org/StandardsDocs/ILDA_IDTF14_rev011.pdf) files ready to drive a laser projector. It runs a full pipeline — preprocessing, vectorization, path optimization, and ILDA encoding — in a single command.

## Features

- **End-to-end pipeline**: bitmap → binary mask → vector polylines → optimized laser path → `.ild` file.
- **Multiple thresholding methods**: binary, adaptive mean, adaptive Gaussian, and Otsu.
- **Vectorization backends**: [Potrace](https://potrace.sourceforge.net/) (high quality) and an OpenCV contour approximator (fast).
- **Laser-aware post-processing**: vertex welding, Eulerian path stitching, resampling, corner dwell, blanking anchors, and galvo/color signal shifting.
- **Inspectable artifacts**: every pipeline stage saves intermediate PBM, SVG, and ILDA files for tuning and debugging.
- **Debug tooling**: scripts to count points and render `.ild` files back to SVG for visual inspection.

## Pipeline overview

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

Each stage writes its output to `data/<image_name>/...` so you can compare results side by side.

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

Outputs are written under `data/<image_name>/`:

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

Lower-level building blocks live in `src/preprocessing`, `src/vectorization`, `src/postprocessing`, and `src/ilda` — each is usable on its own.

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

> [!NOTE]
> See [`docs/`](./docs) for the ILDA specification and the laser-projection paper that informed several optimization stages.

## References

- [ILDA Image Data Transfer Format Specification](http://www.laserist.org/StandardsDocs/ILDA_IDTF14_rev011.pdf)
- [ILDA Test Pattern 95](./docs/ILDA_TestPattern95_rev002.pdf)
- *Accurate and Efficient Drawing Method for Laser Projection* (see `docs/`)
- [OpenCV documentation](https://docs.opencv.org/)
- [Potrace algorithm](https://potrace.sourceforge.net/potrace.pdf)
