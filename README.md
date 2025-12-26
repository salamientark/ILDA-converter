# ILDA Converter

[![CI](https://github.com/salamientark/ILDA-converter/actions/workflows/ci.yml/badge.svg)](https://github.com/salamientark/ILDA-converter/actions/workflows/ci.yml)

A Python toolkit for converting bitmap and vector images to ILDA format for laser show control systems.

## Overview

ILDA (International Laser Display Association) format is the industry standard for laser show control data. This project provides scripts to convert standard image formats (bitmap and vector) into ILDA files that can be used directly with laser projection systems.

## Project Goals

- **Bitmap Processing**: Convert raster images (PNG, JPG, etc.) to ILDA format through preprocessing, edge detection, and vectorization
- **Vector Processing**: Convert SVG vector files directly to ILDA format
- **CLI Interface**: Command-line tools for batch processing and automation

## Current Status

**Implemented:**
- ✓ Bitmap preprocessing with multiple thresholding algorithms
  - Binary thresholding
  - Adaptive mean thresholding
  - Adaptive gaussian thresholding

**In Development:**
- Edge detection and vectorization pipeline
- ILDA file format writer
- SVG to ILDA conversion

## Features

### Bitmap Preprocessing

The bitmap preprocessing module provides three different thresholding methods to convert color/grayscale images to black and white, preparing them for vectorization:

- **Binary Threshold**: Simple fixed threshold at 127
- **Adaptive Mean Threshold**: Local mean-based adaptive thresholding
- **Adaptive Gaussian Threshold**: Gaussian-weighted local thresholding

Each method produces different results depending on the input image characteristics, allowing you to choose the best preprocessing approach for your specific image.

## Installation

### Requirements

- Python >= 3.14
- uv (recommended) or pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd ilda

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## Usage

### Bitmap Preprocessing

```bash
# Run preprocessing on an image
python main.py --input path/to/image.jpg

# Outputs will be saved to data/smiley/preprocessing/ directory:
# - binary_image.jpg
# - mean_threshold_image.jpg
# - gaussian_threshold_image.jpg
```

### Python API

```python
from src.bitmap.preprocessing import binary_img, mean_tresh_img, gaussian_tresh_img

# Apply binary thresholding
binary_result = binary_img("path/to/image.jpg")

# Apply adaptive mean thresholding
mean_result = mean_tresh_img("path/to/image.jpg")

# Apply adaptive gaussian thresholding
gaussian_result = gaussian_tresh_img("path/to/image.jpg")
```

## Technical Pipeline

### Planned Processing Pipeline

1. **Input Stage**
   - Bitmap: PNG, JPG, BMP → Preprocessing → Edge Detection → Vectorization
   - Vector: SVG → Path Extraction

2. **Conversion Stage**
   - Vector paths → ILDA coordinates
   - Optimization (path ordering, blanking, etc.)

3. **Output Stage**
   - ILDA file generation
   - Format validation

## Roadmap

1. **Phase 1: Bitmap Processing** (Current)
   - [x] Image preprocessing with multiple threshold algorithms
   - [ ] Edge detection (Canny, Sobel)
   - [ ] Contour extraction
   - [ ] Path optimization

2. **Phase 2: ILDA Writer**
   - [ ] ILDA format specification implementation
   - [ ] Coordinate system conversion
   - [ ] Point sequence optimization
   - [ ] File writer with proper headers

3. **Phase 3: Vector Input**
   - [ ] SVG parser
   - [ ] Path extraction from SVG elements
   - [ ] Curve approximation
   - [ ] Direct SVG to ILDA conversion

4. **Phase 4: Optimization**
   - [ ] Blanking optimization
   - [ ] Scan speed optimization
   - [ ] Color support
   - [ ] Animation frame support

## Dependencies

- **opencv-python**: Image processing and computer vision operations
- **matplotlib**: Visualization and plotting (development)
- **ruff**: Code formatting and linting

## Contributing

Contributions are welcome! This is a technical project aimed at laser control applications.

## License

MIT License - see LICENSE file for details.

## References

- [ILDA Image Data Transfer Format Specification](http://www.laserist.org/StandardsDocs/ILDA_IDTF14_rev011.pdf)
- OpenCV Documentation: https://docs.opencv.org/
