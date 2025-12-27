"""
Preprocessing package for bitmap image processing and vectorization.

Provides thresholding functions for converting grayscale images to binary format
and Potrace-based vectorization utilities for converting binary images to vector paths.
"""

from .preprocessing import (
    binary_img,
    mean_thresh_img,
    gaussian_thresh_img,
    otsu_thresholding,
)

from .vectorization import POTRACE_CONFIGS, vectorize_img

__all__ = [
    # PREPROCESSING
    "binary_img",
    "mean_thresh_img",
    "gaussian_thresh_img",
    "otsu_thresholding",
    # VECTORIZATION
    "POTRACE_CONFIGS",
    "vectorize_img",
]
