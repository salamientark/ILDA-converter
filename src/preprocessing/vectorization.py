"""
Vectorization module for converting binary images to vector paths using Potrace.

Provides predefined Potrace configurations optimized for different use cases
(default, high quality, smooth, fast) and a wrapper function for vectorization.
"""

from typing import Any

import cv2
import potrace

from src.logger.logging_config import get_logger

logger = get_logger(__name__)

POTRACE_CONFIGS = {
    "default": {
        "turdsize": 2,
        "turnpolicy": potrace.POTRACE_TURNPOLICY_MINORITY,
        "alphamax": 1.0,
        "opticurve": True,
        "opttolerance": 0.2,
    },
    "high_quality": {
        "turdsize": 0,  # No despeckle
        "turnpolicy": potrace.POTRACE_TURNPOLICY_MINORITY,
        "alphamax": 0.5,  # More corners
        "opticurve": True,
        "opttolerance": 0.1,  # Tighter curves
    },
    "smooth": {
        "turdsize": 10,  # Remove small speckles
        "turnpolicy": potrace.POTRACE_TURNPOLICY_MINORITY,
        "alphamax": 1.3333,  # Fewer corners
        "opticurve": True,
        "opttolerance": 0.5,  # Looser curves
    },
    "fast": {
        "turdsize": 2,
        "turnpolicy": potrace.POTRACE_TURNPOLICY_MINORITY,
        "alphamax": 1.0,
        "opticurve": False,  # Skip optimization
        "opttolerance": 0.2,
    },
}


def vectorize_img(
    img: cv2.typing.MatLike, config: dict[str, Any] | None
) -> potrace.Path:
    """
    Convert a binary image to a vector path using Potrace.

    Wraps the Potrace bitmap tracing functionality with optional configuration
    for controlling corner detection, curve optimization, and despeckling.

    Parameters:
        img (cv2.typing.MatLike): Binary input image (black and white).
        config (dict[str, Any] | None): Potrace configuration dictionary with
            parameters like turdsize, turnpolicy, alphamax, etc. Use predefined
            configs from POTRACE_CONFIGS or None for defaults.

    Returns:
        potrace.Path: Vectorized path representation of the binary image.
    """
    bitmap = potrace.Bitmap(img)
    if config is None:
        path = bitmap.trace()
    else:
        path = bitmap.trace(**config)

    logger.debug("Image vectorized")
    return path
