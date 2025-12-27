import potrace
import cv2
from typing import Optional

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


def vectorize_img(img: cv2.typing.MatLike, config: Optional[any]) -> potrace.Path:
    """
    Convert a bitmap image to potrace Path object for the specified config.

    Parameters:
    """
    bitmap = potrace.Bitmap(img)
    if config is None:
        return bitmap.trace()
    return bitmap.trace(**config)
