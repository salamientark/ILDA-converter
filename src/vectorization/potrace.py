from typing import Any
from src.logger.logging_config import get_logger

import potrace as potrace

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

def path_to_polylines(path: potrace.Path) -> list[list[tuple[float, float]]]:
    """
    Convert a potrace path to a list of polylines.
    Parameters:
        path (potrace.Path): The potrace path to convert.
    Returns:
        list[list[tuple[float, float]]]: List of polylines as lists of (x, y) points.
    """
    polylines: list[list[tuple[float, float]]] = []
    for curve in path:
        points: list[tuple[float, float]] = []
        start = curve.start_point
        points.append((float(start.x), float(start.y)))
        for segment in curve:
            if segment.is_corner:
                c = segment.c
                end = segment.end_point
                points.append((float(c.x), float(c.y)))
                points.append((float(end.x), float(end.y)))
            else:
                seg = segment._segment
                for point in seg.c:
                    points.append((float(point.x), float(point.y)))
        polylines.append(points)
    return polylines

def vectorize(img: Any, config: dict[str, Any] | None) -> list[list[tuple[float, float]]]:
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

    polylines = path_to_polylines(path)
    logger.debug("Image vectorized")
    return polylines
