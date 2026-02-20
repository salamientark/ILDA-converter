"""
Linear interpolation of a laser path.

Inserts intermediate points between consecutive points whose Euclidean
distance exceeds *max_step*, enforcing a constant apparent velocity at a
fixed PPS playback rate.
"""

from __future__ import annotations

import math

from src.logger.logging_config import get_logger
from src.postprocessing.laser_path import LaserPath
from src.postprocessing.laser_point import LaserPoint

logger = get_logger(__name__)


def resample_path(path: LaserPath, *, max_step: float = 50.0) -> LaserPath:
    """Insert intermediate points so no consecutive pair exceeds *max_step*.

    Blanking segments are interpolated too — galvos need time to travel even
    when the laser is off.  Intermediate points inherit the ``status`` and
    colour of the preceding point (colour does not change within a segment).

    Args:
        path: Input laser path (list of LaserPoint).
        max_step: Maximum Euclidean distance allowed between consecutive output
            points.  Defaults to 50.0.

    Returns:
        A new LaserPath with all gaps capped at *max_step*.
    """
    if max_step <= 0:
        raise ValueError(f"max_step must be positive, got {max_step}")
    if len(path) < 2:
        return list(path)

    out: list[LaserPoint] = []

    for a, b in zip(path, path[1:]):
        d = math.hypot(b.x - a.x, b.y - a.y)
        out.append(a)
        if d > max_step:
            n = math.ceil(d / max_step) - 1
            for i in range(1, n + 1):
                t = i / (n + 1)
                out.append(
                    LaserPoint(
                        x=a.x + t * (b.x - a.x),
                        y=a.y + t * (b.y - a.y),
                        r=a.r,
                        g=a.g,
                        b=a.b,
                        status=a.status,
                    )
                )

    out.append(path[-1])

    logger.debug(
        "resample_path: %d → %d points (max_step=%.3g)",
        len(path),
        len(out),
        max_step,
    )
    return out
