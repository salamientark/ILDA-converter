"""
LaserPath type alias and conversion helpers.
"""

from __future__ import annotations

from src.logger.logging_config import get_logger
from src.postprocessing.laser_point import LaserPoint

logger = get_logger(__name__)

# A laser path is an ordered sequence of LaserPoints.
LaserPath = list[LaserPoint]


def from_polylines(
    polylines: list[list[tuple[float, float]]],
    *,
    r: int = 0,
    g: int = 0,
    b: int = 0,
) -> LaserPath:
    """Convert the existing polylines structure to a LaserPath.

    All points within every polyline are marked as visible (status=0).
    Blanking travel segments between polylines are *not* inserted here —
    that is the responsibility of the Eulerian path algorithm (Phase 1).

    Args:
        polylines: Nested list of (x, y) tuples representing polylines.
        r: Default red channel for all points (0–255).
        g: Default green channel for all points (0–255).
        b: Default blue channel for all points (0–255).

    Returns:
        A flat LaserPath of visible LaserPoints.

    Raises:
        ValueError: If any point does not have exactly two coordinates.
    """
    path: LaserPath = []
    for polyline in polylines:
        for point in polyline:
            if len(point) != 2:
                raise ValueError(
                    f"Each point must have exactly 2 coordinates, got {len(point)}: {point!r}"
                )
            path.append(LaserPoint.from_xy(point[0], point[1], r=r, g=g, b=b, status=0))
    logger.debug(
        "from_polylines: converted %d polylines → %d points", len(polylines), len(path)
    )
    return path
