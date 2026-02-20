"""
Corner dwell insertion for laser paths.

Galvo mirrors have inertia — at sharp corners they overshoot unless the laser
dwells on the vertex point for a few extra frames.  The dwell count is
proportional to corner sharpness: a U-turn (0°) gets *max_dwell* extra copies;
a straight line (180°) gets none.
"""

from __future__ import annotations

import math

from src.logger.logging_config import get_logger
from src.postprocessing.laser_path import LaserPath
from src.postprocessing.laser_point import LaserPoint

_log = get_logger(__name__)


def _vertex_angle_deg(
    prev: LaserPoint, pt: LaserPoint, nxt: LaserPoint
) -> float | None:
    """Return the opening angle in degrees at *pt*, or None for degenerate segments."""
    ax = prev.x - pt.x
    ay = prev.y - pt.y
    bx = nxt.x - pt.x
    by = nxt.y - pt.y
    mag_a = math.hypot(ax, ay)
    mag_b = math.hypot(bx, by)
    if mag_a == 0.0 or mag_b == 0.0:
        return None
    cos_angle = max(-1.0, min(1.0, (ax * bx + ay * by) / (mag_a * mag_b)))
    return math.degrees(math.acos(cos_angle))


def apply_corner_dwell(
    path: LaserPath,
    *,
    max_dwell: int = 8,
) -> LaserPath:
    """Insert dwell copies at sharp corners to compensate for galvo inertia.

    For each interior, non-blanking point whose neighbours are also non-blanking,
    the opening angle at the vertex is measured and ``dwell_count`` extra copies
    of the point are appended immediately after it::

        dwell_count = round(max_dwell * (1 - angle / 180))

    A straight line (angle = 180°) adds no copies; a U-turn (angle ≈ 0°) adds
    *max_dwell* copies.

    Args:
        path: Input laser path (list of LaserPoint).
        max_dwell: Maximum number of dwell copies inserted at a single vertex.
            Defaults to 8.

    Returns:
        A new LaserPath with dwell points inserted at corners.

    Raises:
        ValueError: If *max_dwell* is not positive.
    """
    if max_dwell <= 0:
        raise ValueError(f"max_dwell must be non-negative, got {max_dwell}")
    if len(path) < 3:
        return list(path)

    out: list[LaserPoint] = []
    dwell_added = 0

    for i, pt in enumerate(path):
        out.append(pt)

        # Skip blanking points, first, and last
        if pt.is_blanking or i == 0 or i == len(path) - 1:
            continue

        prev = path[i - 1]
        nxt = path[i + 1]

        # Skip if either neighbour is blanking
        if prev.is_blanking or nxt.status != 0:
            continue

        angle = _vertex_angle_deg(prev, pt, nxt)
        if angle is None:
            continue

        dwell_count = round(max_dwell * (1 - angle / 180))

        for _ in range(dwell_count):
            out.append(
                LaserPoint(x=pt.x, y=pt.y, r=pt.r, g=pt.g, b=pt.b, status=pt.status)
            )

        dwell_added += dwell_count

    _log.debug(
        "apply_corner_dwell: %d → %d points (%d dwell points added)",
        len(path),
        len(out),
        dwell_added,
    )
    return out
