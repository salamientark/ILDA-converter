"""
Blanking anchor insertion for laser paths.

Galvo mirrors have two sync problems at blanking transitions:
- Lead-out: the mirror drifts after the last visible point.  Extra visible
  copies let it decelerate cleanly before the laser shuts off.
- Lead-in: after a blanking jump the mirror overshoots the target.  Extra
  blanking copies dwell at the destination so the mirror settles before the
  laser turns on.
"""

from __future__ import annotations

from src.logger.logging_config import get_logger
from src.postprocessing.laser_path import LaserPath
from src.postprocessing.laser_point import LaserPoint

_log = get_logger(__name__)


def add_blanking_anchors(path: LaserPath, *, repeats: int = 4) -> LaserPath:
    """Insert lead-out and lead-in anchor copies around blanking transitions.

    For each visible point adjacent to a blanking segment:

    - **Lead-out**: when a visible point is immediately followed by a blanking
      point, ``repeats`` extra visible copies of that point are appended after
      it so the galvo mirror can decelerate before the laser shuts off.
    - **Lead-in**: when a visible point is immediately preceded by a blanking
      point, ``repeats`` blanking copies at that point's coordinates are
      prepended before it so the mirror can settle before the laser turns on.

    Args:
        path: Input laser path (list of LaserPoint).
        repeats: Number of anchor copies inserted at each transition.
            Defaults to 4.

    Returns:
        A new LaserPath with anchor points inserted at blanking transitions.

    Raises:
        ValueError: If *repeats* is not positive.
    """
    if repeats <= 0:
        raise ValueError(f"repeats must be positive, got {repeats}")

    n = len(path)
    out: list[LaserPoint] = []
    lead_out_added = 0
    lead_in_added = 0

    for i, pt in enumerate(path):
        next_is_blanking = i + 1 < n and path[i + 1].is_blanking
        prev_is_blanking = i > 0 and path[i - 1].is_blanking

        if not pt.is_blanking and prev_is_blanking:
            # Lead-in: prepend blanking copies before the first visible point
            for _ in range(repeats):
                out.append(LaserPoint(x=pt.x, y=pt.y, r=0, g=0, b=0, status=1))
            lead_in_added += repeats

        out.append(pt)

        if not pt.is_blanking and next_is_blanking:
            # Lead-out: append visible copies after the last visible point
            for _ in range(repeats):
                out.append(LaserPoint(x=pt.x, y=pt.y, r=pt.r, g=pt.g, b=pt.b, status=0))
            lead_out_added += repeats

    _log.debug(
        "add_blanking_anchors: %d → %d points (%d lead-out, %d lead-in copies added)",
        len(path),
        len(out),
        lead_out_added,
        lead_in_added,
    )
    return out
