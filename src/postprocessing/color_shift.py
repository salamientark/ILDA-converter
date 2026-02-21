"""
Color shift compensation for laser graphics.

Galvanometer mirrors have inertia and lag behind the electronic control signal.
The laser diode, however, responds almost instantly. This causes colors to
shift ahead of the intended coordinates. We compensate by shifting the color
signal backward (delaying it) relative to the X/Y coordinate path.
"""

from __future__ import annotations

from src.logger.logging_config import get_logger
from src.postprocessing.laser_path import LaserPath
from src.postprocessing.laser_point import LaserPoint

logger = get_logger(__name__)


def shift_color_signal(path: LaserPath, *, shift_amount: int = -2) -> LaserPath:
    """Shift the RGB color and blanking signal relative to the X/Y coordinates.

    Because scanners have mechanical inertia, they lag behind the DAC signal.
    The laser diodes are optical and have no inertia, so they respond instantly.
    By shifting the color/blanking signal backward (negative shift_amount),
    we delay the laser modulation so it aligns with the delayed scanner position.

    Args:
        path: Input laser path (list of LaserPoint).
        shift_amount: Number of points to shift the color signal. Negative values
            delay the color (the typical requirement). Defaults to -2.

    Returns:
        A new LaserPath with shifted color and status values.
    """
    if not path:
        return []

    if shift_amount == 0:
        return list(path)

    n = len(path)
    out: list[LaserPoint] = []

    for i, pt in enumerate(path):
        # Calculate the index of the point from which to take the color
        # Clamp to the bounds of the path [0, n - 1]
        color_idx = max(0, min(n - 1, i + shift_amount))
        color_pt = path[color_idx]

        out.append(
            LaserPoint(
                x=pt.x,
                y=pt.y,
                r=color_pt.r,
                g=color_pt.g,
                b=color_pt.b,
                status=color_pt.status,
            )
        )

    logger.debug(
        "shift_color_signal: shifted %d points by %d",
        len(out),
        shift_amount,
    )
    return out
