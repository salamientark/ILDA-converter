"""
Laser path optimization pipeline.

Algorithms are applied sequentially. Later phases will add:
  Phase 2: resample_path, apply_corner_dwell
  Phase 3: add_blanking_anchors, shift_color_signal
"""

from __future__ import annotations

from src.logger.logging_config import get_logger
from src.postprocessing.corner_dwell import apply_corner_dwell
from src.postprocessing.find_eulerian_path import find_eulerian_path
from src.postprocessing.laser_path import LaserPath
from src.postprocessing.resample_path import resample_path
from src.postprocessing.weld_vertices import weld_vertices

logger = get_logger(__name__)


def optimize(
    polylines: list[list[tuple[float, float]]],
    *,
    r: int = 0,
    g: int = 0,
    b: int = 0,
    max_step: float = 50.0,
    max_dwell: int | None = None,
) -> LaserPath:
    """Run the full optimization pipeline on a polylines structure.

    Phase 1 welds near-coincident endpoints then builds an Eulerian path so
    all polylines are covered in one continuous laser sweep with minimal
    blanking travel.

    Phase 2 resamples the path at constant velocity and, when *max_dwell* is
    provided, inserts dwell copies at sharp corners to compensate for galvo
    inertia (see :func:`apply_corner_dwell`).

    Args:
        polylines: Nested list of (x, y) tuples representing polylines.
        r: Default red channel for all points (0–255).
        g: Default green channel for all points (0–255).
        b: Default blue channel for all points (0–255).
        max_step: Maximum Euclidean distance between consecutive output points.
            Defaults to 50.0.
        max_dwell: Maximum number of dwell copies inserted at a single corner
            vertex.  When ``None`` (default) corner dwell is skipped entirely.

    Returns:
        An optimized LaserPath.
    """
    logger.debug("optimize: start, %d polylines", len(polylines))

    # Phase 1.1: snap near-coincident endpoints
    polylines = weld_vertices(polylines)

    path = LaserPath.from_polylines(polylines)

    # Phase 1.2: Eulerian path — single continuous sweep with blanking jumps
    path = find_eulerian_path(polylines, r=r, g=g, b=b)

    # Phase 2.1: constant-velocity resampling
    path = resample_path(path, max_step=max_step)

    # Phase 2.2: corner dwell (optional)
    if max_dwell is not None:
        path = apply_corner_dwell(path, max_dwell=max_dwell)

    # Phase 3: add_blanking_anchors, shift_color_signal

    logger.debug("optimize: done, %d points", len(path))
    return path
