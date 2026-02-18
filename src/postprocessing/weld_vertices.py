"""
Vertex welding for polyline endpoints.

Snaps numerically near-identical endpoints to a common coordinate so that
downstream graph algorithms can treat them as the same node.
"""

from __future__ import annotations

import math

from src.logger.logging_config import get_logger

logger = get_logger(__name__)


def weld_vertices(
    polylines: list[list[tuple[float, float]]],
    *,
    threshold: float = 5.0,
) -> list[list[tuple[float, float]]]:
    """Snap near-coincident polyline endpoints to a shared canonical coordinate.

    Only the first and last point of each polyline are considered for welding;
    interior points are left untouched.  Two endpoints are welded when their
    Euclidean distance is strictly less than *threshold*.  The first-encountered
    coordinate becomes the canonical value — no midpoint averaging — so the
    result is deterministic for a given input ordering.

    Args:
        polylines: Nested list of (x, y) tuples representing polylines.
        threshold: Maximum distance (in the same units as the coordinates)
            at which two endpoints are merged.  Defaults to 5.0.

    Returns:
        A new ``list[list[tuple[float, float]]]`` with snapped endpoints.
        The input is never mutated.
    """
    # --- 1. Collect all endpoints ----------------------------------------
    # Each entry: (poly_idx, role, x, y)  where role is "start" or "end".
    endpoints: list[tuple[int, str, float, float]] = []
    for i, poly in enumerate(polylines):
        if not poly:
            continue
        endpoints.append((i, "start", poly[0][0], poly[0][1]))
        if len(poly) > 1:
            endpoints.append((i, "end", poly[-1][0], poly[-1][1]))

    # --- 2. Build canonical list & record snaps --------------------------
    # canonicals: list of (cx, cy)
    canonicals: list[tuple[float, float]] = []
    # snapped_coords[(poly_idx, role)] = (cx, cy)
    snapped_coords: dict[tuple[int, str], tuple[float, float]] = {}
    snap_count = 0

    for poly_idx, role, x, y in endpoints:
        matched: tuple[float, float] | None = None
        for cx, cy in canonicals:
            dist = math.hypot(x - cx, y - cy)
            if dist < threshold:
                matched = (cx, cy)
                break
        if matched is None:
            canonicals.append((x, y))
            snapped_coords[(poly_idx, role)] = (x, y)
        else:
            snapped_coords[(poly_idx, role)] = matched
            if matched != (x, y):
                snap_count += 1

    logger.debug("weld_vertices: %d endpoints snapped (threshold=%.3g)", snap_count, threshold)

    # --- 3. Rebuild polylines with snapped endpoints ---------------------
    result: list[list[tuple[float, float]]] = []
    for i, poly in enumerate(polylines):
        if not poly:
            result.append(list(poly))
            continue
        new_poly = list(poly)
        new_poly[0] = snapped_coords.get((i, "start"), poly[0])
        if len(poly) > 1:
            new_poly[-1] = snapped_coords.get((i, "end"), poly[-1])
        result.append(new_poly)

    return result
