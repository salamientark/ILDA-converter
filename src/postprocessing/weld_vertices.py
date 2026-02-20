"""
Vertex welding for polyline endpoints.

Snaps numerically near-identical endpoints to a common coordinate so that
downstream graph algorithms can treat them as the same node.
"""

from __future__ import annotations

import math

from src.logger.logging_config import get_logger

logger = get_logger(__name__)


def _collect_endpoints(
    polylines: list[list[tuple[float, float]]],
) -> list[tuple[int, str, float, float]]:
    """Return (poly_idx, role, x, y) for the start/end of every non-empty polyline."""
    endpoints: list[tuple[int, str, float, float]] = []
    for i, poly in enumerate(polylines):
        if not poly:
            continue
        endpoints.append((i, "start", poly[0][0], poly[0][1]))
        if len(poly) > 1:
            endpoints.append((i, "end", poly[-1][0], poly[-1][1]))
    return endpoints


def _snap_endpoints(
    endpoints: list[tuple[int, str, float, float]],
    threshold: float,
) -> tuple[
    list[tuple[float, float]],
    dict[tuple[int, str], tuple[float, float]],
    int,
]:
    """Build canonical coords and record per-endpoint snaps.

    Returns:
        ``(canonicals, snapped_coords, snap_count)`` where *canonicals* is the
        ordered list of first-seen unique coordinates, *snapped_coords* maps
        ``(poly_idx, role)`` to its canonical coordinate, and *snap_count* is
        the number of endpoints that were moved.
    """
    canonicals: list[tuple[float, float]] = []
    # Grid bucket: cell size == threshold so neighbours live in the 3×3 surrounding cells.
    grid: dict[tuple[int, int], list[tuple[float, float]]] = {}
    snapped_coords: dict[tuple[int, str], tuple[float, float]] = {}
    snap_count = 0

    def _bucket(px: float, py: float) -> tuple[int, int]:
        return (int(math.floor(px / threshold)), int(math.floor(py / threshold)))

    for poly_idx, role, x, y in endpoints:
        bx, by = _bucket(x, y)
        matched: tuple[float, float] | None = None
        for nx in range(bx - 1, bx + 2):
            if matched is not None:
                break
            for ny in range(by - 1, by + 2):
                for cx, cy in grid.get((nx, ny), []):
                    if math.hypot(x - cx, y - cy) < threshold:
                        matched = (cx, cy)
                        break
                if matched is not None:
                    break
        if matched is None:
            canonicals.append((x, y))
            grid.setdefault(_bucket(x, y), []).append((x, y))
            snapped_coords[(poly_idx, role)] = (x, y)
        else:
            snapped_coords[(poly_idx, role)] = matched
            if matched != (x, y):
                snap_count += 1

    return canonicals, snapped_coords, snap_count


def _rebuild_polylines(
    polylines: list[list[tuple[float, float]]],
    snapped_coords: dict[tuple[int, str], tuple[float, float]],
) -> list[list[tuple[float, float]]]:
    """Return a new polyline list with endpoints replaced by their canonical coords."""
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
    if threshold <= 0:
        raise ValueError("Threshold must be positive")
    endpoints = _collect_endpoints(polylines)
    _canonicals, snapped_coords, snap_count = _snap_endpoints(endpoints, threshold)
    logger.debug(
        "weld_vertices: %d endpoints snapped (threshold=%.3g)", snap_count, threshold
    )
    return _rebuild_polylines(polylines, snapped_coords)
