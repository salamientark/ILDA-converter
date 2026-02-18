from __future__ import annotations

import math
import tempfile
import webbrowser
from pathlib import Path


def _format_svg_number(value: float, *, decimals: int) -> str:
    """Format numeric values for compact SVG output.

    Parameters:
        value (float): Numeric value to format.
        decimals (int): Maximum number of decimal digits.

    Returns:
        str: A compact string representation (e.g. ``"12.3"`` instead of ``"12.300"``).
    """
    if decimals <= 0:
        as_int = int(value)
        return str(as_int) if float(as_int) == value else str(value)

    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".")


def _polylines_bounds(
    polylines: list[list[tuple[float, float]]],
) -> tuple[float, float, float, float]:
    """Compute a bounding box for a list of polylines.

    Parameters:
        polylines (list[list[tuple[float, float]]]): Polylines as lists of (x, y) points.

    Returns:
        tuple[float, float, float, float]: ``(min_x, min_y, max_x, max_y)``.

    Raises:
        ValueError: If ``polylines`` contains no points.
    """
    min_x = math.inf
    min_y = math.inf
    max_x = -math.inf
    max_y = -math.inf

    any_point = False
    for polyline in polylines:
        for x, y in polyline:
            any_point = True
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

    if not any_point:
        raise ValueError("No points in `polylines` (empty input).")

    return min_x, min_y, max_x, max_y


def draw_svg(
    polylines: list[list[tuple[float, float]]],
    *,
    width: int | None = None,
    height: int | None = None,
    padding: float = 10.0,
    stroke: str = "black",
    stroke_width: float = 1.0,
    point_radius: float = 2.0,
    point_fill: str = "red",
    decimals: int = 2,
    output_path: str | Path | None = None,
    open_in_browser: bool = True,
) -> Path:
    """Write an SVG for polylines and open it in a browser.

    This is a small debug helper meant to visualize the output of the vectorization
    stage. It draws each polyline as an SVG ``<polyline>`` and emphasizes every
    vertex with a small ``<circle>``.

    Parameters:
        polylines (list[list[tuple[float, float]]]): Polylines as lists of (x, y) points.
        width (int | None): Output SVG width in pixels. If ``None``, computed from bounds.
        height (int | None): Output SVG height in pixels. If ``None``, computed from bounds.
        padding (float): Margin added around the computed bounds (SVG units).
        stroke (str): Polyline stroke color.
        stroke_width (float): Polyline stroke width (SVG units).
        point_radius (float): Circle radius for each vertex (SVG units).
        point_fill (str): Circle fill color for each vertex.
        decimals (int): Maximum number of decimals for coordinates.
        output_path (str | Path | None): Where to write the SVG. If ``None``, uses a temp file.
        open_in_browser (bool): If ``True``, opens the SVG file in the default browser.

    Returns:
        Path: The written SVG file path.

    Raises:
        ValueError: If ``polylines`` contains no points.
    """
    min_x, min_y, max_x, max_y = _polylines_bounds(polylines)

    view_x = min_x - padding
    view_y = min_y - padding
    view_w = (max_x - min_x) + 2 * padding
    view_h = (max_y - min_y) + 2 * padding

    # Avoid a degenerate viewBox (invalid SVG).
    view_w = max(view_w, 1.0)
    view_h = max(view_h, 1.0)

    if width is None:
        width = int(math.ceil(view_w))
    if height is None:
        height = int(math.ceil(view_h))

    def fmt(value: float) -> str:
        return _format_svg_number(value, decimals=decimals)

    svg_lines: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="{fmt(view_x)} {fmt(view_y)} {fmt(view_w)} {fmt(view_h)}">',
        f'  <g fill="none" stroke="{stroke}" stroke-width="{stroke_width}" stroke-linecap="round">',
    ]

    for polyline in polylines:
        if not polyline:
            continue

        points_attr = " ".join(f"{fmt(x)},{fmt(y)}" for x, y in polyline)
        svg_lines.append(f'    <polyline points="{points_attr}" />')

    svg_lines.append("  </g>")
    svg_lines.append(f'  <g fill="{point_fill}">')

    for polyline in polylines:
        for x, y in polyline:
            svg_lines.append(
                f'    <circle cx="{fmt(x)}" cy="{fmt(y)}" r="{point_radius}" />'
            )

    svg_lines.append("  </g>")
    svg_lines.append("</svg>")

    svg_text = "\n".join(svg_lines)

    if output_path is None:
        tmp_file = tempfile.NamedTemporaryFile(
            prefix="ilda_",
            suffix=".svg",
            delete=False,
        )
        path = Path(tmp_file.name)
        tmp_file.close()
    else:
        path = Path(output_path)

    path.write_text(svg_text, encoding="utf-8")

    if open_in_browser:
        webbrowser.open(path.resolve().as_uri())

    return path
