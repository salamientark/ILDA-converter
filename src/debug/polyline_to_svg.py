def polyline_to_svg(
    polylines: list[list[tuple[float, float]]], width: int, height: int
) -> list[str]:
    """Convert a list of polylines to a minimal SVG.

    Parameters:
        polylines (list[list[tuple[float, float]]]): List of polylines. Each polyline
            is a list of `(x, y)` points.
        width (int): Output SVG width.
        height (int): Output SVG height.

    Returns:
        list[str]: SVG lines suitable for writing with `"\n".join(...)`.
    """
    parts: list[str] = []

    parts.append(
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    )
    parts.append('<path d="')

    for line in polylines:
        start_x, start_y = line[0]
        parts.append(f"M {start_x},{start_y}")

        for x, y in line[1:]:
            parts.append(f"L {x},{y}")

        parts.append("Z")

    parts.append('" stroke="black" fill="none"/>')
    parts.append("</svg>")

    return parts
