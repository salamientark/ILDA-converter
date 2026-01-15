"""Draw an ILDA file as an SVG polyline.

This script is a small debug helper that reads an ``.ild`` (ILDA format 0) file,
converts it into the project's polyline representation, and renders it via
:func:`src.debug.draw_svg.draw_svg`.

The output SVG is written to a temporary file and opened in your default browser.
"""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path


WIDTH=1280
HEIGHT=720


def _get_scale(
    polylines: list[list[tuple[float, float]]],
    *, 
    width: float = 1600,
    height: float = 900,
    ) -> float:
    """
    Compute a uniform scale factor to fit the polylines within the given width and height.

    Parameters:
        polylines (list[list[tuple[float, float]]]): List of polylines.
        width (float): Target width.
        height (float): Target height.

    Returns:
        float: Scale factor.
    """
    x_points, y_points = zip(*(point for polyline in polylines for point in polyline))

    min_x, max_x = min(x_points), max(x_points)
    min_y, max_y = min(y_points), max(y_points)
    polyline_width = max_x - min_x
    polyline_height = max_y - min_y

    if polyline_width <= 0 or polyline_height <= 0:
        raise ValueError("Polylines have zero width or height, cannot compute scale.")

    scale_x = width / polyline_width
    scale_y = height / polyline_height

    return min(scale_x, scale_y)


def _rescale_polyline(
    polylines: list[list[tuple[float, float]]],
    scale: float,
    invert_y: bool = True,
) -> list[list[tuple[float, float]]]:
    """
    Rescale polylines by a uniform scale factor.

    Parameters:
        polylines (list[list[tuple[float, float]]]): List of polylines.
        scale (float): Scale factor.
        invert_y (bool): Whether to invert the y-axis.

    Returns:
        list[list[tuple[float, float]]]: Rescaled polylines.
    """
    new_polylines = []
    y_impact = -1 if invert_y else 1
    for polyline in polylines:
        new_polyline = [(x * scale, y * scale * y_impact) for x, y in polyline]
        new_polylines.append(new_polyline)
    return new_polylines

def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    project_root_str = str(project_root)
    if project_root_str not in sys.path:
        sys.path.insert(0, project_root_str)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the ILDA drawing helper.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Draw an ILDA (.ild) file as an SVG")
    parser.add_argument("--input", required=True, help="Path to input ILDA file (.ild)")
    return parser.parse_args()


def main() -> None:
    """Main entry point for drawing an ILDA file."""
    _ensure_project_root_on_path()

    from src.debug import polyline_to_svg, draw_svg, get_polylines_info
    from src.ilda.ilda_to_polylines import ilda_to_polylines
    from src.ilda.ilda_3d import polylines_to_ilda
    from src.logger.logging_config import get_logger, setup_logging

    setup_logging()
    logger = get_logger(__name__)

    args = parse_args()
    input_path = Path(args.input)
    logger.info(f"Drawing ILDA file: {input_path}")

    try:
        data = input_path.read_bytes()
        polylines = ilda_to_polylines(data)
        scale = _get_scale(polylines, width=WIDTH, height=HEIGHT)


        print(f"Computed scale factor: {scale}")
        rescaled_polylines = _rescale_polyline(polylines, scale)
        point_nbr, polyline_nbr = get_polylines_info(rescaled_polylines)
        print(
            f"Rescaled polylines: {polyline_nbr} polylines, {point_nbr} points"
        )


        raw_svg_rescaled = polyline_to_svg(polylines, width=WIDTH, height=HEIGHT)

        # Save SVG to file
        os.makedirs("./tmp", exist_ok=True)
        filename = os.path.splitext(os.path.basename(args.input))[0]
        with open(
            f"./tmp/{filename}.svg", "w"
        ) as svg_file:
            svg_file.writelines("\n".join(raw_svg_rescaled))
            logger.info(
                f"Saved SVG: tmp/{filename}.svg"
            )

        # Save as ilda file
        rescaled_ilda, _, _, _ = polylines_to_ilda(
            rescaled_polylines
        )
        with open(f"./tmp/{filename}_rescaled.ild", "wb") as ilda_file:
            for chunk in rescaled_ilda:
                ilda_file.write(chunk)
            logger.info(f"Saved ILDA: tmp/{filename}_rescaled.ild")

        draw_svg(rescaled_polylines, open_in_browser=True)
        logger.info("Rendered SVG successfully")
    except Exception:
        logger.error("Failed to draw ILDA", exc_info=True)
        exit(1)


if __name__ == "__main__":
    main()
