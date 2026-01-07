from typing import Any
from src.logger.logging_config import get_logger

from .base import VectorizationEngine

import potrace as potrace

logger = get_logger(__name__)

POTRACE_CONFIGS = {
    "default": {
        "turdsize": 2,
        "turnpolicy": potrace.POTRACE_TURNPOLICY_MINORITY,
        "alphamax": 1.0,
        "opticurve": True,
        "opttolerance": 0.2,
    },
    "high_quality": {
        "turdsize": 0,  # No despeckle
        "turnpolicy": potrace.POTRACE_TURNPOLICY_MINORITY,
        "alphamax": 0.5,  # More corners
        "opticurve": True,
        "opttolerance": 0.1,  # Tighter curves
    },
    "smooth": {
        "turdsize": 10,  # Remove small speckles
        "turnpolicy": potrace.POTRACE_TURNPOLICY_MINORITY,
        "alphamax": 1.3333,  # Fewer corners
        "opticurve": True,
        "opttolerance": 0.5,  # Looser curves
    },
    "fast": {
        "turdsize": 2,
        "turnpolicy": potrace.POTRACE_TURNPOLICY_MINORITY,
        "alphamax": 1.0,
        "opticurve": False,  # Skip optimization
        "opttolerance": 0.2,
    },
}

class PotraceEngine(VectorizationEngine):
    """
    Class for potrace vectorization engines.
    """
    @classmethod
    def vectorize(self, img: Any, config: dict[str, Any] | None) -> Any:
        """
        Convert a binary image to a vector path using Potrace.

        Wraps the Potrace bitmap tracing functionality with optional configuration
        for controlling corner detection, curve optimization, and despeckling.

        Parameters:
            img (cv2.typing.MatLike): Binary input image (black and white).
            config (dict[str, Any] | None): Potrace configuration dictionary with
                parameters like turdsize, turnpolicy, alphamax, etc. Use predefined
                configs from POTRACE_CONFIGS or None for defaults.

        Returns:
            potrace.Path: Vectorized path representation of the binary image.
        """
        bitmap = potrace.Bitmap(img)
        if config is None:
            path = bitmap.trace()
        else:
            path = bitmap.trace(**config)

        logger.debug("Image vectorized")
        return path

    @classmethod
    def convert_to_svg(self, vector_path: Any, width: int, height: int) -> str:
        """
        Convert a potrace path to SVG format.

        Parameters:
            path (potrace.Path): The potrace path to convert.
            width (int): Output SVG width.
            height (int): Output SVG height.

        Returns:
            list[str]: SVG lines.
        """
        parts: list[str] = []

        parts.append(
            f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
        )
        parts.append('<path d="')

        for curve in vector_path:
            start = curve.start_point
            parts.append(f"M {start.x},{start.y}")

            for segment in curve:
                if segment.is_corner:
                    c = segment.c
                    end = segment.end_point
                    parts.append(f"L {c.x},{c.y} L {end.x},{end.y}")
                else:
                    c1 = segment.c1
                    c2 = segment.c2
                    end = segment.end_point
                    parts.append(f"C {c1.x},{c1.y} {c2.x},{c2.y} {end.x},{end.y}")

            parts.append("Z")

        parts.append('" stroke="black" fill="none"/>')
        parts.append("</svg>")

        return parts

    @classmethod
    def convert_to_ilda(self, vector_path: Any) -> str:
        """
        Convert a potrace path to ILDA

        Parameters:
            vector_path (potrace.Path): The potrace path to convert.
        Returns:
            str: ILDA formatted string.
        """
        pass

    @staticmethod
    def path_to_polyline(path: potrace.Path) -> list[list[tuple[float, float]]]:
        """
        Convert a potrace path to a list of polylines.
        Parameters:
            path (potrace.Path): The potrace path to convert.
        Returns:
            list[list[tuple[float, float]]]: List of polylines as lists of (x, y) points.
        """
        polylines: list[list[tuple[float, float]]] = []
        for curve in path:
            points: list[tuple[float, float]] = []
            start = curve.start_point
            points.append((float(start.x), float(start.y)))
            for segment in curve:
                if segment.is_corner:
                    c = segment.c
                    end = segment.end_point
                    points.append((float(c.x), float(c.y)))
                    points.append((float(end.x), float(end.y)))
                else:
                    # For Bezier curves, sample points along the curve
                    num_samples = 10
                    for t in range(1, num_samples + 1):
                        t /= num_samples
                        x = (
                            (1 - t) ** 3 * start.x
                            + 3 * (1 - t) ** 2 * t * segment.c1.x
                            + 3 * (1 - t) * t ** 2 * segment.c2.x
                            + t ** 3 * segment.end_point.x
                        )
                        y = (
                            (1 - t) ** 3 * start.y
                            + 3 * (1 - t) ** 2 * t * segment.c1.y
                            + 3 * (1 - t) * t ** 2 * segment.c2.y
                            + t ** 3 * segment.end_point.y
                        )
                        points.append((float(x), float(y)))
            polylines.append(points)
        return polylines
