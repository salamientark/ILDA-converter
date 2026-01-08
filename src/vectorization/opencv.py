"""OpenCV-based vectorization utilities.

This module extracts contours from a binary image using OpenCV and approximates
those contours into simplified polygons via ``cv2.approxPolyDP``.

Two representations are useful depending on your next step:

- **OpenCV contours** (``list[np.ndarray]`` with shape ``(N, 1, 2)``):
  compatible with ``cv2.drawContours`` and therefore easy to visualize with
  ``cv2.imshow``.
- **Polylines** (``list[list[tuple[float, float]]]``):
  easy to consume for SVG/ILDA generation code.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

# Common retrieval modes:
# - cv2.RETR_EXTERNAL: outer contours only
# - cv2.RETR_LIST: all contours, no hierarchy
# - cv2.RETR_TREE: all contours with hierarchy
OPENCV2_CONFIGS: dict[str, dict[str, Any]] = {
    "default": {
        "retrieval_mode": cv2.RETR_LIST,
        "epsilon_ratio": 0.01,
        "invert": False,
    }
}


def _as_binary_uint8(img: cv2.typing.MatLike, *, invert: bool) -> np.ndarray:
    """Normalize an image to a 0/255 uint8 binary image.

    Parameters:
        img (cv2.typing.MatLike): Input image, expected to be single-channel.
        invert (bool): Whether to invert the image (swap foreground/background).

    Returns:
        np.ndarray: Binary uint8 image with values in {0, 255}.
    """
    binary = np.asarray(img)

    # Ensure uint8; OpenCV contour ops expect 8-bit single channel.
    if binary.dtype != np.uint8:
        binary = binary.astype(np.uint8)

    # Robust conversion to 0/255 regardless of whether the input is {0,1},
    # {0,255}, or some other "binary-ish" values.
    binary = np.where(binary > 0, 255, 0).astype(np.uint8)

    if invert:
        binary = cv2.bitwise_not(binary)

    # OpenCV can mutate the input; keep it contiguous and owned.
    return np.ascontiguousarray(binary)


def find_image_contours(
    processed_img: cv2.typing.MatLike,
    *,
    retrieval_mode: int = cv2.RETR_LIST,
    invert: bool = False,
) -> tuple[list[np.ndarray], np.ndarray | None]:
    """Extract contours from a binary image.

    Notes:
        ``cv2.findContours`` expects white foreground (255) on black background (0).

    Parameters:
        processed_img (cv2.typing.MatLike): Binary image.
        retrieval_mode (int): OpenCV contour retrieval mode.
        invert (bool): Invert image before contour extraction.

    Returns:
        tuple[list[np.ndarray], np.ndarray | None]: (contours, hierarchy).
    """
    binary = _as_binary_uint8(processed_img, invert=invert)
    contours, hierarchy = cv2.findContours(
        binary.copy(),
        retrieval_mode,
        cv2.CHAIN_APPROX_NONE,
    )
    return contours, hierarchy


def approximate_contours(
    contours: list[np.ndarray],
    *,
    epsilon_ratio: float = 0.01,
    closed: bool = True,
    min_points: int = 3,
) -> list[np.ndarray]:
    """Simplify contours using Douglas–Peucker polygon approximation.

    Parameters:
        contours (list[np.ndarray]): Output of ``cv2.findContours``.
        epsilon_ratio (float): Epsilon as ratio of perimeter. Larger simplifies more.
        closed (bool): Whether contours are closed shapes.
        min_points (int): Skip contours with fewer points than this.

    Returns:
        list[np.ndarray]: Approximated contours, each shaped ``(N, 1, 2)``.
    """
    approximated: list[np.ndarray] = []

    for contour in contours:
        if contour is None or len(contour) < min_points:
            continue

        perimeter = cv2.arcLength(contour, closed)
        if perimeter <= 0:
            continue

        epsilon = float(epsilon_ratio) * float(perimeter)
        approx = cv2.approxPolyDP(contour, epsilon, closed)

        if approx is None or len(approx) < min_points:
            continue

        approximated.append(approx)

    return approximated


def contours_to_polylines(
    contours: list[np.ndarray],
    *,
    close_loop: bool = True,
) -> list[list[tuple[float, float]]]:
    """Convert OpenCV contours to a pure-Python polyline representation.

    Parameters:
        contours (list[np.ndarray]): Contours shaped ``(N, 1, 2)``.
        close_loop (bool): Append the first point to the end if not already closed.

    Returns:
        list[list[tuple[float, float]]]: Polylines as lists of (x, y) points.
    """
    polylines: list[list[tuple[float, float]]] = []

    for contour in contours:
        if contour is None or len(contour) == 0:
            continue

        pts = contour.reshape(-1, 2)
        points = [(float(x), float(y)) for x, y in pts]

        if close_loop and points and points[0] != points[-1]:
            points.append(points[0])

        polylines.append(points)

    return polylines


def draw_contours_for_debug(
    processed_img: cv2.typing.MatLike,
    contours: list[np.ndarray],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """Render contours on top of the image for visualization.

    This returns a BGR image you can directly show with ``cv2.imshow``.

    Parameters:
        processed_img (cv2.typing.MatLike): Input image (usually binary grayscale).
        contours (list[np.ndarray]): Contours to draw.
        color (tuple[int, int, int]): BGR color.
        thickness (int): Line thickness.

    Returns:
        np.ndarray: BGR image with contours drawn.
    """
    base = np.asarray(processed_img)

    if base.ndim == 2:
        canvas = cv2.cvtColor(base.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    else:
        canvas = base.astype(np.uint8).copy()

    cv2.drawContours(canvas, contours, contourIdx=-1, color=color, thickness=thickness)
    return canvas


def vectorize_opencv(
    processed_img: cv2.typing.MatLike,
    *,
    epsilon_ratio: float = 0.01,
    retrieval_mode: int = cv2.RETR_LIST,
    invert: bool = False,
) -> tuple[list[list[tuple[float, float]]], list[np.ndarray]]:
    """Vectorize an image via OpenCV contours + polygon approximation.

    Parameters:
        processed_img (cv2.typing.MatLike): Binary image.
        epsilon_ratio (float): Epsilon as ratio of contour perimeter.
        retrieval_mode (int): OpenCV retrieval mode.
        invert (bool): Invert image before contour extraction.

    Returns:
        tuple[list[list[tuple[float, float]]], list[np.ndarray]]:
            (polylines, approx_contours)

            - ``polylines`` is convenient for SVG/ILDA conversion.
            - ``approx_contours`` is convenient for OpenCV visualization
              (``cv2.drawContours`` + ``cv2.imshow``).
    """
    contours, _hierarchy = find_image_contours(
        processed_img,
        retrieval_mode=retrieval_mode,
        invert=invert,
    )

    approx_contours = approximate_contours(contours, epsilon_ratio=epsilon_ratio)
    polylines = contours_to_polylines(approx_contours)

    return polylines, approx_contours
