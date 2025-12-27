"""
Image preprocessing functions for converting grayscale images to binary.

Provides various thresholding techniques including fixed threshold, adaptive
mean threshold, adaptive Gaussian threshold, and Otsu's automatic thresholding.
"""

import cv2


def binary_img(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """
    Convert a grayscale image to binary using fixed thresholding.

    Applies a fixed threshold value of 127 to convert the image to black and white.
    Pixels above the threshold become white (255), pixels below become black (0).

    Parameters:
        img (cv2.typing.MatLike): Grayscale input image.

    Returns:
        cv2.typing.MatLike: Binary image (black and white only).
    """
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary_img


def mean_thresh_img(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """
    Convert a grayscale image to binary using adaptive mean thresholding.

    Uses adaptive thresholding where the threshold value is the mean of the
    neighborhood area (11x11 block) minus a constant (2).

    Parameters:
        img (cv2.typing.MatLike): Grayscale input image.

    Returns:
        cv2.typing.MatLike: Binary image with adaptive thresholding applied.
    """
    mean_thresh_img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return mean_thresh_img


def gaussian_thresh_img(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """
    Convert a grayscale image to binary using adaptive Gaussian thresholding.

    Uses adaptive thresholding where the threshold value is a Gaussian-weighted sum
    of the neighborhood area (11x11 block) minus a constant (2).

    Parameters:
        img (cv2.typing.MatLike): Grayscale input image.

    Returns:
        cv2.typing.MatLike: Binary image with Gaussian adaptive thresholding applied.
    """
    gaussian_thresh_img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return gaussian_thresh_img


def otsu_thresholding(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """
    Convert a grayscale image to binary using Otsu's automatic thresholding.

    Uses Otsu's method to automatically determine the optimal threshold value
    by minimizing intra-class variance between foreground and background pixels.

    Parameters:
        img (cv2.typing.MatLike): Grayscale input image.

    Returns:
        cv2.typing.MatLike: Binary image with Otsu's thresholding applied.
    """
    _, otsu_thresh_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return otsu_thresh_img
