"""
Pipeline stages for modular image processing operations.

Defines individual processing stages that can be composed into larger workflows,
currently including the preprocessing stage for bitmap thresholding.
"""

import cv2

from src.preprocessing.preprocessing import (
    binary_img,
    mean_tresh_img,
    gaussian_tresh_img,
    otsu_thresholding,
)


def run_preprocessing_stage(input: str) -> list[cv2.typing.MatLike]:
    """
    Run the preprocessing stage on a bitmap image file.

    Applies multiple thresholding techniques to convert the grayscale input
    into binary images suitable for vectorization.

    Parameters:
        input (str): Path to the input image file.

    Returns:
        list[cv2.typing.MatLike]: List of processed binary images, one for each
            preprocessing technique (binary, mean threshold, Gaussian threshold,
            Otsu's threshold, and Otsu's with Gaussian blur).

    Raises:
        FileNotFoundError: If the input image file does not exist or cannot be read.
    """
    img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {input}")

    image_results = []
    image_results.append(binary_img(img))
    image_results.append(mean_tresh_img(img))
    image_results.append(gaussian_tresh_img(img))
    image_results.append(otsu_thresholding(img))

    gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)
    image_results.append(otsu_thresholding(gaussian_blur))

    return image_results
