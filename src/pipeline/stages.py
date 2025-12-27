import cv2
import potrace

from src.preprocessing.preprocessing import binary_img, mean_tresh_img, gaussian_tresh_img, otsu_thresholding


def run_preprocessing_stage(input: str) -> list[cv2.typing.MatLike]:
    """
    Run the preprocessing stage on bitmap file.

    Parameters:
        input (str): Path to the input image file.
    """
    # Read raw image
    img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)

    # Create treated image buffer
    image_results = []
    image_results.append(binary_img(img))
    image_results.append(mean_tresh_img(img))
    image_results.append(gaussian_tresh_img(img))
    image_results.append(otsu_thresholding(img))

    gaussian_blur = cv2.GaussianBlur(img, (5, 5), 0)
    image_results.append(otsu_thresholding(gaussian_blur))

    return image_results

def run_vectorization_stage(images: list[cv2.typing.MatLike]) -> list[potrace.Path]:
    """
    Run the vectorization stage.
    """
    for img in images:

