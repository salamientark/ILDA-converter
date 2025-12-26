import cv2


def binary_img(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """
    Convert an image to black and white bitmap.

    This function is used in the bitmap to vector preprocess.

    Parameter:
        img (cv2.typing.MatLike): Grayscale image as a cv2 MatLike object.

    Returns:
        cv2.typing.MatLike: Black and white image as a cv2 MatLike object.
    """
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary_img


def mean_tresh_img(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """
    Convert an image to black and white bitmap using mean thresholding.

    This function is used in the bitmap to vector preprocess.

    Parameter:
        img (cv2.typing.MatLike): Grayscale image as a cv2 MatLike object.

    Returns:
        cv2.typing.MatLike: Black and white image as a cv2 MatLike object.
    """
    mean_tresh_img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return mean_tresh_img


def gaussian_tresh_img(img: cv2.typing.MatLike) -> cv2.typing.MatLike:
    """
    Convert an image to black and white bitmap using gaussian thresholding.

    This function is used in the bitmap to vector preprocess.

    Parameter:
        img (cv2.typing.MatLike): Grayscale image as a cv2 MatLike object.

    Returns:
        cv2.typing.MatLike: Black and white image as a cv2 MatLike object.
    """
    gaussian_tresh_img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return gaussian_tresh_img
