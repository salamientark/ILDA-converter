import cv2


def binary_img(input: str) -> cv2.Matlike | None:
    """
    Convert an image to black and white bitmap.

    This function is used in the bitmap to vector preprocess.

    Parameter:
        input (str) : Path to the input image file.


    Returns:
        cv2.Matlike | None : Black and white image as a cv2 Matlike object, or None if conversion fails.
    """
    img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary_img


def mean_tresh_img(input: str) -> cv2.Matlike | None:
    """
    Convert an image to black and white bitmap using mean thresholding.

    This function is used in the bitmap to vector preprocess.
    Parameter:
        input (str) : Path to the input image file.
    Returns:
        cv2.Matlike | None : Black and white image as a cv2 Matlike object, or None if conversion fails.
    """
    img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    mean_tresh_img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return mean_tresh_img


def gaussian_tresh_img(input: str) -> cv2.Matlike | None:
    """
    Convert an image to black and white bitmap using gaussian thresholding.

    This function is used in the bitmap to vector preprocess.
    Parameter:
        input (str) : Path to the input image file.
    Returns:
        cv2.Matlike | None : Black and white image as a cv2 Matlike object, or None if conversion fails.
    """
    img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    mean_tresh_img = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )
    return mean_tresh_img
