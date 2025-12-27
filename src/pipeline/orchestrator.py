"""
Pipeline orchestrator for managing the complete bitmap-to-vector workflow.

Coordinates preprocessing, vectorization, and output generation for multiple
configuration combinations, saving intermediate results and final SVG files.
"""

import os

import cv2
import potrace

from src.preprocessing.preprocessing import (
    binary_img,
    mean_thresh_img,
    gaussian_thresh_img,
    otsu_thresholding,
)
from src.preprocessing.vectorization import vectorize_img, POTRACE_CONFIGS


def path_to_svg(path: potrace.Path, width: int, height: int) -> list[str]:
    """
    Convert a bitmap path to SVG format.

    Parameters:
        path (potrace.Path): The potrace path to convert.
        width (int): The width of the output SVG.
        height (int): The height of the output SVG.
    Returns:
        list[str]: The SVG representation as a list of strings.
    """
    parts = []

    # SVG Header
    parts.append(
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    )
    parts.append('<path d="')

    # Iterate over the curves (shapes) in the path
    for curve in path:
        # 1. Move to the start point of the shape
        start = curve.start_point
        parts.append(f"M {start.x},{start.y}")

        # 2. Iterate over segments in the curve
        for segment in curve:
            if segment.is_corner:
                # Corner segments are composed of two straight lines meeting at 'c'
                # Potrace Corner: Start -> c -> End
                c = segment.c
                end = segment.end_point
                parts.append(f"L {c.x},{c.y} L {end.x},{end.y}")
            else:
                # Bezier segments are cubic curves
                # Potrace Bezier: Start -> c1 -> c2 -> End
                c1 = segment.c1
                c2 = segment.c2
                end = segment.end_point
                parts.append(f"C {c1.x},{c1.y} {c2.x},{c2.y} {end.x},{end.y}")

        # 3. Close the shape loop
        parts.append("Z")

    # SVG Footer
    parts.append('" stroke="black" fill="none"/>')
    parts.append("</svg>")

    return parts


def save_img(workspace: str, filename: str, img: cv2.typing.MatLike) -> None:
    """
    Save the processed image to the specified workspace.

    File format is determined by the filename (jpg, pbm, pgm, ppm...)

    Parameters:
        workspace (str): Path to the workspace directory.
        filename (str): Name of the file to save the image as.
        img (cv2.typing.MatLike): The image to be saved.
    """
    if workspace.endswith("/"):
        workspace = workspace[:-1]
    cv2.imwrite(f"{workspace}/{filename}", img)


def run_pipeline(input: str):
    """
    Execute the complete image processing pipeline from bitmap to vector.

    Applies multiple preprocessing techniques (binary, mean threshold, Gaussian threshold,
    Otsu's thresholding) to the input image, then vectorizes each result using different
    Potrace configurations and saves the output as SVG files.

    Parameters:
        input (str): Path to the input image file (must be readable by OpenCV).

    Raises:
        FileNotFoundError: If the input image file does not exist or cannot be read.
    """
    data_dir = "smiley"
    pre_workspace = f"data/{data_dir}/preprocessing"
    svg_workspace = f"data/{data_dir}/svg"

    os.makedirs(pre_workspace, exist_ok=True)
    os.makedirs(svg_workspace, exist_ok=True)

    instructions = [
        ("binary_image", binary_img),
        ("mean_threshold_image", mean_thresh_img),
        ("gaussian_threshold_image", gaussian_thresh_img),
        ("otsu_threshold_image", otsu_thresholding),
        ("otsu_threshold_gaussian_blur_image", otsu_thresholding),
    ]

    img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {input}")

    for filename, func in instructions:
        print(f"Running preprocessing with {func.__name__}")

        if filename == "otsu_threshold_gaussian_blur_image":
            # Apply Gaussian blur (5x5 kernel, sigma=0) before Otsu thresholding
            tmp = cv2.GaussianBlur(img, (5, 5), 0)
            processed_img = func(tmp)
        else:
            processed_img = func(img)

        save_img(pre_workspace, f"{filename}.pbm", processed_img)

        for cfg_name, trace_cfg in POTRACE_CONFIGS.items():
            # TODO: Add timing and logging for performance metrics
            print(f"Vectorization using {cfg_name} mode")

            path = vectorize_img(processed_img, trace_cfg)

            print("Saving to svg")
            raw_svg = path_to_svg(path, img.shape[1], img.shape[0])
            with open(f"{svg_workspace}/{filename}_{cfg_name}.svg", "w") as svg_file:
                svg_file.writelines("\n".join(raw_svg))
                print(f"Saved SVG: {svg_workspace}/{filename.split('.')[0]}.svg")
