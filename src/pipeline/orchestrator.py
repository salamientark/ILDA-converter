"""
Pipeline orchestrator for managing the complete bitmap-to-vector workflow.

Coordinates preprocessing, vectorization, and output generation for multiple
configuration combinations, saving intermediate results and final SVG files.
"""
import numpy as np

import os
from collections.abc import Callable
from typing import Any

import cv2
import potrace

from src.preprocessing.preprocessing import (
    binary_img,
    mean_thresh_img,
    gaussian_thresh_img,
    otsu_thresholding,
)
from src.vectorization.potrace import POTRACE_CONFIGS, vectorize_potrace
from src.vectorization import POTRACE_CONFIGS, vectorize_potrace, find_image_contour, vectorize_img_opencv
from src.logger.logging_config import get_logger
from src.logger.timing import Timer
from src.ilda.ilda_3d import path_to_ilda_3d

logger = get_logger(__name__)


def create_instructions(
    preprocessing: str, vectorization: str
) -> tuple[list[tuple[str, Callable]], list[tuple[str, dict[str, Any]]]]:
    """
    Filter preprocessing and vectorization instructions based on user selections.

    Parameters:
        preprocessing (str): Preprocessing method to use. Options: 'binary',
            'mean', 'gaussian', 'otsu', or 'all' to apply all methods.
        vectorization (str): Vectorization configuration to use. Options:
            'default', 'fast', 'high', 'smooth', or 'all' to apply all configurations.

    Returns:
        tuple[list[tuple[str, Callable]], list[tuple[str, dict[str, Any]]]]: A tuple containing:
            - List of tuples with (preprocessing_name, preprocessing_function)
            - List of tuples with (config_name, config_dict) from POTRACE_CONFIGS
    """
    preprocessing_map = [
        ("binary", binary_img),
        ("mean", mean_thresh_img),
        ("gaussian", gaussian_thresh_img),
        ("otsu", otsu_thresholding),
    ]

    # Map user-friendly names to POTRACE_CONFIGS keys
    vectorization_name_map = {
        "default": "default",
        "fast": "fast",
        "high": "high_quality",
        "smooth": "smooth",
    }

    preprocessing_instructions = [
        pre for pre in preprocessing_map if preprocessing in (pre[0], "all")
    ]

    if vectorization == "all":
        vectorization_instructions = list(POTRACE_CONFIGS.items())
    else:
        config_key = vectorization_name_map.get(vectorization, vectorization)
        vectorization_instructions = [(config_key, POTRACE_CONFIGS[config_key])]

    return preprocessing_instructions, vectorization_instructions


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
    logger.debug(f"Image saved: {workspace}/{filename}")


def run_pipeline(input: str, preprocessing: str, vectorization: str):
    """
    Execute the complete image processing pipeline from bitmap to vector.

    Applies selected preprocessing technique(s) to the input image, then vectorizes
    each result using specified Potrace configuration(s) and saves the output as SVG files.

    Parameters:
        input (str): Path to the input image file (must be readable by OpenCV).
        preprocessing (str): Preprocessing method to use. Options: 'binary',
            'mean', 'gaussian', 'otsu', or 'all' to apply all methods.
        vectorization (str): Vectorization configuration to use. Options:
            'default', 'fast', 'high', 'smooth', or 'all' to apply all configurations.

    Raises:
        FileNotFoundError: If the input image file does not exist or cannot be read.
    """
    base_filename = os.path.splitext(os.path.basename(input))[0]
    pre_workspace = f"data/{base_filename}/preprocessing"
    svg_workspace = f"data/{base_filename}/svg"
    ilda_workspace = f"data/{base_filename}/ilda"
    os.makedirs(f"data/{base_filename}", exist_ok=True)
    os.makedirs(pre_workspace, exist_ok=True)
    os.makedirs(svg_workspace, exist_ok=True)
    os.makedirs(ilda_workspace, exist_ok=True)

    preproc_instructions, vectorization_instructions = create_instructions(
        preprocessing, vectorization
    )

    img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {input}")

    logger.info(f"Image loaded: {img.shape[1]}x{img.shape[0]} pixels")

    for pre_type, func in preproc_instructions:
        logger.info(f"Running preprocessing with {func.__name__}")

        # Apply preprocessing
        filename = f"{pre_type}_{base_filename}"
        processed_img = func(img)

        c = find_image_contour(processed_img, retrieval_mode=cv2.RETR_EXTERNAL, invert=False)

        for eps in np.linspace(0.001, 0.05, 10):
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, eps * peri, True)
            # draw the approximated contour on the image
            output = processed_img.copy()
            cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
            text = "eps={:.4f}, num_pts={}".format(eps, len(approx))
            # show the approximated contour image
            print("[INFO] {}".format(text))
            cv2.imshow("Approximated Contour", output)
            cv2.waitKey(0)

        save_img(pre_workspace, f"{filename}_opencv2_contour.pbm", approx)
        return

        # save_img(pre_workspace, f"{filename}.pbm", processed_img)

        for cfg_name, trace_cfg in vectorization_instructions:
            logger.info(f"Vectorization using {cfg_name} mode")

            # Vectorize image
            with Timer("vectorization", config=cfg_name):
                path = vectorize_potrace(processed_img, trace_cfg)

            # Save as SVG
            logger.debug("Converting path to SVG")
            raw_svg = path_to_svg(path, img.shape[1], img.shape[0])
            with open(f"{svg_workspace}/{filename}_{cfg_name}.svg", "w") as svg_file:
                svg_file.writelines("\n".join(raw_svg))
                logger.info(f"Saved SVG: {svg_workspace}/{filename}_{cfg_name}.svg")

            # Save as ILDA
            logger.debug("Converting path to ILDA")
            raw_ilda = path_to_ilda_3d(path)
            with open(f"{ilda_workspace}/{filename}_{cfg_name}.ild", "wb") as ilda_file:
                for chunk in raw_ilda:
                    ilda_file.write(chunk)
                logger.info(f"Saved ILDA: {ilda_workspace}/{filename}_{cfg_name}.ild")
