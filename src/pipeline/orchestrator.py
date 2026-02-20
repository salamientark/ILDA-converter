"""Pipeline orchestrator for managing the complete bitmap-to-vector workflow.

Coordinates preprocessing, vectorization, and output generation for multiple
configuration combinations, saving intermediate results and final SVG/ILDA files.

This module primarily uses Potrace for vectorization, but also includes an
optional OpenCV contour approximation preview (useful for tuning epsilon).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import potrace

from src.ilda.ilda_3d import laser_path_to_ilda, polylines_to_ilda
from src.logger.logging_config import get_logger
from src.logger.timing import Timer
from src.preprocessing.preprocessing import (
    binary_img,
    gaussian_thresh_img,
    mean_thresh_img,
    otsu_thresholding,
)
from src.postprocessing.find_eulerian_path import find_eulerian_path

from src.postprocessing.corner_dwell import apply_corner_dwell
from src.postprocessing.resample_path import resample_path
from src.postprocessing.weld_vertices import weld_vertices
from src.vectorization import POTRACE_CONFIGS, vectorize_opencv

logger = get_logger(__name__)


def polyline_to_svg(
    polylines: list[list[tuple[float, float]]], width: int, height: int
) -> list[str]:
    """Convert a list of polylines to a minimal SVG.

    Parameters:
        polylines (list[list[tuple[float, float]]]): List of polylines. Each polyline
            is a list of `(x, y)` points.
        width (int): Output SVG width.
        height (int): Output SVG height.

    Returns:
        list[str]: SVG lines suitable for writing with `"\n".join(...)`.
    """
    parts: list[str] = []

    parts.append(
        f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">'
    )
    parts.append('<path d="')

    for line in polylines:
        start_x, start_y = line[0]
        parts.append(f"M {start_x},{start_y}")

        for x, y in line[1:]:
            parts.append(f"L {x},{y}")

        parts.append("Z")

    parts.append('" stroke="black" fill="none"/>')
    parts.append("</svg>")

    return parts


def create_instructions(
    preprocessing: str, vectorization: str
) -> tuple[list[tuple[str, Callable]], list[tuple[str, dict[str, Any]]]]:
    """Filter preprocessing and vectorization instructions based on user selections.

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
    """Convert a potrace path to SVG format.

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

    for curve in path:
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


def save_img(workspace: str, filename: str, img: cv2.typing.MatLike) -> None:
    """Save an image to the specified workspace.

    File format is determined by the filename extension.

    Parameters:
        workspace (str): Output directory.
        filename (str): Output filename.
        img (cv2.typing.MatLike): Image to save.
    """
    if workspace.endswith("/"):
        workspace = workspace[:-1]

    cv2.imwrite(f"{workspace}/{filename}", img)
    logger.debug(f"Image saved: {workspace}/{filename}")


def run_pipeline(input: str, preprocessing: str, vectorization: str) -> None:
    """Execute the complete image processing pipeline from bitmap to vector.

    Parameters:
        input (str): Path to the input image file.
        preprocessing (str): Preprocessing method selection.
        vectorization (str): Potrace vectorization config selection.

    Raises:
        FileNotFoundError: If the input image file does not exist or cannot be read.
    """
    base_filename = os.path.splitext(os.path.basename(input))[0]
    pre_workspace = f"data/{base_filename}/preprocessing"
    svg_workspace = f"data/{base_filename}/svg"
    ilda_workspace = f"data/{base_filename}/ilda"
    optimizer_workspace = f"data/{base_filename}/optimizer"

    os.makedirs(f"data/{base_filename}", exist_ok=True)
    os.makedirs(pre_workspace, exist_ok=True)
    os.makedirs(svg_workspace, exist_ok=True)
    os.makedirs(ilda_workspace, exist_ok=True)
    os.makedirs(optimizer_workspace, exist_ok=True)

    preproc_instructions, vectorization_instructions = create_instructions(
        preprocessing,
        vectorization,
    )

    img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Input image not found: {input}")

    logger.info(f"Image loaded: {img.shape[1]}x{img.shape[0]} pixels")

    for pre_type, func in preproc_instructions:
        logger.info(f"Running preprocessing with {func.__name__}")

        filename = f"{pre_type}_{base_filename}"
        processed_img = func(img)

        # Save binary preprocessing output for inspection.
        save_img(pre_workspace, f"{filename}.pbm", processed_img)

        for cfg_name, trace_cfg in vectorization_instructions:
            logger.info(f"Vectorization using {cfg_name} mode")

            # with Timer("vectorization", config=cfg_name):
            #     path = vectorize_potrace(processed_img, trace_cfg)

            with Timer("vectorization", config=cfg_name):
                polyline, _ = vectorize_opencv(
                    processed_img, epsilon_ratio=0.0001, invert=True
                )

            logger.debug("Converting path to SVG")
            raw_svg = polyline_to_svg(polyline, img.shape[1], img.shape[0])
            with open(f"{svg_workspace}/{filename}_{cfg_name}.svg", "w") as svg_file:
                svg_file.writelines("\n".join(raw_svg))
                logger.info(f"Saved SVG: {svg_workspace}/{filename}_{cfg_name}.svg")

            logger.debug("Converting path to ILDA")
            raw_ilda, _, _, _ = polylines_to_ilda(polyline)
            with open(f"{ilda_workspace}/{filename}_{cfg_name}.ild", "wb") as ilda_file:
                ilda_file.write(b"".join(raw_ilda))
                logger.info(f"Saved ILDA: {ilda_workspace}/{filename}_{cfg_name}.ild")

            # Stage 1: Raw point count
            raw_count = sum(len(pl) for pl in polyline)
            logger.info(
                "Stage raw: %d points across %d polylines", raw_count, len(polyline)
            )

            # Stage 2: Weld
            welded = weld_vertices(polyline)
            weld_count = sum(len(pl) for pl in welded)
            logger.info(
                "Stage weld: %d points across %d polylines", weld_count, len(welded)
            )
            raw_ilda_welded, _, _, _ = polylines_to_ilda(welded)
            with open(
                f"{optimizer_workspace}/{filename}_{cfg_name}_weld.ild", "wb"
            ) as f:
                f.write(b"".join(raw_ilda_welded))
            logger.info(
                "Saved weld ILDA: %s/%s_%s_weld.ild",
                optimizer_workspace,
                filename,
                cfg_name,
            )

            # Stage 3: Optimized (Eulerian path)
            opt_path = find_eulerian_path(welded, r=255, g=255, b=255)
            if not opt_path:
                logger.info("Stage optimized: empty path; skipping ILDA output")
                continue
            visible = sum(1 for pt in opt_path if not pt.is_blanking)
            blanking = sum(1 for pt in opt_path if pt.is_blanking)
            logger.info(
                "Stage optimized: %d total points (%d visible, %d blanking)",
                len(opt_path),
                visible,
                blanking,
            )
            raw_ilda_opt, _, _, _ = laser_path_to_ilda(opt_path)
            with open(
                f"{optimizer_workspace}/{filename}_{cfg_name}_optimized.ild", "wb"
            ) as f:
                f.write(b"".join(raw_ilda_opt))
            logger.info(
                "Saved optimized ILDA: %s/%s_%s_optimized.ild",
                optimizer_workspace,
                filename,
                cfg_name,
            )

            # Stage 4: Resampled
            resampled = resample_path(opt_path)
            visible_r = sum(1 for pt in resampled if not pt.is_blanking)
            blanking_r = sum(1 for pt in resampled if pt.is_blanking)
            logger.info(
                "Stage resampled: %d total points (%d visible, %d blanking)",
                len(resampled),
                visible_r,
                blanking_r,
            )
            raw_ilda_resampled, _, _, _ = laser_path_to_ilda(resampled)
            output_path = (
                Path(optimizer_workspace) / f"{filename}_{cfg_name}_resampled.ild"
            )
            output_path.write_bytes(b"".join(raw_ilda_resampled))
            logger.info("Saved resampled ILDA: %s", output_path)

            # Stage 5: Corner dwell
            dwell_path = apply_corner_dwell(resampled, max_dwell=8)
            visible_d = sum(1 for pt in dwell_path if not pt.is_blanking)
            blanking_d = sum(1 for pt in dwell_path if pt.is_blanking)
            logger.info(
                "Stage corner_dwell: %d total points (%d visible, %d blanking)",
                len(dwell_path),
                visible_d,
                blanking_d,
            )
            raw_ilda_dwell, _, _, _ = laser_path_to_ilda(dwell_path)
            dwell_out = (
                Path(optimizer_workspace) / f"{filename}_{cfg_name}_corner_dwell.ild"
            )
            dwell_out.write_bytes(b"".join(raw_ilda_dwell))
            logger.info("Saved corner_dwell ILDA: %s", dwell_out)
