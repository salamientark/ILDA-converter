
"""
Vectorization module for converting binary images to vector paths using Potrace.

Provides predefined Potrace configurations optimized for different use cases
(default, high quality, smooth, fast) and a wrapper function for vectorization.
"""

from typing import Any
from abc import ABC, abstractmethod


class VectorizationEngine(ABC):
    """
    Abstract base class for vectorization engines.
    Defines the interface for vectorization engines that convert binary images
    into vector paths.
    """
    @abstractmethod
    def vectorize(self, img: Any, config: dict[str, Any] | None) -> Any:
        """
        Convert a binary image to a vector path.
        Parameters:
            img (Any): Binary input image (black and white).
            config (dict[str, Any] | None): Configuration dictionary for the
                vectorization engine.
        Returns:
            Any: Vector path representation of the input image.
        """
        pass

    @abstractmethod
    def convert_to_svg(self, vector_path: Any, width: int, height: int) -> str:
        """
        Convert the vector path to an SVG format.
        Parameters:
            vector_path (Any): The vector path representation.
            width (int): Width of the original image.
            height (int): Height of the original image.
        Returns:
            str: SVG representation of the vector path.
        """
