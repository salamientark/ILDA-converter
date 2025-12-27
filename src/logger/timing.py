"""
Timing utilities for performance measurement.

Provides a simple context manager for timing operations and logging results.
"""

import time

from src.logger.logging_config import get_logger

logger = get_logger(__name__)


class Timer:
    """
    Context manager for timing code execution.

    Automatically logs execution duration when the context exits.
    Times are measured in milliseconds with 2 decimal precision.

    Example:
        with Timer("vectorization", config="default"):
            result = vectorize_img(img, config)

    Logs:
        Console: [2025-12-27 21:05:33] vectorization completed (2801.23ms)
        JSON: {"timestamp": "...", "level": "INFO", "message": "vectorization completed",
               "duration_ms": 2801.23, "operation": "vectorization",
               "metadata": {"config": "default"}}
    """

    def __init__(self, operation: str, **metadata):
        """
        Initialize timer for an operation.

        Parameters:
            operation (str): Name of the operation being timed.
            **metadata: Additional context (e.g., config="default").
        """
        self.operation = operation
        self.metadata = metadata
        self.start_time: float = 0.0

    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Stop timing and log results.

        Logs timing even if an exception occurred within the context.
        """
        duration_ms = (time.perf_counter() - self.start_time) * 1000

        # Log with structured data for JSON output
        logger.info(
            f"{self.operation} completed ({duration_ms:.2f}ms)",
            extra={
                "duration_ms": round(duration_ms, 2),
                "operation": self.operation,
                "metadata": self.metadata,
            },
        )

        # Don't suppress exceptions
        return False
