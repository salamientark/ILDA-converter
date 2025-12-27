"""
Logging configuration for ILDA project.

Provides structured JSON logging to files and human-readable console output.
"""

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path


class StructuredFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging.

    Converts log records to JSON format with timestamp, level, module,
    function, line number, message, and optional exception information.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record as a JSON string.

        Parameters:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: JSON-formatted log entry.
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            # "module": record.module,
            # "function": record.funcName,
            # "line": record.lineno,
            "message": record.getMessage(),
        }

        # Add timing information if present
        duration_ms = getattr(record, "duration_ms", None)
        if duration_ms is not None:
            log_data["duration_ms"] = duration_ms

        operation = getattr(record, "operation", None)
        if operation is not None:
            log_data["operation"] = operation

        metadata = getattr(record, "metadata", None)
        if metadata is not None:
            log_data["metadata"] = metadata

        # Add exception information if present
        if record.exc_info:
            log_data["exception"] = "".join(
                traceback.format_exception(*record.exc_info)
            )

        return json.dumps(log_data)


def setup_logging() -> None:
    """
    Initialize logging configuration for ILDA.

    Sets up two handlers:
    1. Console handler: Human-readable format with timestamp [YYYY-MM-DD HH:MM:SS]
    2. File handler: Structured JSON format saved to logs/ilda_YYYYMMDD_HHMMSS.log

    Creates logs/ directory if it doesn't exist.
    """
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = logs_dir / f"ilda_{timestamp}.log"

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture everything at root level

    # Remove any existing handlers
    root_logger.handlers.clear()

    # Console handler: Human-readable format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    console_handler.setFormatter(console_format)
    root_logger.addHandler(console_handler)

    # File handler: Structured JSON format
    file_handler = logging.FileHandler(log_filename, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(StructuredFormatter())
    root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance for a module.

    Parameters:
        name (str): The name of the module (typically __name__).

    Returns:
        logging.Logger: Configured logger instance.
    """
    return logging.getLogger(name)
