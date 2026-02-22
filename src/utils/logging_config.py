"""
OR-Symphony: Structured Logging Configuration

Sets up JSON-structured logging with file and console handlers.
Immutable log files for audit compliance.

Usage:
    from src.utils.logging_config import setup_logging
    setup_logging()
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

from src.utils.constants import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL, LOGS_DIR


def setup_logging(
    level: str = LOG_LEVEL,
    log_dir: Path | None = None,
    enable_file_logging: bool = True,
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_dir: Directory for log files. Defaults to project logs/ directory.
        enable_file_logging: Whether to write logs to files.
    """
    if log_dir is None:
        log_dir = LOGS_DIR

    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    if enable_file_logging:
        # General application log
        app_log_path = log_dir / f"app_{datetime.now().strftime('%Y%m%d')}.log"
        file_handler = logging.FileHandler(app_log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        root_logger.addHandler(file_handler)

    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

    root_logger.info("Logging initialized — level=%s, file_logging=%s", level, enable_file_logging)


def get_transcript_logger() -> logging.Logger:
    """
    Get a dedicated logger for immutable transcript logs.

    Returns:
        Logger that writes to logs/transcripts/YYYYMMDD.log (append-only).
    """
    transcript_dir = LOGS_DIR / "transcripts"
    transcript_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("or_symphony.transcripts")
    if not logger.handlers:
        log_path = transcript_dir / f"{datetime.now().strftime('%Y%m%d')}.log"
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


def get_state_change_logger() -> logging.Logger:
    """
    Get a dedicated logger for state change audit logs.

    Returns:
        Logger that writes to logs/state_changes.log (append-only).
    """
    logger = logging.getLogger("or_symphony.state_changes")
    if not logger.handlers:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        log_path = LOGS_DIR / "state_changes.log"
        handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt=LOG_DATE_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    return logger


if __name__ == "__main__":
    setup_logging(level="DEBUG", enable_file_logging=False)
    logger = logging.getLogger("test")
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    print("Logging setup complete.")
