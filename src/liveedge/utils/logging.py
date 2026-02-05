"""Logging setup and utilities.

This module provides utilities for setting up logging with rich formatting.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.logging import RichHandler


def setup_logging(
    level: int | str = logging.INFO,
    log_file: str | Path | None = None,
    console: bool = True,
    rich_format: bool = True,
) -> logging.Logger:
    """Set up logging with optional file output and rich formatting.

    Args:
        level: Logging level (e.g., logging.INFO, "DEBUG").
        log_file: Optional path to log file.
        console: Whether to log to console.
        rich_format: Whether to use rich formatting for console output.

    Returns:
        Configured root logger.
    """
    # Convert string level to int
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    # Get root logger
    logger = logging.getLogger("liveedge")
    logger.setLevel(level)

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    if console:
        if rich_format:
            console_handler = RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=True,
                rich_tracebacks=True,
            )
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            console_handler = logging.StreamHandler(sys.stderr)
            console_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                )
            )
        console_handler.setLevel(level)
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Get a logger instance.

    Args:
        name: Logger name (defaults to "liveedge").

    Returns:
        Logger instance.
    """
    if name is None:
        return logging.getLogger("liveedge")
    return logging.getLogger(f"liveedge.{name}")


class LoggerAdapter(logging.LoggerAdapter):
    """Logger adapter that adds extra context to log messages."""

    def __init__(self, logger: logging.Logger, extra: dict[str, Any] | None = None):
        """Initialize the adapter.

        Args:
            logger: Base logger.
            extra: Extra context to add to all messages.
        """
        super().__init__(logger, extra or {})

    def process(self, msg: str, kwargs: Any) -> tuple[str, Any]:
        """Process the log message.

        Args:
            msg: Log message.
            kwargs: Keyword arguments.

        Returns:
            Processed message and kwargs.
        """
        if self.extra:
            prefix = " ".join(f"[{k}={v}]" for k, v in self.extra.items())
            msg = f"{prefix} {msg}"
        return msg, kwargs


def log_dict(logger: logging.Logger, data: dict[str, Any], level: int = logging.INFO) -> None:
    """Log a dictionary in a readable format.

    Args:
        logger: Logger to use.
        data: Dictionary to log.
        level: Logging level.
    """
    for key, value in data.items():
        logger.log(level, f"  {key}: {value}")


def log_metrics(
    logger: logging.Logger,
    metrics: dict[str, float],
    prefix: str = "",
    level: int = logging.INFO,
) -> None:
    """Log metrics in a formatted way.

    Args:
        logger: Logger to use.
        metrics: Dictionary of metric name to value.
        prefix: Prefix for the log message.
        level: Logging level.
    """
    msg = prefix
    for name, value in metrics.items():
        if isinstance(value, float):
            msg += f" | {name}: {value:.4f}"
        else:
            msg += f" | {name}: {value}"
    logger.log(level, msg.strip(" |"))
