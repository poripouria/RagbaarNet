import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO", name: Optional[str] = None) -> logging.Logger:
    """Configure application logging once and return a module-specific logger.

    - Simple, fast console handler
    - Reasonable default format
    - Idempotent (safe to call multiple times)
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler(stream=sys.stdout)
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        datefmt = "%H:%M:%S"
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(handler)
        # Default level can be raised/lowered later per logger
        root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Return a namespaced logger for the caller
    return logging.getLogger(name or __name__)


def set_level(logger: logging.Logger, level: str) -> None:
    """Set the level on the provided logger and keep propagation to root."""
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
