import logging
import os
import sys
from typing import Optional

LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "platform_logs.log")


def setup_logging(level: str = "INFO", name: Optional[str] = None) -> logging.Logger:
    """Configure application logging once and return a module-specific logger.

    - Simple, fast console handler
    - Reasonable default format
    - Idempotent (safe to call multiple times)

    Outputs:
    - Console (stdout)
    - File (platform_logs/platform.log)
    """
    root = logging.getLogger()
    if not root.handlers:
        # Ensure log directory exists
        os.makedirs(LOG_DIR, exist_ok=True)
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        datefmt = "%H:%M:%S"
        # Console handler
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(handler)
        # File handler
        file_handler = logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(fmt=fmt, datefmt=datefmt))
        root.addHandler(file_handler)
        # Default level can be raised/lowered later per logger
        root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Reduce verbosity of werkzeug logs (used by Flask)
    werkzeug_logger = logging.getLogger('werkzeug')
    werkzeug_logger.setLevel(logging.WARNING)

    werkzeug_logger.addFilter(lambda record: 
        "development server" not in record.getMessage() and 
        "Press CTRL+C" not in record.getMessage()
    )

    # Return a namespaced logger for the caller
    return logging.getLogger(name or __name__)


def set_level(logger: logging.Logger, level: str) -> None:
    """Set the level on the provided logger and keep propagation to root."""
    
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
