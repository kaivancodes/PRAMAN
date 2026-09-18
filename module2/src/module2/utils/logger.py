"""Structured logging for Module 2.

Omits sensitive document images, raw bytes, and PII.
"""

import logging
import sys
from typing import Optional

_LOGGERS = {}


def get_logger(name: str = "module2", level: Optional[str] = None) -> logging.Logger:
    """Retrieve or configure a standardized logger for Module 2.

    Args:
        name: Logger name hierarchy.
        level: Optional log level string (e.g. 'INFO', 'DEBUG', 'WARNING').

    Returns:
        Configured logging.Logger instance.
    """
    if name in _LOGGERS:
        if level:
            _LOGGERS[name].setLevel(getattr(logging, level.upper(), logging.INFO))
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.propagate = False

    log_level = getattr(logging, level.upper(), logging.INFO) if level else logging.INFO
    logger.setLevel(log_level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    _LOGGERS[name] = logger
    return logger
