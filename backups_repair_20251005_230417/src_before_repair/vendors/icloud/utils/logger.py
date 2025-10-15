"""Unified logger for all modules - SINGLE SOURCE OF TRUTH"""

import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Configuration via environment
LOG_DIR = Path(os.environ.get("JEFFREY_LOG_DIR", "logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_LEVEL = os.environ.get("JEFFREY_LOG_LEVEL", "INFO")

# Cache pour éviter reconfiguration
_configured_loggers = set()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with proper configuration"""
    logger_name = f"jeffrey.{name}"
    logger = logging.getLogger(logger_name)

    # Configure only once
    if logger_name not in _configured_loggers:
        # Protection contre niveau invalide
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
        logger.setLevel(level)

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(console)

        # File handler with rotation and compression
        file_handler = RotatingFileHandler(
            LOG_DIR / f"{name}.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s")
        )
        logger.addHandler(file_handler)

        # Prevent propagation to root logger
        logger.propagate = False

        _configured_loggers.add(logger_name)

    return logger


# Root logger for backward compatibility
logger = get_logger("main")


# Decorator for method logging
def log_method(func):
    """Decorator to log method calls with timing"""
    import functools
    import time

    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.perf_counter()
        logger.debug(f"→ {func.__name__} starting")
        try:
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(f"← {func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            logger.error(f"✗ {func.__name__} failed: {e}")
            raise

    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start = time.perf_counter()
        logger.debug(f"→ {func.__name__} starting")
        try:
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.debug(f"← {func.__name__} completed in {elapsed:.3f}s")
            return result
        except Exception as e:
            logger.error(f"✗ {func.__name__} failed: {e}")
            raise

    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


__all__ = ["logger", "get_logger", "log_method"]
