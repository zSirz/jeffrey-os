#!/usr/bin/env python3
"""
Logger wrapper pour Jeffrey OS
Évite les dépendances Kivy dans le Core
"""

import logging
import sys

# Configuration de base
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)

# Logger principal
logger = logging.getLogger("jeffrey")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name"""
    return logging.getLogger(f"jeffrey.{name}")


# Export
__all__ = ["logger", "get_logger"]
