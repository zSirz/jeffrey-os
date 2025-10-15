"""
Package initialization for Jeffrey OS
"""

__version__ = "0.1.0"
__all__ = []

# Auto-découverte des modules
import importlib
from pathlib import Path

_current_dir = Path(__file__).parent
for py_file in _current_dir.glob("*.py"):
    if not py_file.name.startswith("_") and py_file.name != "__init__.py":
        module_name = py_file.stem
        try:
            importlib.import_module(f".{module_name}", package=__package__)
            __all__.append(module_name)
        except ImportError:
            pass
