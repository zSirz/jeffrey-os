"""
⚠️ COMPATIBILITY SHIM - TO BE REMOVED (Sprint M+2)

Ce module est un alias de compatibilité pour unified_memory.
Il redirige vers l'implémentation Production Ready avec fallback automatique.

ARCHITECTURE:
============
Target principal: jeffrey.core.memory.unified_memory (V3 Production Ready)
  → Async, sécurité XSS/SQL, SQLite backend, background tasks

Fallback automatique: jeffrey.core.memory.advanced_unified_memory (V2.1 Advanced)
  → Threading, dataclasses, compression gzip, si V3 échoue

Équipe: Claude + GPT/Marc + Grok + Gemini
"""

from __future__ import annotations

import importlib
import warnings
from typing import Any

warnings.warn(
    "jeffrey.core.unified_memory est un alias. Utilisez jeffrey.core.memory.unified_memory. (shim temporaire)",
    DeprecationWarning,
    stacklevel=2,
)

_Target: Any = None
_Unified: Any = None
_singleton: Any | None = None


def _load():
    global _Target, _Unified
    try:
        _Target = importlib.import_module("jeffrey.core.memory.unified_memory")
    except Exception:
        _Target = importlib.import_module("jeffrey.core.memory.advanced_unified_memory")
    _Unified = getattr(_Target, "UnifiedMemory", getattr(_Target, "AdvancedUnifiedMemory", None))
    if _Unified is None:
        raise ImportError("UnifiedMemory/AdvancedUnifiedMemory introuvable dans le module cible.")


_load()

# ré-export public
for _n in dir(_Target):
    if not _n.startswith("_"):
        globals()[_n] = getattr(_Target, _n)

UnifiedMemory = _Unified


def get_unified_memory(*a, **k):
    global _singleton
    if _singleton is None:
        _singleton = UnifiedMemory(*a, **k)
    return _singleton


__all__ = sorted({*(n for n in dir(_Target) if not n.startswith('_')), "UnifiedMemory", "get_unified_memory"})

# Métadonnées du shim
__shim__ = True
__target__ = _Target.__name__ if _Target else "unknown"
__version__ = "1.0.0-shim"
__deprecated__ = True
__removal_sprint__ = "M+2"
__team__ = "Claude + GPT/Marc + Grok + Gemini"
