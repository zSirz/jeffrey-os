#!/usr/bin/env python3
"""
SHIM STRICT - Redirection vers implémentation réelle
GÉNÉRÉ AUTOMATIQUEMENT - NE PAS ÉDITER
Target: src/jeffrey/core/jeffrey_emotional_core.py
"""

import importlib.util
import os
import sys
from pathlib import Path

REAL_PATH = "src/jeffrey/core/jeffrey_emotional_core.py"
STRICT_MODE = os.getenv("JEFFREY_ALLOW_FALLBACK", "0") != "1"

_LOADING = set()


def _check_cycle(module_name: str):
    """Détection de cycles."""
    if module_name in _LOADING:
        raise ImportError(f"❌ CYCLE : {module_name}")
    _LOADING.add(module_name)


def _load_real_module():
    """Charge le module réel."""
    module_name = __name__
    _check_cycle(module_name)

    real_file = Path(__file__).parent.parent.parent / REAL_PATH

    if not real_file.exists():
        if STRICT_MODE:
            raise ImportError(f"❌ Fichier manquant : {real_file}")
        return None

    try:
        spec = importlib.util.spec_from_file_location("_real", real_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Spec invalide : {real_file}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        _LOADING.discard(module_name)
        return module
    except Exception as e:
        _LOADING.discard(module_name)
        raise ImportError(f"Erreur chargement {real_file}: {e}")


_real_module = _load_real_module()

if _real_module:
    globals().update({k: v for k, v in vars(_real_module).items() if not k.startswith('_')})
