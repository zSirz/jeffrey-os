# Shim: ré-exporte depuis l'emplacement actuel (temporaire)
# TODO P1: Déplacer le vrai fichier ici et supprimer ce shim

try:
    from ..consciousness.dream_engine import *  # noqa: F401,F403
    from ..consciousness.dream_engine import DreamEngine  # noqa: F401
except ImportError as e:
    print(f"Warning: Shim import failed: {e}")

    # Fallback pour éviter les erreurs complètes
    class DreamEngine:
        def __init__(self):
            raise ImportError(f"{DreamEngine} not available - check ..consciousness.dream_engine")


# Pour debug
__shim__ = True
__original_location__ = "..consciousness.dream_engine"
