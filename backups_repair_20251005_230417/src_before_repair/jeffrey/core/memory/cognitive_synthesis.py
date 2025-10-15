# Shim: ré-exporte depuis l'emplacement actuel (temporaire)
# TODO P1: Déplacer le vrai fichier ici et supprimer ce shim

try:
    from ..consciousness.cognitive_synthesis import *  # noqa: F401,F403
    from ..consciousness.cognitive_synthesis import CognitiveSynthesis  # noqa: F401
except ImportError as e:
    print(f"Warning: Shim import failed: {e}")

    # Fallback pour éviter les erreurs complètes
    class CognitiveSynthesis:
        def __init__(self):
            raise ImportError(f"{CognitiveSynthesis} not available - check ..consciousness.cognitive_synthesis")


# Pour debug
__shim__ = True
__original_location__ = "..consciousness.cognitive_synthesis"
