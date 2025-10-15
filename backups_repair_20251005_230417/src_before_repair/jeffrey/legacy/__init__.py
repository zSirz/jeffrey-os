"""
Legacy P1 modules - Migration progressive vers P2
Ces modules seront progressivement refactorisés et intégrés dans la nouvelle architecture
"""

# Réexporter pour compatibilité
try:
    from .consciousness import *
except ImportError:
    pass

try:
    from .memory_manager import *
except ImportError:
    pass

try:
    from .emotional_core import *
except ImportError:
    pass

try:
    from .dream_engine import *
except ImportError:
    pass

try:
    from .symbiosis import *
except ImportError:
    pass

try:
    from .brain_kernel import *
except ImportError:
    pass

print("✅ Legacy modules loaded for progressive migration")
