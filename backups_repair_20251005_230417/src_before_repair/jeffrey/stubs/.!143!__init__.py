"""
Stubs pour modules manquants
"""

from pathlib import Path


def inject_stubs_to_sys_modules(config=None):
    """Injecte les stubs dans sys.modules pour les modules manquants"""
    stubs_dir = Path(__file__).parent
