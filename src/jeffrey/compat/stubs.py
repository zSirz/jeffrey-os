"""
Stubs pour modules manquants (Rust, modules récupérés).
Permet à la démo de tourner même si certains modules ne sont pas encore intégrés.
"""

import logging
import os

logger = logging.getLogger(__name__)

DEMO_MODE = os.getenv("JEFFREY_DEMO", "1") == "1"


# Stub pour ImmuneSystem (module Rust)
class ImmuneSystem:
    def __init__(self, config=None):
        if DEMO_MODE:
            logger.warning("ImmuneSystem running in stub mode (JEFFREY_DEMO=1)")
        self.config = config or {}

    def scan(self):
        return {"status": "healthy", "threats": [], "stub": True}

    def defend(self, threat):
        return {"blocked": True, "stub": True}


# Stub pour PapaControl (module Rust)
class PapaControl:
    def __init__(self, config=None):
        if DEMO_MODE:
            logger.warning("PapaControl running in stub mode (JEFFREY_DEMO=1)")
        self.config = config or {}

    def check_privileges(self):
        return {"admin": False, "stub": True}


# Stub pour modules Python manquants
class IAOrchestratorUltimate:
    def __init__(self, config=None):
        if DEMO_MODE:
            logger.warning("IAOrchestratorUltimate running in stub mode")
        self.config = config or {}

    def orchestrate(self, task):
        return {"status": "simulated", "stub": True}


# Helper pour import avec fallback
def import_with_fallback(module_path, class_name, stub_class):
    """
    Tente d'importer le vrai module, sinon utilise le stub.

    Usage:
        ImmuneSystem = import_with_fallback(
            "jeffrey_rust.security",
            "ImmuneSystem",
            stubs.ImmuneSystem
        )
    """
    try:
        module = __import__(module_path, fromlist=[class_name])
        return getattr(module, class_name)
    except (ImportError, AttributeError):
        logger.info(f"Using stub for {module_path}.{class_name}")
        return stub_class
