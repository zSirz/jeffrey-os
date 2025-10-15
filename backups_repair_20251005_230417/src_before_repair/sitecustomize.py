"""
Jeffrey OS - Import Hook Production (Version Finale Corrigée)
=============================================================

Import hook via MetaPathFinder avec corrections critiques :
- PathFinder au lieu de find_spec (évite récursion)
- Interrupteur d'urgence (JEFFREY_ALIAS_DISABLE)
- Cache LRU pour performance
- Pas de sys.exit() dans sitecustomize
- Gestion propre des namespace packages

Principe : ZÉRO INVENTION - Uniquement vers modules existants

Auteurs: Team Jeffrey (David, Claude, GPT, Grok, Gemini)
Date: 2025-10-05 (Version Production)
"""

import importlib
import importlib.abc
import importlib.util
import os
import sys
from functools import lru_cache  # CORRECTION GPT 3
from importlib.machinery import PathFinder  # CORRECTION GPT 1
from types import ModuleType

# ============================================================================
# CONFIGURATION
# ============================================================================

DEBUG_MOCKS = os.getenv("JEFFREY_DEBUG_MOCKS", "0") == "1"

# CORRECTION GPT 2 : Interrupteur d'urgence (sans sys.exit)
HOOK_DISABLED = os.getenv("JEFFREY_ALIAS_DISABLE", "0") == "1"
if DEBUG_MOCKS and HOOK_DISABLED:
    print("[sitecustomize] ⚠️  Import hook DÉSACTIVÉ (env)")

if DEBUG_MOCKS and not HOOK_DISABLED:
    print("[sitecustomize] Jeffrey OS - Import Hook Production")


# ============================================================================
# RÈGLES D'ALIAS (MINIMALES - Prouvées sur Disque)
# ============================================================================

# Alias exacts (uniquement vers des fichiers confirmés existants)
ALIAS_EXACT = {
    # Top 5 vérifiés
    "core.agi_fusion.agi_orchestrator": "jeffrey.core.orchestration.agi_orchestrator",
    "core.config": "jeffrey.core.neuralbus.config",
    "core.consciousness.jeffrey_consciousness_v3": "jeffrey.core.consciousness.jeffrey_consciousness_v3",
    "core.conversation": "jeffrey.core.personality",
    "core.emotional_memory": "jeffrey.core.memory.advanced.emotional_memory",
    # Modules vendorisés iCloud
    "jeffrey.core.jeffrey_emotional_core": "vendors.icloud.jeffrey_emotional_core",
    "core.jeffrey_emotional_core": "vendors.icloud.jeffrey_emotional_core",
    "jeffrey.core.emotions.emotion_prompt_detector": "vendors.icloud.emotions.emotion_prompt_detector",
    # Autres modules déplacés (vérifiés)
    "jeffrey.modules.config.secrets_manager": "jeffrey.infrastructure.security.secrets_manager",
    "jeffrey.core.personality.style_affectif_adapter": "jeffrey.core.personality.conversation_personality",
    # NOTE : Ajouter ici UNIQUEMENT après vérification sur disque
    # Commande : find src/jeffrey -name "*<module>*.py"
}

# Option de sécurité pour le package racine core (CORRECTION GPT 3)
MAP_CORE_ROOT = os.getenv("JEFFREY_MAP_CORE_ROOT", "0") == "1"

# Règles de préfixe (appliquées si pas d'alias exact)
PREFIX_RULES = [("core.", "jeffrey.core.")] + ([("core", "jeffrey.core")] if MAP_CORE_ROOT else [])

# Préfixes à ne JAMAIS toucher
NO_TOUCH = ("jeffrey", "vendors", "sitecustomize", "test", "__pycache__")


# ============================================================================
# CACHE POUR PERFORMANCE (CORRECTION GPT 3)
# ============================================================================


@lru_cache(maxsize=4096)
def _module_exists(module_name: str, path_tuple=None) -> bool:
    """
    Vérifie si un module existe via importlib.util.find_spec (avec cache).
    Évite les scans disque répétés.

    Args:
        module_name: Nom du module à vérifier
        path_tuple: Chemin pour namespace packages (tuple ou None)
    """
    try:
        path = list(path_tuple) if path_tuple else None
        spec = importlib.util.find_spec(module_name, path)
        return spec is not None
    except (ValueError, ModuleNotFoundError, AttributeError):
        return False


# ============================================================================
# FONCTION DE CALCUL DE CIBLE
# ============================================================================


def _target_for(fullname: str) -> str | None:
    """
    Calcule la cible de redirection pour un module.

    Returns:
        Nom du module cible, ou None si pas de redirection
    """
    # Ne jamais toucher ces préfixes
    for prefix in NO_TOUCH:
        if fullname == prefix or fullname.startswith(prefix + "."):
            return None

    # Alias exact prioritaire
    if fullname in ALIAS_EXACT:
        return ALIAS_EXACT[fullname]

    # Règles de préfixe
    for src_prefix, dst_prefix in PREFIX_RULES:
        if fullname == src_prefix:
            return dst_prefix
        if fullname.startswith(src_prefix + "."):
            return dst_prefix + fullname[len(src_prefix) :]

    return None


# ============================================================================
# IMPORT HOOK (MetaPathFinder + Loader)
# ============================================================================


class JeffreyAliasFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """
    Import hook production avec corrections GPT :
    - PathFinder au lieu de find_spec
    - Cache LRU pour performance
    - Gestion d'erreurs robuste
    - Support namespace packages
    """

    def find_spec(self, fullname, path=None, target=None):
        """Appelé par Python pour chaque import"""
        # Ne jamais écraser un module déjà importé
        if fullname in sys.modules:
            return None

        # Calculer la cible en premier
        target_name = _target_for(fullname)
        if target_name is None:
            return None

        # Vérifier que la cible existe directement avec PathFinder (évite récursion)
        try:
            path_tuple = tuple(path) if path else None
            real_path = list(path_tuple) if path_tuple else None
            target_spec = PathFinder.find_spec(target_name, real_path)
            if target_spec is None:
                return None
        except (ValueError, ModuleNotFoundError, AttributeError):
            return None

        # Ne pas rediriger si le module source existe déjà (évite conflits)
        try:
            source_spec = PathFinder.find_spec(fullname, real_path)
            if source_spec is not None:
                return None
        except (ValueError, ModuleNotFoundError, AttributeError):
            pass

        # On peut gérer ce module
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        """Créer le module - Python gère"""
        return None

    def exec_module(self, module):
        """Exécuter = charger la cible et créer l'alias"""
        fullname = module.__name__
        target_name = _target_for(fullname)

        if target_name is None:
            raise ImportError(f"Pas de cible pour {fullname}")

        if DEBUG_MOCKS:
            print(f"[alias] {fullname} → {target_name}")

        # Importer le module cible réel
        try:
            real_module = importlib.import_module(target_name)
        except Exception as e:
            raise ImportError(f"Échec import {target_name}: {e}") from e

        # Créer l'alias
        sys.modules[fullname] = real_module

        # Attacher au parent
        parts = fullname.split(".")
        if len(parts) > 1:
            parent_name = ".".join(parts[:-1])

            if parent_name not in sys.modules:
                parent = ModuleType(parent_name)
                parent.__path__ = []
                sys.modules[parent_name] = parent
            else:
                parent = sys.modules[parent_name]

            setattr(parent, parts[-1], real_module)


# ============================================================================
# INSTALLATION DU HOOK
# ============================================================================


def install_import_hook():
    """Installe le hook si pas déjà présent"""
    for finder in sys.meta_path:
        if isinstance(finder, JeffreyAliasFinder):
            if DEBUG_MOCKS:
                print("[sitecustomize] Import hook déjà installé")
            return

    sys.meta_path.insert(0, JeffreyAliasFinder())

    if DEBUG_MOCKS:
        print("[sitecustomize] ✅ Import hook installé (production)")


# Installation automatique (seulement si pas désactivé)
if not HOOK_DISABLED:
    install_import_hook()


# ============================================================================
# FINALISATION
# ============================================================================

if DEBUG_MOCKS:
    print(f"[sitecustomize] ✅ Jeffrey OS prêt ({len(sys.modules)} modules)")
