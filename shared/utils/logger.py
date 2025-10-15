#!/usr/bin/env python3
"""
Logger simplifié pour Jeffrey OS
Évite les dépendances Kivy dans le Core
"""

import logging
import os
import sys

# Détection du mode headless
IS_HEADLESS = os.environ.get('KIVY_NO_ARGS') == '1'

# Configuration de base
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', stream=sys.stdout
    )

# Logger principal
logger = logging.getLogger('jeffrey')

# Si Kivy est disponible et non headless, on peut l'utiliser
try:
    if not IS_HEADLESS:
        from kivy.logger import Logger as KivyLogger

        # Wrapper pour compatibilité
        logger = KivyLogger
except ImportError:
    # Kivy n'est pas disponible, on garde le logger Python standard
    pass

# Export
__all__ = ['logger']
