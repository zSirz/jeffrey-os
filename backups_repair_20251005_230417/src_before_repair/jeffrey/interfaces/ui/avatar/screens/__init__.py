"""Module d'écrans pour Jeffrey.

Ce module contient les différents écrans de l'interface utilisateur.
"""

from ui.screens.emotion_garden_screen import EmotionGardenScreen
from ui.screens.emotion_visualizer import EmotionVisualizer
from ui.screens.emotional_navigation import EmotionalNavigation

# Exportation des écrans disponibles
from ui.screens.jeffrey_main_screen import JeffreyMainScreen
from ui.screens.main_integration import MainIntegrationApp

# Liste des écrans disponibles
SCREENS = ["JeffreyMainScreen", "EmotionVisualizer", "EmotionGardenScreen", "EmotionalNavigation"]
