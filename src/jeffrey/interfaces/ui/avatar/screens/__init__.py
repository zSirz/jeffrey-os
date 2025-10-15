"""Module d'écrans pour Jeffrey.

Ce module contient les différents écrans de l'interface utilisateur.
"""

from jeffrey.interfaces.ui.screens.emotion_garden_screen import EmotionGardenScreen
from jeffrey.interfaces.ui.screens.emotion_visualizer import EmotionVisualizer
from jeffrey.interfaces.ui.screens.emotional_navigation import EmotionalNavigation
from jeffrey.interfaces.ui.screens.jeffrey_main_screen import JeffreyMainScreen
from jeffrey.interfaces.ui.screens.main_integration import MainIntegrationApp

SCREENS = ['JeffreyMainScreen', 'EmotionVisualizer', 'EmotionGardenScreen', 'EmotionalNavigation']
