"""
visual_emotion_controller.py – Contrôle visuel des émotions

Ce module agit comme un pont entre l'état émotionnel de Jeffrey
et l'affichage graphique dynamique (couleurs, ambiance, effets visuels).

Il récupère l'émotion dominante et applique les paramètres visuels adaptés.
"""

from jeffrey.core.emotions.emotional_engine import EmotionalEngine
from jeffrey.interfaces.ui.emotion_visual_engine import EmotionVisualEngine


class VisualEmotionController:
    def __init__(self, emotional_engine: EmotionalEngine | None = None):
        self.engine = emotional_engine or EmotionalEngine()
        self.visual_engine = EmotionVisualEngine()
        self.current_emotion = None
        self.visual_state: dict[str, any] = {}

    def update_visual_state(self):
        """
        Met à jour les paramètres visuels en fonction de l’émotion dominante.
        """
        new_emotion = self.engine.get_dominant_emotion() or 'neutre'
        if new_emotion != self.current_emotion:
            transition = self.visual_engine.get_transition(self.current_emotion or 'neutre', new_emotion)
            self.visual_state = self.visual_engine.get_visual_state_with_fallback(new_emotion)
            self.current_emotion = new_emotion
            print(f"[🎨] Changement visuel → émotion '{new_emotion}' avec transition : {transition}")
            return (self.visual_state, transition)
        else:
            return (self.visual_state, None)

    def get_current_visual_state(self) -> dict[str, any]:
        """
        Retourne le dernier état visuel appliqué.
        """
        return self.visual_state
