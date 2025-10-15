#!/usr/bin/env python
"""
Module d'intégration pour le visage de Jeffrey.
Facilite la transition entre l'ancienne et la nouvelle architecture.
"""

from widgets.energy_face import EnergyFaceWidget


class EnergyFaceIntegration:
    """
    Classe d'intégration pour faciliter la transition entre l'ancienne
    et la nouvelle architecture de EnergyFaceWidget.

    Cette classe assure que les codes existants qui utilisent l'ancienne
    implémentation continueront à fonctionner avec la nouvelle architecture
    refactorisée.
    """

    @staticmethod
    def create_energy_face(**kwargs):
        """
        Crée une nouvelle instance de EnergyFaceWidget refactorisée.

        Args:
            **kwargs: Arguments à passer au constructeur

        Returns:
            Instance de EnergyFaceWidget
        """
        return EnergyFaceWidget(**kwargs)

    @staticmethod
    def update_from_jeffrey_core(widget, jeffrey_core):
        """
        Met à jour le widget avec les informations du noyau Jeffrey.

        Args:
            widget: Instance de EnergyFaceWidget
            jeffrey_core: Instance de JeffreyEmotionalCore
        """
        if hasattr(widget, "update_from_jeffrey_core"):
            widget.update_from_jeffrey_core(jeffrey_core)

    @staticmethod
    def set_emotion(widget, emotion, intensity=0.5):
        """
        Définit l'émotion et son intensité.

        Args:
            widget: Instance de EnergyFaceWidget
            emotion: Nom de l'émotion
            intensity: Intensité de l'émotion (0.0 à 1.0)
        """
        if hasattr(widget, "emotion"):
            widget.emotion = emotion
        if hasattr(widget, "intensity"):
            widget.intensity = intensity

    @staticmethod
    def set_speaking(widget, is_speaking):
        """
        Définit l'état de parole.

        Args:
            widget: Instance de EnergyFaceWidget
            is_speaking: True si en train de parler, False sinon
        """
        if hasattr(widget, "is_speaking"):
            widget.is_speaking = is_speaking

    @staticmethod
    def trigger_effect(widget, effect_name, **effect_params):
        """
        Déclenche un effet émotionnel.

        Args:
            widget: Instance de EnergyFaceWidget
            effect_name: Nom de l'effet à déclencher
            **effect_params: Paramètres spécifiques à l'effet
        """
        # Mapper les noms d'effets à leurs méthodes
        effect_map = {
            "visual_feedback": lambda w, p: w.effects.trigger_visual_feedback(**p),
            "tears": lambda w, p: w.effects.trigger_tears(**p),
            "warmth_surge": lambda w, p: w.effects.trigger_warmth_surge(**p),
            "emotional_emptiness": lambda w, p: w.effects.trigger_emotional_emptiness(**p),
            "emotional_shiver": lambda w, p: w.effects.trigger_emotional_shiver(**p),
            "cheek_enhancement": lambda w, p: w.effects.trigger_cheek_enhancement(**p),
            "start_pleasure_halo": lambda w, p: w.effects.start_pleasure_halo(**p),
            "start_intimite_effect": lambda w, p: w.effects.start_intimite_effect(**p),
            "start_muscle_tensions": lambda w, p: w.effects.start_muscle_tensions(**p),
            "start_vibration_effect": lambda w, p: w.effects.start_vibration_effect(**p),
            "start_mental_nebula": lambda w, p: w.effects.start_mental_nebula(**p),
        }

        # Vérifier si l'effet est supporté
        if effect_name in effect_map:
            # Déclencher l'effet
            effect_map[effect_name](widget, effect_params)
        else:
            print(f"Effet non supporté: {effect_name}")

    @staticmethod
    def stop_effect(widget, effect_name):
        """
        Arrête un effet émotionnel.

        Args:
            widget: Instance de EnergyFaceWidget
            effect_name: Nom de l'effet à arrêter
        """
        # Mapper les noms d'effets à leurs méthodes d'arrêt
        stop_map = {
            "pleasure_halo": lambda w: w.effects.stop_pleasure_halo(),
            "intimite_effect": lambda w: w.effects.stop_intimite_effect(),
            "muscle_tensions": lambda w: w.effects.stop_muscle_tensions(),
            "vibration_effect": lambda w: w.effects.stop_vibration_effect(),
            "mental_nebula": lambda w: w.effects.stop_mental_nebula(),
            "all_effects": lambda w: w.effects.stop_all_effects(),
        }

        # Vérifier si l'effet est supporté
        if effect_name in stop_map:
            # Arrêter l'effet
            stop_map[effect_name](widget)
        else:
            print(f"Arrêt d'effet non supporté: {effect_name}")

    @staticmethod
    def animate_lips_from_text(widget, text):
        """
        Anime les lèvres à partir d'un texte.

        Args:
            widget: Instance de EnergyFaceWidget
            text: Texte à traiter pour l'animation labiale
        """
        if hasattr(widget, "animate_lips_from_text"):
            widget.animate_lips_from_text(text)

    @staticmethod
    def apply_blink_state(widget):
        """
        Fait cligner les yeux.

        Args:
            widget: Instance de EnergyFaceWidget
        """
        if hasattr(widget, "apply_blink_state"):
            widget.apply_blink_state()


# Fonction d'aide pour la transition
def get_energy_face(**kwargs):
    """
    Fonction d'aide pour obtenir une instance de EnergyFaceWidget.

    Args:
        **kwargs: Arguments à passer au constructeur

    Returns:
        Instance de EnergyFaceWidget
    """
    return EnergyFaceIntegration.create_energy_face(**kwargs)
