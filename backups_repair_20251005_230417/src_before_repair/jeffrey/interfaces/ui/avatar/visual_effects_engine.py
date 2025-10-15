"""
VisualEffectsEngine - Moteur d'effets visuels pour l'interface de Jeffrey

Ce module gère les effets visuels liés aux émotions et aux interactions,
permettant de synchroniser les manifestations visuelles avec l'état émotionnel
et les réponses vocales de Jeffrey.
"""

import logging
import random
import threading
import time
from typing import Any


class VisualEffectsEngine:
    """
    Moteur d'effets visuels qui coordonne les animations et effets en fonction
    des émotions et des interactions.
    """

    def __init__(self, ui_controller=None):
        """
        Initialise le moteur d'effets visuels.

        Args:
            ui_controller: Contrôleur d'interface utilisateur (optionnel)
        """
        self.logger = logging.getLogger(__name__)
        self.ui_controller = ui_controller

        # État interne
        self.current_emotion = "neutral"
        self.effect_intensity = 0.5
        self.active_effects = []
        self.effect_threads = []

        # Mapping des émotions aux effets visuels
        self.emotion_effects_map = {
            "happy": ["halo_lumineux", "sparkles", "warm_glow"],
            "excited": ["energy_waves", "sparkles", "pulsating_halo"],
            "joyful": ["halo_lumineux", "sparkles", "rainbow_effect"],
            "sad": ["brume", "blue_tint", "slow_fadeout"],
            "melancholic": ["brume", "blue_tint", "rain_effect"],
            "disappointed": ["brume", "gray_tint", "slow_fadeout"],
            "angry": ["flash_rouge", "pulsating_red", "heat_waves"],
            "frustrated": ["flash_rouge", "pulsating_orange", "heat_waves"],
            "annoyed": ["flash_jaune", "pulsating_orange", "heat_waves"],
            "calm": ["slow_pulse_bleu", "soft_glow", "gentle_waves"],
            "peaceful": ["slow_pulse_bleu", "soft_glow", "gentle_waves"],
            "relaxed": ["slow_pulse_vert", "soft_glow", "gentle_waves"],
            "neutral": ["subtle_glow", "gentle_pulse", "balanced_aura"],
        }

        # Effets disponibles et leurs configurations
        self.available_effects = {
            "halo_lumineux": {
                "color": (255, 255, 190),  # Jaune clair
                "fade_in": 0.5,  # secondes
                "fade_out": 1.0,
                "scale": 1.2,
                "opacity": 0.7,
            },
            "sparkles": {
                "color": (255, 255, 255),  # Blanc
                "count": 20,
                "duration": 2.0,
                "speed": 1.0,
            },
            "brume": {
                "color": (100, 100, 150),  # Bleu grisâtre
                "opacity": 0.5,
                "spread": 1.0,
                "duration": 3.0,
            },
            "flash_rouge": {
                "color": (255, 0, 0),  # Rouge
                "intensity": 0.8,
                "duration": 0.3,
                "count": 2,
            },
            "slow_pulse_bleu": {
                "color": (0, 100, 255),  # Bleu
                "frequency": 0.5,  # Hz
                "duration": 5.0,
                "fade": 1.0,
            },
            "subtle_glow": {
                "color": (220, 220, 255),  # Bleu très clair
                "opacity": 0.3,
                "scale": 1.1,
                "pulse": False,
            },
        }

        self.logger.info("Moteur d'effets visuels initialisé")

    def trigger_emotion_effect(self, emotion: str, intensity: float = 0.5) -> list[str]:
        """
        Déclenche des effets visuels basés sur l'émotion spécifiée.

        Args:
            emotion: L'émotion à exprimer visuellement
            intensity: L'intensité de l'émotion (0.0 à 1.0)

        Returns:
            Liste des effets déclenchés
        """
        self.current_emotion = emotion.lower()
        self.effect_intensity = max(0.0, min(1.0, intensity))

        # Récupérer les effets associés à cette émotion
        emotion_key = self._normalize_emotion(self.current_emotion)
        available_effects = self.emotion_effects_map.get(emotion_key, ["subtle_glow"])

        # Sélectionner les effets en fonction de l'intensité
        num_effects = 1
        if self.effect_intensity > 0.3:
            num_effects = 2
        if self.effect_intensity > 0.7:
            num_effects = 3

        # Limiter au nombre d'effets disponibles
        num_effects = min(num_effects, len(available_effects))

        # Sélectionner des effets au hasard
        selected_effects = random.sample(available_effects, num_effects)

        # Déclencher chaque effet
        triggered_effects = []
        for effect_name in selected_effects:
            if self._trigger_effect(effect_name, self.effect_intensity):
                triggered_effects.append(effect_name)

        # Journaliser les effets déclenchés
        effect_str = ", ".join(triggered_effects) if triggered_effects else "aucun"
        self.logger.info(f"Effets visuels pour '{emotion}' (intensité {intensity:.1f}): {effect_str}")

        return triggered_effects

    def _normalize_emotion(self, emotion: str) -> str:
        """
        Normalise le nom de l'émotion en la faisant correspondre à une catégorie connue.

        Args:
            emotion: Nom de l'émotion à normaliser

        Returns:
            Nom de l'émotion normalisé
        """
        emotion = emotion.lower()

        # Mapping des synonymes aux émotions principales
        emotion_mapping = {
            "heureux": "happy",
            "joyeux": "happy",
            "content": "happy",
            "triste": "sad",
            "malheureux": "sad",
            "chagrin": "sad",
            "fâché": "angry",
            "furieux": "angry",
            "irrité": "angry",
            "en colère": "angry",
            "calme": "calm",
            "serein": "calm",
            "tranquille": "calm",
            "neutre": "neutral",
            "normal": "neutral",
            "standard": "neutral",
            "excité": "excited",
            "enthousiaste": "excited",
            "joie": "joyful",
            "jubilation": "joyful",
            "mélancolique": "melancholic",
            "nostalgique": "melancholic",
            "déçu": "disappointed",
            "désappointé": "disappointed",
            "frustré": "frustrated",
            "contrarié": "frustrated",
            "agacé": "annoyed",
            "ennuyé": "annoyed",
            "paisible": "peaceful",
            "apaisé": "peaceful",
            "détendu": "relaxed",
            "relax": "relaxed",
        }

        # Essayer de trouver une correspondance directe
        if emotion in self.emotion_effects_map:
            return emotion

        # Essayer de trouver une correspondance dans le mapping des synonymes
        if emotion in emotion_mapping:
            return emotion_mapping[emotion]

        # Si aucune correspondance, retourner neutral
        return "neutral"

    def _trigger_effect(self, effect_name: str, intensity: float) -> bool:
        """
        Déclenche un effet visuel spécifique.

        Args:
            effect_name: Nom de l'effet à déclencher
            intensity: Intensité de l'effet (0.0 à 1.0)

        Returns:
            True si l'effet a été déclenché avec succès, False sinon
        """
        # Vérifier si l'effet est disponible
        if effect_name not in self.available_effects:
            self.logger.warning(f"Effet '{effect_name}' non disponible")
            return False

        # Récupérer la configuration de l'effet
        effect_config = self.available_effects[effect_name].copy()

        # Ajuster les paramètres en fonction de l'intensité
        self._adjust_effect_config(effect_config, intensity)

        # Si l'UI controller est disponible, lui demander de jouer l'effet
        if self.ui_controller:
            try:
                self.ui_controller.play_visual_effect(effect_name, effect_config)
                return True
            except Exception as e:
                self.logger.error(f"Erreur lors du déclenchement de l'effet '{effect_name}': {e}")
                return False

        # Simuler l'effet si aucun UI controller n'est disponible
        self.logger.debug(f"Simulation de l'effet '{effect_name}' avec intensité {intensity:.1f}")

        # Ajouter l'effet à la liste des effets actifs
        self.active_effects.append((effect_name, effect_config))

        # Lancer un thread pour simuler la durée de l'effet
        effect_thread = threading.Thread(
            target=self._simulate_effect_duration, args=(effect_name, effect_config), daemon=True
        )
        self.effect_threads.append(effect_thread)
        effect_thread.start()

        return True

    def _adjust_effect_config(self, effect_config: dict[str, Any], intensity: float) -> None:
        """
        Ajuste les paramètres de l'effet en fonction de l'intensité.

        Args:
            effect_config: Configuration de l'effet à ajuster
            intensity: Intensité de l'effet (0.0 à 1.0)
        """
        # Ajuster l'opacité, l'échelle, la durée, etc. en fonction de l'intensité
        if "opacity" in effect_config:
            effect_config["opacity"] = effect_config["opacity"] * intensity

        if "scale" in effect_config:
            # Augmenter l'échelle avec l'intensité
            effect_config["scale"] = 1.0 + (effect_config["scale"] - 1.0) * intensity

        if "duration" in effect_config:
            # Augmenter la durée avec l'intensité
            effect_config["duration"] = effect_config["duration"] * (0.5 + intensity * 0.5)

        if "count" in effect_config:
            # Plus d'éléments ou de répétitions avec plus d'intensité
            effect_config["count"] = int(effect_config["count"] * (0.5 + intensity * 0.5))

    def _simulate_effect_duration(self, effect_name: str, effect_config: dict[str, Any]) -> None:
        """
        Simule la durée d'un effet visuel.

        Args:
            effect_name: Nom de l'effet à simuler
            effect_config: Configuration de l'effet
        """
        duration = effect_config.get("duration", 1.0)
        self.logger.debug(f"Effet '{effect_name}' actif pendant {duration:.1f} secondes")

        # Simuler la durée de l'effet
        time.sleep(duration)

        # Retirer l'effet de la liste des effets actifs
        self.active_effects = [(name, cfg) for name, cfg in self.active_effects if name != effect_name]
        self.logger.debug(f"Effet '{effect_name}' terminé")

    def get_active_effects(self) -> list[str]:
        """
        Récupère la liste des effets actuellement actifs.

        Returns:
            Liste des noms d'effets actifs
        """
        return [name for name, _ in self.active_effects]

    def clear_all_effects(self) -> None:
        """
        Efface tous les effets visuels actifs.
        """
        if self.ui_controller:
            self.ui_controller.clear_visual_effects()

        # Réinitialiser l'état interne
        self.active_effects = []
        self.logger.info("Tous les effets visuels ont été effacés")

    def sync_with_voice(self, emotion: str, text: str, intensity: float = 0.5) -> list[str]:
        """
        Synchronise les effets visuels avec la voix et l'émotion.

        Args:
            emotion: L'émotion à exprimer
            text: Le texte à prononcer (pour détecter des mots clés)
            intensity: L'intensité de l'émotion (0.0 à 1.0)

        Returns:
            Liste des effets déclenchés
        """
        # Ajuster l'intensité en fonction du contenu du texte
        adjusted_intensity = self._adjust_intensity_by_text(text, intensity)

        # Déclencher les effets visuels
        return self.trigger_emotion_effect(emotion, adjusted_intensity)

    def _adjust_intensity_by_text(self, text: str, base_intensity: float) -> float:
        """
        Ajuste l'intensité en fonction du contenu du texte.

        Args:
            text: Le texte à analyser
            base_intensity: L'intensité de base

        Returns:
            Intensité ajustée
        """
        # Mots d'intensification
        intensifiers = ["très", "extrêmement", "incroyablement", "vraiment", "totalement"]
        diminishers = ["un peu", "légèrement", "modérément", "à peine"]

        # Points d'exclamation
        exclamation_boost = min(0.3, text.count("!") * 0.1)

        # Majuscules
        uppercase_ratio = sum(1 for c in text if c.isupper()) / max(1, len(text))
        uppercase_boost = min(0.2, uppercase_ratio)

        # Mots intensifieurs/diminueurs
        text_lower = text.lower()
        intensity_mod = 0

        for word in intensifiers:
            if word in text_lower:
                intensity_mod += 0.1

        for word in diminishers:
            if word in text_lower:
                intensity_mod -= 0.1

        # Calculer l'intensité ajustée
        adjusted_intensity = base_intensity + exclamation_boost + uppercase_boost + intensity_mod

        # S'assurer que l'intensité reste dans les limites valides
        return max(0.1, min(1.0, adjusted_intensity))
