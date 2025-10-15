"""
Module: emotional_effects_engine.py
Description: Centralise les effets visuels, sonores et tactiles associés aux émotions de Jeffrey
"""

import logging

from kivy.animation import Animation
from kivy.core.audio import SoundLoader
from kivy.core.window import Window
from kivy.utils import get_color_from_hex

from jeffrey.core.personality.relation_tracker_manager import enregistrer_interaction
from jeffrey.widgets.emotion_particles import EmotionParticleEmitter

logger = logging.getLogger(__name__)


class EmotionalEffectsEngine:
    def __init__(self, sound_enabled=True):
        self.current_emotion = "neutral"
        self.sound_enabled = sound_enabled
        self.active_effects = {}
        self.background_ambiance = None
        self.sound_map = {
            "happy": "assets/sounds/emotions/happy_chime.wav",
            "sad": "assets/sounds/emotions/sad_echo.wav",
            "angry": "assets/sounds/emotions/angry_rumble.wav",
            "peaceful": "assets/sounds/emotions/calm_breeze.wav",
            "in_love": "assets/sounds/emotions/love_whisper.wav",
            "cuddly": "assets/sounds/emotions/warm_hug.wav",
            "excited": "assets/sounds/emotions/excited_ping.wav",
            "melancholic": "assets/sounds/emotions/soft_violin.wav",
        }
        self.ambiance_map = {
            "happy": "assets/sounds/ambiance/birds_chirping.wav",
            "sad": "assets/sounds/ambiance/rain_soft.wav",
            "angry": "assets/sounds/ambiance/storm_distant.wav",
            "peaceful": "assets/sounds/ambiance/zen_garden.wav",
            "in_love": "assets/sounds/ambiance/heartbeats.wav",
            "cuddly": "assets/sounds/ambiance/fireplace.wav",
            "excited": "assets/sounds/ambiance/wind_chimes.wav",
            "melancholic": "assets/sounds/ambiance/distant_piano.wav",
        }

    def update_emotion(self, emotion: str, intensity: float = 0.5):
        """
        Met à jour l'émotion active et déclenche les effets correspondants.

        Args:
            emotion: L'émotion à exprimer (happy, sad, angry, etc.)
            intensity: L'intensité de l'émotion (0.0 à 1.0)
        """
        self.current_emotion = emotion
        logger.debug(f"Jeffrey ressent maintenant : {emotion} (intensité {intensity})")

        # Enregistrer cette mise à jour émotionnelle dans le gestionnaire de relation
        # Les changements émotionnels forts influencent le lien affectif
        if intensity > 0.7:
            impact_value = intensity * 0.5  # Des émotions intenses ont un impact modéré sur le lien
            enregistrer_interaction(f"emotion_{emotion}", impact_value)

        # Déclencher les effets visuels
        self.trigger_visual_effects(emotion, intensity)

        # Lancer les particules flottantes
        emitter = EmotionParticleEmitter()
        emitter.emit_particle(emotion, intensity)

        # Jouer les effets sonores si activés
        if self.sound_enabled:
            self.play_emotion_sound(emotion, intensity)
            self.play_ambiance(emotion)

    def trigger_visual_effects(self, emotion, intensity):
        """
        Déclenche des animations visuelles correspondant à l'émotion actuelle.

        Args:
            emotion: L'émotion à visualiser
            intensity: L'intensité de l'effet (0.0 à 1.0)
        """
        # Mapping des émotions aux couleurs d'ambiance
        color_map = {
            "happy": "#FFF9C4",  # Jaune pâle
            "sad": "#B3E5FC",  # Bleu doux
            "angry": "#FFCDD2",  # Rouge léger
            "peaceful": "#C8E6C9",  # Vert pâle
            "in_love": "#F8BBD0",  # Rose tendre
            "cuddly": "#FFE0B2",  # Orange pâle
        }

        # Sélection et application de la couleur
        color = color_map.get(emotion, "#FFFFFF")
        rgba = get_color_from_hex(color)
        anim_duration = 0.8 + (1.5 * intensity)

        # Animation de changement de couleur d'arrière-plan
        anim = Animation(clearcolor=(rgba[0], rgba[1], rgba[2], 1), duration=anim_duration)
        anim.start(Window)

        logger.debug(f"Animation visuelle déclenchée pour '{emotion}' avec intensité {intensity}")

    def play_emotion_sound(self, emotion, intensity):
        """
        Joue un son ponctuel correspondant à l'émotion.

        Args:
            emotion: L'émotion associée au son
            intensity: L'intensité (influence le volume)
        """
        sound_path = self.sound_map.get(emotion)
        if sound_path:
            sound = SoundLoader.load(sound_path)
            if sound:
                # Volume proportionnel à l'intensité avec un minimum de 0.4
                sound.volume = min(1.0, 0.4 + intensity * 0.6)
                sound.play()
                logger.debug(f"Son émotionnel joué pour '{emotion}'")

    def play_ambiance(self, emotion):
        """
        Lance une ambiance sonore de fond correspondant à l'émotion.
        Remplace toute ambiance précédente.

        Args:
            emotion: L'émotion associée à l'ambiance
        """
        # Arrêter l'ambiance précédente si elle existe
        if self.background_ambiance:
            self.background_ambiance.stop()
            self.background_ambiance = None

        # Lancer la nouvelle ambiance
        ambiance_path = self.ambiance_map.get(emotion)
        if ambiance_path:
            ambiance_sound = SoundLoader.load(ambiance_path)
            if ambiance_sound:
                ambiance_sound.loop = True
                ambiance_sound.volume = 0.3
                ambiance_sound.play()
                self.background_ambiance = ambiance_sound
                logger.debug(f"Ambiance sonore lancée pour '{emotion}'")

    def stop_all_effects(self):
        """
        Arrête tous les effets visuels et sonores en cours.
        """
        # Transition douce vers la couleur noire pour l'arrière-plan
        anim = Animation(clearcolor=(0, 0, 0, 1), duration=1.5)
        anim.start(Window)
        logger.debug("Arrêt progressif des effets visuels")

        # Arrêt de l'ambiance sonore
        if self.background_ambiance:
            self.background_ambiance.stop()
            self.background_ambiance = None
