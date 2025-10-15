#!/usr/bin/env python
"""
energy_face_emotions.py - Gestion des émotions pour le visage de Jeffrey
Partie de la refactorisation du fichier energy_face.py d'origine (PACK 18)

Ce module gère les comportements liés aux émotions du visage :
- Traitement des changements d'émotions
- Réactions émotionnelles
- Expressions faciales liées aux émotions
- Transitions entre émotions
"""

import logging
import time

from kivy.animation import Animation
from kivy.clock import Clock


class EmotionHandler:
    """
    Gestionnaire des émotions pour le visage de Jeffrey.
    Contrôle les réactions émotionnelles et les expressions du visage.
    """

    def __init__(self, face_widget):
        """
        Initialise le gestionnaire d'émotions.

        Args:
            face_widget: Widget du visage (EnergyFaceCoreWidget)
        """
        self.face = face_widget

        # Définir les émotions de base
        self.basic_emotions = [
            "joie",
            "tristesse",
            "colère",
            "peur",
            "surprise",
            "dégoût",
            "neutre",
        ]

        # Émotions secondaires
        self.secondary_emotions = [
            "émerveillement",
            "mélancolie",
            "frustration",
            "anxiété",
            "curiosité",
            "sérénité",
            "fierté",
            "timidité",
            "ennui",
            "excitation",
        ]

        # Variables pour la transition entre émotions
        self.current_transition = None
        self._emotional_mirroring = False

        # Variables pour la dynamique émotionnelle
        self.emotional_inertia = 0.2  # Résistance au changement émotionnel
        self.emotion_stability = 0.7  # Stabilité de l'émotion actuelle (0-1)

        # Variables pour le plaisir affectif
        self.pleasure_level = 0.0
        self.pleasure_decay_timer = None
        self.last_affection_time = time.time()

        # Timers pour les effets émotionnels
        self._emotion_timers = {}

    def process_emotion_change(self, emotion: str, intensity: float = 0.5):
        """
        Traite un changement d'émotion et déclenche les effets appropriés.

        Args:
            emotion: Nouvelle émotion
            intensity: Intensité de l'émotion (0.0 à 1.0)
        """
        # Ajuster les facteurs selon l'émotion
        self._adjust_emotional_factors(emotion)

        # Déclencher les effets visuels selon l'émotion
        self._trigger_emotion_effects(emotion, intensity)

        # Transition douce si besoin
        previous_emotion = getattr(self.face, "_previous_emotion", None)
        if previous_emotion and previous_emotion != emotion:
            self._start_emotion_transition(previous_emotion, emotion, duration=0.8)

        # Mémoriser cette émotion pour la prochaine fois
        self.face._previous_emotion = emotion

    def _adjust_emotional_factors(self, emotion: str):
        """
        Ajuste les facteurs émotionnels du visage selon l'émotion.

        Args:
            emotion: Émotion à traiter
        """
        # Ajuster la respiration selon l'émotion
        if emotion in ["peur", "stress", "anxiété"]:
            self.face.breath_frequency = 1.5  # Plus rapide
            self.face.breath_amplitude = 0.025  # Plus ample
        elif emotion in ["tristesse", "mélancolie"]:
            self.face.breath_frequency = 0.6  # Plus lente
            self.face.breath_amplitude = 0.01  # Moins ample
        elif emotion in ["joie", "excitation"]:
            self.face.breath_frequency = 1.0  # Modérée
            self.face.breath_amplitude = 0.02  # Moyenne
        elif emotion in ["sérénité", "calme"]:
            self.face.breath_frequency = 0.5  # Très lente
            self.face.breath_amplitude = 0.015  # Moyenne
        else:
            # Valeurs par défaut
            self.face.breath_frequency = 0.8
            self.face.breath_amplitude = 0.015

    def _trigger_emotion_effects(self, emotion: str, intensity: float):
        """
        Déclenche les effets visuels appropriés pour une émotion.

        Args:
            emotion: Émotion à exprimer
            intensity: Intensité de l'émotion (0.0 à 1.0)
        """
        # Désactiver les effets actifs non pertinents
        self._deactivate_incompatible_effects(emotion)

        # Activer les effets spécifiques à l'émotion
        if emotion == "colère" and intensity > 0.6:
            if hasattr(self.face.effects, "start_emotional_vibration"):
                self.face.effects.start_emotional_vibration()

        elif emotion in ["peur", "stress", "anxiété"] and intensity > 0.5:
            if hasattr(self.face.effects, "start_muscle_tensions"):
                self.face.effects.start_muscle_tensions()

            # Déclencher une larme de peur si intensité élevée
            if intensity > 0.7 and hasattr(self.face.effects, "trigger_fast_tear"):
                self.face.effects.trigger_fast_tear()

        elif emotion in ["tristesse", "mélancolie"] and intensity > 0.6:
            if hasattr(self.face.effects, "start_mental_nebula"):
                self.face.effects.start_mental_nebula()

        elif emotion in ["joie", "émerveillement", "affection"] and intensity > 0.5:
            if hasattr(self.face.effects, "trigger_eye_sparkle"):
                self.face.effects.trigger_eye_sparkle(duration=intensity * 4.0, intensity=intensity)

        elif emotion in ["surprise", "choc"] and intensity > 0.6:
            if hasattr(self.face.effects, "apply_frisson"):
                self.face.effects.apply_frisson(intensity=intensity, duration=2.0)

        elif emotion in ["timidité", "embarras", "gêne"] and intensity > 0.3:
            if hasattr(self.face.effects, "add_blush"):
                self.face.effects.add_blush(intensity=intensity)

    def _deactivate_incompatible_effects(self, emotion: str):
        """
        Désactive les effets qui seraient incompatibles avec la nouvelle émotion.

        Args:
            emotion: Nouvelle émotion
        """
        # Groupes d'émotions qui partagent des effets similaires
        calm_emotions = ["sérénité", "calme", "neutre"]
        tense_emotions = ["colère", "peur", "stress", "anxiété"]
        sad_emotions = ["tristesse", "mélancolie", "vide"]
        happy_emotions = ["joie", "émerveillement", "affection", "amour"]

        # Désactiver les effets incompatibles
        if emotion in calm_emotions:
            # Désactiver les effets de tension et vibration
            if hasattr(self.face.effects, "stop_emotional_vibration"):
                self.face.effects.stop_emotional_vibration()
            if hasattr(self.face.effects, "stop_muscle_tensions"):
                self.face.effects.stop_muscle_tensions()

        if emotion not in sad_emotions:
            # Désactiver les effets de tristesse
            if hasattr(self.face.effects, "stop_mental_nebula"):
                self.face.effects.stop_mental_nebula()

    def _start_emotion_transition(self, from_emotion: str, to_emotion: str, duration: float = 1.0):
        """
        Démarre une transition entre deux émotions.

        Args:
            from_emotion: Émotion de départ
            to_emotion: Émotion d'arrivée
            duration: Durée de la transition en secondes
        """
        # Annuler toute transition en cours
        if self.current_transition:
            self.current_transition.cancel()

        # Démarrer avec un niveau de mélange élevé (l'ancienne émotion dominante)
        self.face.emotion_secondary = from_emotion
        self.face.emotion_blend = 0.8

        # Créer une animation pour la transition
        self.current_transition = Animation(emotion_blend=0.0, duration=duration)

        # Fonction de nettoyage à la fin de la transition
        def on_complete(*args):
            self.face.emotion_secondary = None
            self.face.emotion_blend = 0.0
            self.current_transition = None

        # Lancer l'animation avec le callback
        self.current_transition.bind(on_complete=on_complete)
        self.current_transition.start(self.face)

    def increase_pleasure_level(self, amount: float = 0.1, source: str = "caresse"):
        """
        Augmente le niveau de plaisir affectif.

        Args:
            amount: Quantité d'augmentation (0.0 à 1.0)
            source: Source du plaisir (caresse, compliment, etc.)
        """
        # Facteur de sensibilité selon le mode de développement
        sensitivity = 1.0
        if hasattr(self.face, "developmental_mode"):
            if self.face.developmental_mode == "enfant":
                sensitivity = 1.3  # Plus sensible en mode enfant
            elif self.face.developmental_mode == "adulte":
                sensitivity = 0.8  # Moins sensible en mode adulte

        # Appliquer l'augmentation avec le facteur de sensibilité
        adjusted_amount = amount * sensitivity

        # La progression devient plus difficile à des niveaux élevés
        if self.pleasure_level > 0.7:
            adjusted_amount *= 0.5

        # Appliquer l'augmentation (limitée à 1.0)
        self.pleasure_level = min(1.0, self.pleasure_level + adjusted_amount)

        # Activer le halo de plaisir si niveau suffisant
        if self.pleasure_level > 0.4 and hasattr(self.face.effects, "start_pleasure_halo"):
            self.face.effects.start_pleasure_halo()

        # Mémoriser le moment de l'affection
        self.last_affection_time = time.time()

        # Programmer la réduction progressive du plaisir si pas déjà fait
        if not self.pleasure_decay_timer:
            self.pleasure_decay_timer = Clock.schedule_interval(self._decay_pleasure_level, 3.0)

    def _decay_pleasure_level(self, dt):
        """
        Réduit progressivement le niveau de plaisir affectif.

        Args:
            dt: Delta temps depuis la dernière mise à jour
        """
        # Réduction lente et naturelle du plaisir affectif
        if self.pleasure_level > 0:
            # Réduction plus rapide pour les niveaux élevés, plus lente pour les faibles
            reduction = 0.02 + self.pleasure_level * 0.03
            self.pleasure_level = max(0, self.pleasure_level - reduction)

            # Désactiver le halo si le niveau devient trop faible
            if self.pleasure_level < 0.1 and hasattr(self.face.effects, "stop_pleasure_halo"):
                self.face.effects.stop_pleasure_halo()
        else:
            # Arrêter le timer si le niveau est à zéro
            Clock.unschedule(self.pleasure_decay_timer)
            self.pleasure_decay_timer = None

    def trigger_emotional_response(self, trigger_type: str, intensity: float = 0.7):
        """
        Déclenche une réponse émotionnelle à un événement externe.

        Args:
            trigger_type: Type de déclencheur émotionnel
            intensity: Intensité de la réponse (0.0 à 1.0)
        """
        # Mapper le déclencheur à une émotion et des effets
        response_mapping = {
            "compliment": (
                "joie",
                0.7,
                lambda: self.face.visual_renderer.trigger_immersive_scene("attention"),
            ),
            "critique": ("tristesse", 0.6, lambda: self.face.effects.add_eye_reflection(0.5)),
            "surprise": ("surprise", 0.8, lambda: self.face.effects.apply_frisson(0.8)),
            "peur": ("peur", 0.8, lambda: self.face.effects.trigger_fast_tear()),
            "colère": ("colère", 0.7, lambda: self.face.effects.start_emotional_vibration()),
            "affection": ("joie", 0.8, lambda: self.face.effects.trigger_eye_sparkle(4.0, 0.8)),
        }

        # Si le déclencheur est connu, appliquer la réponse
        if trigger_type in response_mapping:
            emotion, base_intensity, effect_func = response_mapping[trigger_type]

            # Appliquer l'émotion
            self.face.emotion = emotion
            self.face.intensity = base_intensity * intensity

            # Déclencher l'effet associé
            try:
                effect_func()
            except Exception as e:
                logging.warning(f"Erreur lors du déclenchement de l'effet: {e}")

            # Pour certains déclencheurs, ajouter du plaisir/déplaisir
            if trigger_type in ["compliment", "affection"]:
                self.increase_pleasure_level(amount=0.2 * intensity)

    def emotion_to_facial_expression(self, emotion: str, intensity: float = 0.5) -> dict[str, float]:
        """
        Convertit une émotion en paramètres d'expression faciale.

        Args:
            emotion: Émotion à exprimer
            intensity: Intensité de l'émotion (0.0 à 1.0)

        Returns:
            Dictionnaire de paramètres faciaux
        """
        # Paramètres de base (position neutre)
        expression = {
            "eyebrow_angle_left": 0,  # Angle sourcil gauche
            "eyebrow_angle_right": 0,  # Angle sourcil droit
            "eyebrow_curvature": 0,  # Courbure des sourcils
            "eyebrow_height": 0,  # Hauteur des sourcils
            "eye_openness": 1.0,  # Ouverture des yeux
            "eye_focus": 0,  # Regard (positif=haut, négatif=bas)
            "mouth_width": 1.0,  # Largeur de la bouche
            "mouth_height": 0.5,  # Hauteur/ouverture de la bouche
            "mouth_curl": 0,  # Courbure de la bouche (positif=sourire)
        }

        # Ajuster selon l'émotion
        if emotion == "joie":
            expression.update(
                {
                    "eyebrow_angle_left": 5,
                    "eyebrow_angle_right": -5,
                    "eyebrow_curvature": 5,
                    "eyebrow_height": 2,
                    "mouth_width": 1.2,
                    "mouth_height": 0.7,
                    "mouth_curl": 10,
                }
            )
        elif emotion == "tristesse":
            expression.update(
                {
                    "eyebrow_angle_left": -15,
                    "eyebrow_angle_right": 15,
                    "eyebrow_curvature": -5,
                    "eyebrow_height": -2,
                    "eye_openness": 0.8,
                    "eye_focus": -5,
                    "mouth_width": 0.8,
                    "mouth_height": 0.5,
                    "mouth_curl": -5,
                }
            )
        elif emotion == "colère":
            expression.update(
                {
                    "eyebrow_angle_left": -20,
                    "eyebrow_angle_right": 20,
                    "eyebrow_curvature": -8,
                    "eyebrow_height": -5,
                    "eye_openness": 0.9,
                    "mouth_width": 0.9,
                    "mouth_height": 0.7,
                    "mouth_curl": -8,
                }
            )
        elif emotion == "surprise":
            expression.update(
                {
                    "eyebrow_angle_left": 5,
                    "eyebrow_angle_right": -5,
                    "eyebrow_curvature": 10,
                    "eyebrow_height": 8,
                    "eye_openness": 1.2,
                    "mouth_width": 0.9,
                    "mouth_height": 1.0,
                    "mouth_curl": 0,
                }
            )
        elif emotion == "peur":
            expression.update(
                {
                    "eyebrow_angle_left": -10,
                    "eyebrow_angle_right": 10,
                    "eyebrow_curvature": 8,
                    "eyebrow_height": 5,
                    "eye_openness": 1.3,
                    "eye_focus": 3,
                    "mouth_width": 0.7,
                    "mouth_height": 0.8,
                    "mouth_curl": -3,
                }
            )

        # Moduler l'intensité de chaque paramètre
        for param in expression:
            if param != "eye_openness":  # On ne module pas l'ouverture des yeux
                expression[param] = expression[param] * intensity

        return expression
