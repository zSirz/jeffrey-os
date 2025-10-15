#!/usr/bin/env python

"""
immersive_emotion_face.py - Widget de visage immersif émotionnel pour Jeffrey

Ce module étend le visage émotionnel standard (EnergyFace) avec des fonctionnalités
immersives avancées qui permettent :
- Visualisation des émotions dominantes
- Micro-réactions et effets subtils (PACK 4, 5)
- Effets relationnels (PACK 11)
- Réactions sensorielles (PACK 20)
- Interface de pilotage dynamique pour emotional_learning et recommendation_engine

Le widget peut s'intégrer dans toute interface Kivy existante, en remplaçant
ou en complétant EnergyFaceWidget standard.
"""

import logging
import math
import random
import time

from core.emotions.emotional_engine import EmotionalEngine

# Import des systèmes liés aux PACKS fonctionnels
from core.emotions.emotional_learning import EmotionalLearning
from core.visual.visual_emotion_renderer import VisualEmotionRenderer
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.properties import (
    BooleanProperty,
    ListProperty,
    NumericProperty,
    ObjectProperty,
)

# Import des moteurs de rendu et d'effets
# Import des composants de base
from widgets.energy_face import EnergyFaceWidget


class ImmersiveEmotionFace(EnergyFaceWidget):
    """
    Widget de visage émotionnel immersif pour Jeffrey.

    Étend EnergyFaceWidget avec des fonctionnalités avancées:
    - Rendu immersif des émotions
    - Micro-animations et effets visuels améliorés
    - Transparence émotionnelle (visualisation des états internes)
    - Adaptation relationnelle (changements basés sur la relation)
    - Réactions sensorielles avancées
    """

    # Propriétés spécifiques à l'immersion
    immersion_level = NumericProperty(0.0)  # Niveau général d'immersion (0.0 à 1.0)
    environment_intensity = NumericProperty(0.5)  # Intensité des effets d'environnement
    emotion_transparency = NumericProperty(0.7)  # Niveau de visibilité des émotions internes
    micro_animation_intensity = NumericProperty(0.6)  # Intensité des micro-animations
    adaptive_style = BooleanProperty(True)  # Adaptation du style aux émotions

    # Propriétés pour les réactions sensorielles (PACK 20)
    sensory_sensitivity = NumericProperty(0.8)  # Sensibilité aux stimuli sensoriels
    sensory_memory_influence = NumericProperty(0.6)  # Influence de la mémoire sensorielle

    # Propriétés pour les relations (PACK 11)
    relationship_adaptation = NumericProperty(0.7)  # Adaptation aux relations
    relationship_visible = BooleanProperty(True)  # Visibilité des effets relationnels
    current_relationship_level = NumericProperty(0.0)  # Niveau de relation actuel

    # États émotionnels complexes
    emotional_blend = NumericProperty(0.0)  # Niveau de mélange entre émotions
    dominant_emotions = ListProperty([])  # Liste des émotions dominantes
    emotional_complexity = NumericProperty(0.5)  # Complexité émotionnelle (0.0 à 1.0)

    # Connecteur pour le moteur émotionnel
    emotional_engine = ObjectProperty(None)  # Moteur émotionnel
    emotional_learning = ObjectProperty(None)  # Système d'apprentissage émotionnel

    # Système de recommandation
    recommendation_engine = ObjectProperty(None)  # Moteur de recommandation

    def __init__(self, **kwargs):
        """
        Initialise le widget de visage immersif.

        Args:
            **kwargs: Arguments Kivy standards pour widgets
        """
        # Initialiser les propriétés avant l'appel parent
        self.effect_groups = {}
        self.animation_timers = {}
        self.micro_animations = {}
        self.active_effects = {}
        self.background_effects = {}
        self.emotional_indicators = {}
        self.relationship_effects = {}
        self.sensory_indicators = {}

        # Initialiser le logger
        self.logger = logging.getLogger("jeffrey.immersive_face")

        # Appeler l'initialisation parent (EnergyFaceWidget)
        super(ImmersiveEmotionFace, self).__init__(**kwargs)

        # Créer le renderer visuel immersif si pas déjà fourni
        if not hasattr(self, "visual_renderer") or self.visual_renderer is None:
            self.visual_renderer = VisualEmotionRenderer(self, self.effects)

        # Initialiser les composants d'immersion avancés
        self._init_immersive_components()

        # Configurer les observateurs de propriétés
        self._configure_property_observers()

        # Initialiser les effets de fond et les animations permanentes
        self._init_background_effects()

        # Initialiser les indicateurs émotionnels
        self._init_emotional_indicators()

        # Démarrer le cycle d'animation principal
        self._start_animation_cycle()

        # Journaliser l'initialisation
        self.logger.info("Widget de visage immersif initialisé")

    def _init_immersive_components(self):
        """
        Initialise les composants spécifiques à l'immersion.
        """
        # Créer le système d'émotions avancé si non fourni
        if self.emotional_engine is None:
            self.emotional_engine = EmotionalEngine(visual_renderer=self.visual_renderer)

        # Créer le système d'apprentissage si non fourni
        if self.emotional_learning is None:
            self.emotional_learning = EmotionalLearning()

        # Initialiser les gestionnaires d'effets par type
        self.effect_groups = {
            "base": {},  # Effets de base (tous les PACKS)
            "micro": {},  # Micro-animations (clignements, frissons)
            "relationship": {},  # Effets liés aux relations (PACK 11)
            "sensory": {},  # Effets sensoriels (PACK 20)
            "atmospheric": {},  # Effets d'ambiance (brume, particules)
            "indicators": {},  # Indicateurs visuels d'état interne
        }

        # Initialiser des hooks pour l'intégration externe
        self.hooks = {
            "before_emotion_update": [],  # Hooks avant mise à jour émotionnelle
            "after_emotion_update": [],  # Hooks après mise à jour émotionnelle
            "before_draw": [],  # Hooks avant dessin
            "after_draw": [],  # Hooks après dessin
        }

    def _configure_property_observers(self):
        """
        Configure les observateurs pour les propriétés dynamiques.
        """
        # Observer les changements d'émotion pour les effets immersifs
        self.bind(
            emotion=self._on_emotion_change_immersive,
            immersion_level=self._on_immersion_level_change,
            emotional_complexity=self._on_emotional_complexity_change,
            relationship_adaptation=self._on_relationship_adaptation_change,
        )

    def _init_background_effects(self):
        """
        Initialise les effets de fond et les animations permanentes.
        """
        # Effet de respiration subtile du visage entier
        breath_anim = self._create_breathing_animation()
        breath_anim.start(self)
        self.background_effects["breathing"] = breath_anim

        # Effet de halo ambiant subtil
        if hasattr(self.effects, "pulse_light"):
            self.effects.pulse_light(intensity=0.3, color=(0.95, 0.95, 1.0, 0.1))
            self.background_effects["ambient_halo"] = True

        # Effet de micro-mouvements aléatoires
        Clock.schedule_interval(self._update_micro_movements, 1 / 30)

    def _init_emotional_indicators(self):
        """
        Initialise les indicateurs visuels des états émotionnels internes.
        """
        # Créer des indicateurs pour les émotions dominantes
        self.emotional_indicators = {
            "dominant": self._create_dominant_emotion_indicator(),
            "secondary": self._create_secondary_emotion_indicator(),
            "complexity": self._create_emotional_complexity_indicator(),
        }

    def _start_animation_cycle(self):
        """
        Démarre le cycle principal d'animation et de mise à jour.
        """
        # Mise à jour des effets immersifs (30 FPS)
        Clock.schedule_interval(self._update_immersive_effects, 1 / 30)

        # Mise à jour des réactions émotionnelles (2 fois par seconde)
        Clock.schedule_interval(self._update_emotional_state, 0.5)

        # Mise à jour des effets relationnels (toutes les 2 secondes)
        Clock.schedule_interval(self._update_relationship_effects, 2.0)

        # Mise à jour des indicateurs émotionnels (4 fois par seconde)
        Clock.schedule_interval(self._update_emotional_indicators, 0.25)

        # Vérification des micro-réactions (15 FPS)
        Clock.schedule_interval(self._check_micro_reactions, 1 / 15)

    def _create_breathing_animation(self):
        """
        Crée une animation de respiration subtile pour le visage.

        Returns:
            Animation: Animation Kivy de respiration
        """
        # Animation de respiration en 2 temps (inspiration et expiration)
        breath_in = Animation(
            scale=1.02,  # Légère expansion
            breathing_phase=math.pi,  # Phase max
            d=2.0,  # Durée de 2 secondes
            t="out_sine",  # Courbe d'easing
        )

        breath_out = Animation(
            scale=0.98,  # Légère contraction
            breathing_phase=0,  # Phase min
            d=3.0,  # Durée de 3 secondes
            t="in_out_sine",  # Courbe d'easing
        )

        # Combiner les animations et les répéter infiniment
        breath_anim = breath_in + breath_out
        breath_anim.repeat = True

        return breath_anim

    def _create_dominant_emotion_indicator(self):
        """
        Crée un indicateur visuel pour l'émotion dominante.

        Returns:
            Dict: Configuration de l'indicateur
        """
        return {
            "active": True,
            "position": "top-right",
            "size": 30,
            "opacity": 0.7,
            "colors": {
                "joie": (1.0, 0.9, 0.3),
                "tristesse": (0.3, 0.5, 0.9),
                "colère": (0.9, 0.3, 0.3),
                "peur": (0.7, 0.3, 0.9),
                "surprise": (0.9, 0.6, 0.3),
                "dégoût": (0.4, 0.8, 0.4),
                "neutre": (0.7, 0.7, 0.7),
            },
            "last_update": time.time(),
        }

    def _create_secondary_emotion_indicator(self):
        """
        Crée un indicateur visuel pour l'émotion secondaire.

        Returns:
            Dict: Configuration de l'indicateur
        """
        return {
            "active": True,
            "position": "top-left",
            "size": 20,
            "opacity": 0.5,
            "last_update": time.time(),
        }

    def _create_emotional_complexity_indicator(self):
        """
        Crée un indicateur visuel pour la complexité émotionnelle.

        Returns:
            Dict: Configuration de l'indicateur
        """
        return {
            "active": True,
            "position": "bottom",
            "size": (100, 10),
            "opacity": 0.7,
            "last_update": time.time(),
        }

    def _on_emotion_change_immersive(self, instance, value):
        """
        Gère les changements d'émotion avec des effets immersifs.

        Args:
            instance: Instance qui a changé (self)
            value: Nouvelle valeur d'émotion
        """
        # Appeler d'abord le gestionnaire parent
        super(ImmersiveEmotionFace, self).on_emotion_change(instance, value)

        # Exécuter les hooks avant mise à jour
        self._execute_hooks("before_emotion_update", emotion=value)

        # Obtenir l'intensité actuelle
        intensity = getattr(self, "intensity", 0.5)

        # Appliquer les effets immersifs avancés
        self._apply_immersive_emotion_effects(value, intensity)

        # Mettre à jour les émotions dominantes
        self._update_dominant_emotions(value, intensity)

        # Mettre à jour l'apprentissage émotionnel si disponible
        if self.emotional_learning:
            self.emotional_learning.observe_emotion(value)

        # Adapter le style visuel si activé
        if self.adaptive_style:
            self._adapt_visual_style_to_emotion(value, intensity)

        # Exécuter les hooks après mise à jour
        self._execute_hooks("after_emotion_update", emotion=value, intensity=intensity)

    def _on_immersion_level_change(self, instance, value):
        """
        Ajuste les effets visuels selon le niveau d'immersion.

        Args:
            instance: Instance qui a changé (self)
            value: Nouveau niveau d'immersion
        """
        # Normaliser la valeur
        value = max(0.0, min(1.0, value))

        # Ajuster l'intensité des effets en fonction du niveau d'immersion
        self.environment_intensity = 0.3 + (value * 0.7)
        self.micro_animation_intensity = 0.4 + (value * 0.6)

        # Ajuster les propriétés visuelles d'immersion
        if hasattr(self.effects, "pulse_light"):
            self.effects.pulse_light(intensity=value * 0.4)

        # Journaliser le changement
        self.logger.debug(f"Niveau d'immersion ajusté à {value:.2f}")

    def _on_emotional_complexity_change(self, instance, value):
        """
        Ajuste le rendu visuel en fonction de la complexité émotionnelle.

        Args:
            instance: Instance qui a changé (self)
            value: Nouvelle complexité émotionnelle
        """
        # Normaliser la valeur
        value = max(0.0, min(1.0, value))

        # Ajuster la richesse des expressions faciales
        expression_richness = 0.5 + (value * 0.5)
        if hasattr(self, "expression_richness"):
            self.expression_richness = expression_richness

        # Mettre à jour l'indicateur de complexité
        if "complexity" in self.emotional_indicators:
            self.emotional_indicators["complexity"]["value"] = value
            self.emotional_indicators["complexity"]["last_update"] = time.time()

    def _on_relationship_adaptation_change(self, instance, value):
        """
        Ajuste les effets visuels selon le niveau d'adaptation relationnelle.

        Args:
            instance: Instance qui a changé (self)
            value: Nouveau niveau d'adaptation relationnelle
        """
        # Normaliser la valeur
        value = max(0.0, min(1.0, value))

        # Mettre à jour les effets de relation
        if value > 0.7 and hasattr(self.effects, "trigger_eye_sparkle"):
            # Effet d'étincelle dans les yeux pour relation forte
            self.effects.trigger_eye_sparkle(duration=2.0, intensity=value * 0.7)

        # Journaliser le changement
        self.logger.debug(f"Adaptation relationnelle ajustée à {value:.2f}")

    def _apply_immersive_emotion_effects(self, emotion, intensity):
        """
        Applique des effets visuels immersifs basés sur l'émotion.

        Args:
            emotion: Émotion à exprimer
            intensity: Intensité de l'émotion (0.0 à 1.0)
        """
        # Liste des effets à appliquer selon l'émotion
        emotion_effects = {
            "joie": [
                ("eye_sparkle", 3.0, min(1.0, intensity * 1.2)),
                ("pleasure_halo", 5.0, intensity),
                ("micro_smile", 2.0, intensity),
            ],
            "tristesse": [
                ("eye_reflection", 4.0, intensity),
                ("mental_nebula", 6.0, intensity * 0.8),
                ("slow_breathing", 10.0, intensity * 0.6),
            ],
            "colère": [
                ("vibration", 2.0, intensity),
                ("muscle_tensions", 3.0, intensity * 0.9),
                ("intense_gaze", 4.0, intensity),
            ],
            "peur": [
                ("wide_eyes", 2.5, intensity),
                ("fast_tear", 1.0, intensity * 0.8),
                ("trembling", 3.0, intensity),
            ],
            "surprise": [
                ("instant_widen", 1.0, intensity),
                ("frisson", 2.0, intensity * 0.7),
                ("quick_breath", 1.5, intensity),
            ],
            "dégoût": [
                ("facial_contraction", 2.0, intensity),
                ("nose_wrinkle", 2.5, intensity * 0.8),
            ],
            "neutre": [
                ("subtle_movement", 4.0, intensity * 0.3),
                ("soft_breathing", 8.0, intensity * 0.5),
            ],
            "curiosité": [
                ("eye_sparkle", 2.0, intensity * 0.6),
                ("head_tilt", 3.0, intensity * 0.7),
            ],
            "concentration": [
                ("focus_gaze", 3.0, intensity),
                ("reduced_blinking", 5.0, intensity * 0.8),
            ],
            "sérénité": [
                ("soft_glow", 6.0, intensity * 0.5),
                ("slow_breathing", 10.0, intensity * 0.4),
            ],
        }

        # Si l'émotion n'est pas dans notre mappage, utiliser neutre
        if emotion not in emotion_effects:
            emotion = "neutre"

        # Appliquer chaque effet pour cette émotion
        for effect_name, duration, effect_intensity in emotion_effects[emotion]:
            # Vérifier si l'effet est disponible
            if self._has_effect(effect_name):
                # Déclencher l'effet avec l'intensité ajustée
                self._trigger_effect(effect_name, duration, effect_intensity)

    def _update_dominant_emotions(self, current_emotion, intensity):
        """
        Met à jour la liste des émotions dominantes.

        Args:
            current_emotion: Émotion actuelle
            intensity: Intensité de l'émotion (0.0 à 1.0)
        """
        # Ne mettre à jour que si l'intensité est significative
        if intensity < 0.3:
            return

        # Créer une entrée pour l'émotion
        emotion_entry = {"name": current_emotion, "intensity": intensity, "timestamp": time.time()}

        # Ajouter à la liste des émotions dominantes
        dominant_list = self.dominant_emotions

        # Chercher si l'émotion existe déjà dans la liste
        exists = False
        for i, entry in enumerate(dominant_list):
            if entry.get("name") == current_emotion:
                # Mettre à jour l'entrée existante
                dominant_list[i] = emotion_entry
                exists = True
                break

        # Si l'émotion n'existe pas encore, l'ajouter
        if not exists:
            dominant_list.append(emotion_entry)

        # Limiter à 3 émotions dominantes max, triées par intensité
        if len(dominant_list) > 3:
            dominant_list = sorted(dominant_list, key=lambda x: x.get("intensity", 0.0), reverse=True)[:3]

        # Mettre à jour la propriété
        self.dominant_emotions = dominant_list

        # Calculer un score de complexité émotionnelle
        if len(dominant_list) > 1:
            top_intensities = [e.get("intensity", 0.0) for e in dominant_list[:2]]
            # Plus les émotions sont équilibrées, plus la complexité est élevée
            difference = abs(top_intensities[0] - top_intensities[1])
            self.emotional_complexity = 1.0 - difference
        else:
            self.emotional_complexity = 0.3  # Complexité basse si une seule émotion

    def _adapt_visual_style_to_emotion(self, emotion, intensity):
        """
        Adapte le style visuel du visage en fonction de l'émotion.

        Args:
            emotion: Émotion actuelle
            intensity: Intensité de l'émotion (0.0 à 1.0)
        """
        # Mappings des émotions vers des styles visuels
        style_mappings = {
            "joie": {"palette": "warm", "lines": "flowing", "animation": "bouncy"},
            "tristesse": {"palette": "cool", "lines": "soft", "animation": "slow"},
            "colère": {"palette": "hot", "lines": "sharp", "animation": "abrupt"},
            "peur": {"palette": "cold", "lines": "vibrating", "animation": "jerky"},
            "surprise": {"palette": "bright", "lines": "expanding", "animation": "quick"},
            "dégoût": {"palette": "muted", "lines": "contracted", "animation": "tense"},
            "neutre": {"palette": "neutral", "lines": "standard", "animation": "regular"},
            "curiosité": {"palette": "vibrant", "lines": "inquisitive", "animation": "attentive"},
            "sérénité": {"palette": "peaceful", "lines": "flowing", "animation": "gentle"},
        }

        # Si l'émotion n'est pas dans notre mappage, utiliser neutre
        if emotion not in style_mappings:
            emotion = "neutre"

        # Récupérer le style pour cette émotion
        style = style_mappings[emotion]

        # Appliquer le style visuel si le drawer le supporte
        if hasattr(self, "drawer") and hasattr(self.drawer, "set_visual_style"):
            # L'intensité module le niveau d'application du style
            style_intensity = min(1.0, intensity * 1.2)
            self.drawer.set_visual_style(style, style_intensity)

    def _update_immersive_effects(self, dt):
        """
        Met à jour les effets d'immersion à chaque frame.

        Args:
            dt: Delta temps depuis la dernière mise à jour
        """
        # Phase actuelle (utilisée pour les animations synchronisées)
        current_time = time.time()

        # Mettre à jour les effets actifs
        for effect_type, effects in self.effect_groups.items():
            for effect_name, effect_data in list(effects.items()):
                # Vérifier si l'effet est toujours actif
                if current_time > effect_data.get("end_time", 0):
                    # Effet terminé, le retirer
                    effects.pop(effect_name, None)
                    continue

                # Calculer la progression de l'effet (0.0 à 1.0)
                start_time = effect_data.get("start_time", current_time)
                end_time = effect_data.get("end_time", current_time)
                duration = end_time - start_time
                elapsed = current_time - start_time

                if duration > 0:
                    progress = elapsed / duration
                    # Mettre à jour les données de l'effet
                    effect_data["progress"] = min(1.0, max(0.0, progress))
                    effect_data["remaining"] = max(0.0, duration - elapsed)

                    # Appeler la fonction de mise à jour spécifique à cet effet
                    update_func = effect_data.get("update_func")
                    if update_func and callable(update_func):
                        update_func(effect_data, dt)

        # Forcer le rafraîchissement du canvas
        self.canvas.ask_update()

    def _update_emotional_state(self, dt):
        """
        Met à jour périodiquement l'état émotionnel global.

        Args:
            dt: Delta temps depuis la dernière mise à jour
        """
        # Si le moteur émotionnel est disponible, utiliser son état
        if self.emotional_engine:
            # Récupérer l'émotion dominante du moteur
            dominant_emotion = self.emotional_engine.get_dominant_emotion()
            if dominant_emotion:
                # Obtenir la distribution des émotions pour la complexité
                distribution = self.emotional_engine.get_emotion_distribution()
                if len(distribution) > 1:
                    # Calculer la complexité émotionnelle à partir de la distribution
                    values = list(distribution.values())
                    total = sum(values)
                    if total > 0:
                        values_normalized = [v / total for v in values]
                        # Plus les émotions sont équitablement réparties, plus c'est complexe
                        entropy = -sum(p * math.log(p) if p > 0 else 0 for p in values_normalized)
                        max_entropy = math.log(len(values))
                        if max_entropy > 0:
                            self.emotional_complexity = min(1.0, entropy / max_entropy)

        # Si l'apprentissage émotionnel est disponible, utiliser son profil
        if self.emotional_learning:
            # Récupérer le profil émotionnel actuel
            profile = self.emotional_learning.get_profile()
            dominant_emotions = profile.get("dominant_emotions", [])

            # Mettre à jour la liste des émotions dominantes si disponible
            if dominant_emotions:
                new_dominants = []
                for emotion in dominant_emotions[:3]:
                    # Créer une entrée pour chaque émotion dominante
                    emotion_entry = {
                        "name": emotion,
                        "intensity": 0.8,  # Valeur par défaut, on pourrait affiner
                        "timestamp": time.time(),
                    }
                    new_dominants.append(emotion_entry)

                self.dominant_emotions = new_dominants

    def _update_relationship_effects(self, dt):
        """
        Met à jour les effets visuels liés aux relations.

        Args:
            dt: Delta temps depuis la dernière mise à jour
        """
        # Ne mettre à jour que si les effets relationnels sont visibles
        if not self.relationship_visible:
            return

        # Récupérer le niveau de relation actuel (via un gestionnaire externe)
        level = self.current_relationship_level

        # Appliquer des effets visuels en fonction du niveau de relation
        if level > 0.8:
            # Relation très forte: effets intenses
            if hasattr(self.effects, "trigger_eye_sparkle"):
                if random.random() < 0.3:  # 30% de chance à chaque mise à jour
                    self.effects.trigger_eye_sparkle(duration=2.0, intensity=0.8)

            # Halo de relation forte (subtil)
            if hasattr(self.effects, "start_pleasure_halo"):
                self.effects.start_pleasure_halo()

        elif level > 0.6:
            # Relation forte: effets modérés
            if hasattr(self.effects, "add_eye_reflection"):
                if random.random() < 0.2:  # 20% de chance
                    self.effects.add_eye_reflection(0.6)

        elif level > 0.4:
            # Relation moyenne: effets légers
            if hasattr(self.effects, "pulse_light"):
                if random.random() < 0.1:  # 10% de chance
                    self.effects.pulse_light(intensity=0.3)

    def _update_emotional_indicators(self, dt):
        """
        Met à jour les indicateurs visuels des états émotionnels.

        Args:
            dt: Delta temps depuis la dernière mise à jour
        """
        # Mettre à jour l'indicateur d'émotion dominante
        if self.dominant_emotions and "dominant" in self.emotional_indicators:
            indicator = self.emotional_indicators["dominant"]
            if indicator["active"] and self.emotion_transparency > 0.1:
                # Utiliser la première émotion dominante
                top_emotion = self.dominant_emotions[0].get("name", "neutre")
                indicator["current_emotion"] = top_emotion
                indicator["opacity"] = self.emotion_transparency
                indicator["last_update"] = time.time()

        # Mettre à jour l'indicateur d'émotion secondaire
        if len(self.dominant_emotions) > 1 and "secondary" in self.emotional_indicators:
            indicator = self.emotional_indicators["secondary"]
            if indicator["active"] and self.emotion_transparency > 0.1:
                # Utiliser la deuxième émotion dominante
                second_emotion = self.dominant_emotions[1].get("name", "neutre")
                indicator["current_emotion"] = second_emotion
                indicator["opacity"] = self.emotion_transparency * 0.8  # Légèrement plus transparent
                indicator["last_update"] = time.time()

    def _check_micro_reactions(self, dt):
        """
        Vérifie et déclenche des micro-réactions émotionnelles aléatoires.

        Args:
            dt: Delta temps depuis la dernière mise à jour
        """
        # Probabilité de micro-réaction dépend de l'intensité
        base_probability = 0.02  # 2% de chance par défaut
        intensity_factor = self.micro_animation_intensity
        current_emotion = getattr(self, "emotion", "neutre")
        current_intensity = getattr(self, "intensity", 0.5)

        # Calculer la probabilité finale
        probability = base_probability * intensity_factor * current_intensity

        # Liste des micro-réactions possibles selon l'émotion
        micro_reactions = {
            "joie": ["micro_smile", "eye_twinkle", "cheek_lift"],
            "tristesse": ["slight_frown", "subtle_eye_close", "lip_quiver"],
            "colère": ["brow_twitch", "jaw_tense", "nostril_flare"],
            "peur": ["quick_eye_dart", "subtle_tremble", "quick_breath"],
            "surprise": ["micro_gasp", "fast_blink", "slight_head_back"],
            "dégoût": ["nose_wrinkle", "subtle_lip_curl", "slight_head_turn"],
            "neutre": ["normal_blink", "subtle_head_tilt", "ambient_eye_movement"],
        }

        # Obtenir les réactions pour l'émotion actuelle ou neutre
        reactions = micro_reactions.get(current_emotion, micro_reactions["neutre"])

        # Vérifier si une micro-réaction se produit
        if random.random() < probability:
            # Choisir une réaction aléatoire
            reaction = random.choice(reactions)

            # Appliquer la micro-réaction
            self._apply_micro_reaction(reaction, current_intensity)

    def _apply_micro_reaction(self, reaction_type, intensity):
        """
        Applique une micro-réaction spécifique.

        Args:
            reaction_type: Type de micro-réaction
            intensity: Intensité à appliquer
        """
        # Mapper les types de réaction aux méthodes d'application
        reaction_map = {
            "normal_blink": self._apply_normal_blink,
            "micro_smile": self._apply_micro_smile,
            "eye_twinkle": self._apply_eye_twinkle,
            "cheek_lift": self._apply_cheek_lift,
            "slight_frown": self._apply_slight_frown,
            "subtle_eye_close": self._apply_subtle_eye_close,
            "lip_quiver": self._apply_lip_quiver,
            "brow_twitch": self._apply_brow_twitch,
            "quick_eye_dart": self._apply_quick_eye_dart,
            "subtle_head_tilt": self._apply_subtle_head_tilt,
            "nose_wrinkle": self._apply_nose_wrinkle,
        }

        # Appeler la méthode correspondante si disponible
        if reaction_type in reaction_map:
            reaction_func = reaction_map[reaction_type]
            # Appliquer avec une intensité ajustée (80-120% de l'intensité de base)
            adjusted_intensity = intensity * random.uniform(0.8, 1.2)
            reaction_func(adjusted_intensity)

    # Implémentation des micro-réactions spécifiques

    def _apply_normal_blink(self, intensity):
        """Applique un clignement d'yeux normal."""
        if hasattr(self, "eyelid_openness"):
            # Sauvegarder l'ouverture actuelle
            original_openness = self.eyelid_openness

            # Créer une animation de clignement
            blink_down = Animation(eyelid_openness=0.1, d=0.1)
            blink_up = Animation(eyelid_openness=original_openness, d=0.15)
            blink_anim = blink_down + blink_up

            # Lancer l'animation
            blink_anim.start(self)

    def _apply_micro_smile(self, intensity):
        """Applique un micro-sourire."""
        if hasattr(self, "mouth_curl"):
            # Sauvegarder la courbure actuelle
            original_curl = self.mouth_curl

            # Créer une animation de micro-sourire
            smile_up = Animation(mouth_curl=original_curl + 5 * intensity, d=0.3)
            smile_down = Animation(mouth_curl=original_curl, d=0.5)
            smile_anim = smile_up + smile_down

            # Lancer l'animation
            smile_anim.start(self)

    def _apply_eye_twinkle(self, intensity):
        """Applique un pétillement des yeux."""
        if hasattr(self.effects, "trigger_eye_sparkle"):
            self.effects.trigger_eye_sparkle(duration=0.5, intensity=intensity)

    def _apply_cheek_lift(self, intensity):
        """Applique un léger soulèvement des joues."""
        if hasattr(self.effects, "add_blush"):
            self.effects.add_blush(intensity=intensity * 0.3)

    def _apply_slight_frown(self, intensity):
        """Applique un léger froncement de sourcils."""
        if hasattr(self, "eyebrow_angle_left") and hasattr(self, "eyebrow_angle_right"):
            # Sauvegarder les angles actuels
            original_left = self.eyebrow_angle_left
            original_right = self.eyebrow_angle_right

            # Créer une animation de froncement
            frown_anim = Animation(
                eyebrow_angle_left=original_left - 5 * intensity,
                eyebrow_angle_right=original_right + 5 * intensity,
                d=0.3,
            ) + Animation(eyebrow_angle_left=original_left, eyebrow_angle_right=original_right, d=0.5)

            # Lancer l'animation
            frown_anim.start(self)

    def _apply_subtle_eye_close(self, intensity):
        """Applique une légère fermeture des yeux."""
        if hasattr(self, "eyelid_openness"):
            # Sauvegarder l'ouverture actuelle
            original_openness = self.eyelid_openness

            # Créer une animation de légère fermeture
            close_anim = Animation(eyelid_openness=original_openness * (1 - intensity * 0.3), d=0.4) + Animation(
                eyelid_openness=original_openness, d=0.6
            )

            # Lancer l'animation
            close_anim.start(self)

    def _apply_lip_quiver(self, intensity):
        """Applique un léger tremblement des lèvres."""
        if hasattr(self, "mouth_width"):
            # Sauvegarder les propriétés actuelles
            original_width = self.mouth_width

            # Créer une séquence de micro-mouvements
            quiver_anim = Animation(mouth_width=original_width * 0.95, d=0.1)
            quiver_anim += Animation(mouth_width=original_width * 1.05, d=0.1)
            quiver_anim += Animation(mouth_width=original_width * 0.97, d=0.1)
            quiver_anim += Animation(mouth_width=original_width, d=0.2)

            # Lancer l'animation
            quiver_anim.start(self)

    def _apply_brow_twitch(self, intensity):
        """Applique un tic du sourcil."""
        if hasattr(self, "eyebrow_height"):
            # Sauvegarder la hauteur actuelle
            original_height = self.eyebrow_height

            # Créer une animation de tic
            twitch_anim = Animation(eyebrow_height=original_height + 3 * intensity, d=0.1)
            twitch_anim += Animation(eyebrow_height=original_height, d=0.2)

            # Lancer l'animation
            twitch_anim.start(self)

    def _apply_quick_eye_dart(self, intensity):
        """Applique un mouvement rapide des yeux."""
        if hasattr(self, "eye_position_x") and hasattr(self, "eye_position_y"):
            # Sauvegarder les positions actuelles
            original_x = self.eye_position_x
            original_y = self.eye_position_y

            # Direction aléatoire
            direction_x = random.uniform(-1, 1)
            direction_y = random.uniform(-1, 1)

            # Créer une animation de mouvement rapide
            dart_anim = Animation(
                eye_position_x=original_x + direction_x * 5 * intensity,
                eye_position_y=original_y + direction_y * 5 * intensity,
                d=0.15,
            ) + Animation(eye_position_x=original_x, eye_position_y=original_y, d=0.25)

            # Lancer l'animation
            dart_anim.start(self)

    def _apply_subtle_head_tilt(self, intensity):
        """Applique une légère inclinaison de la tête."""
        if hasattr(self, "head_rotation"):
            # Sauvegarder la rotation actuelle
            original_rotation = self.head_rotation

            # Direction aléatoire
            direction = 1 if random.random() > 0.5 else -1

            # Créer une animation d'inclinaison
            tilt_anim = Animation(head_rotation=original_rotation + direction * 3 * intensity, d=0.4) + Animation(
                head_rotation=original_rotation, d=0.6
            )

            # Lancer l'animation
            tilt_anim.start(self)

    def _apply_nose_wrinkle(self, intensity):
        """Applique un plissement du nez."""
        if hasattr(self, "nose_wrinkle"):
            # Sauvegarder le plissement actuel
            original_wrinkle = self.nose_wrinkle

            # Créer une animation de plissement
            wrinkle_anim = Animation(nose_wrinkle=original_wrinkle + 0.8 * intensity, d=0.3) + Animation(
                nose_wrinkle=original_wrinkle, d=0.5
            )

            # Lancer l'animation
            wrinkle_anim.start(self)

    def _update_micro_movements(self, dt):
        """
        Met à jour les micro-mouvements aléatoires.

        Args:
            dt: Delta temps depuis la dernière mise à jour
        """
        # Ne faire que si l'intensité est suffisante
        if self.micro_animation_intensity < 0.2:
            return

        # Appliquer de petites variations aléatoires à certaines propriétés
        for prop_name, params in self.micro_animations.items():
            # Vérifier si la propriété existe
            if not hasattr(self, prop_name):
                continue

            # Récupérer la valeur actuelle
            current_value = getattr(self, prop_name)

            # Calculer une nouvelle valeur avec variation aléatoire
            base_value = params.get("base_value", current_value)
            amplitude = params.get("amplitude", 0.05)
            speed = params.get("speed", 1.0)

            # Variation sinusoïdale avec composante aléatoire
            time_factor = time.time() * speed
            random_factor = random.uniform(-0.2, 0.2)
            variation = amplitude * math.sin(time_factor) + random_factor * amplitude * 0.5

            # Appliquer la nouvelle valeur
            new_value = base_value + variation * self.micro_animation_intensity
            setattr(self, prop_name, new_value)

    def _has_effect(self, effect_name):
        """
        Vérifie si un effet est disponible.

        Args:
            effect_name: Nom de l'effet à vérifier

        Returns:
            bool: True si l'effet est disponible, False sinon
        """
        # Vérifier en priorité dans les effets propres
        if hasattr(self, f"start_{effect_name}"):
            return True

        # Vérifier ensuite dans le moteur d'effets
        if hasattr(self.effects, f"start_{effect_name}") or hasattr(self.effects, f"trigger_{effect_name}"):
            return True

        # Si c'est une micro-réaction, vérifier dans les mappings
        if effect_name in [
            "micro_smile",
            "eye_twinkle",
            "cheek_lift",
            "normal_blink",
            "slight_frown",
            "subtle_eye_close",
            "brow_twitch",
            "quick_eye_dart",
        ]:
            return True

        return False

    def _trigger_effect(self, effect_name, duration=2.0, intensity=0.5):
        """
        Déclenche un effet visuel avec durée et intensité.

        Args:
            effect_name: Nom de l'effet à déclencher
            duration: Durée en secondes
            intensity: Intensité (0.0 à 1.0)
        """
        # Récupérer les méthodes de démarrage/arrêt appropriées
        start_method = None
        stop_method = None

        # Vérifier d'abord les méthodes propres à cette classe
        if hasattr(self, f"start_{effect_name}"):
            start_method = getattr(self, f"start_{effect_name}")
        elif hasattr(self, f"trigger_{effect_name}"):
            start_method = getattr(self, f"trigger_{effect_name}")

        # Sinon, vérifier dans le moteur d'effets
        elif hasattr(self.effects, f"start_{effect_name}"):
            start_method = getattr(self.effects, f"start_{effect_name}")
        elif hasattr(self.effects, f"trigger_{effect_name}"):
            start_method = getattr(self.effects, f"trigger_{effect_name}")

        # Chercher aussi la méthode d'arrêt
        if hasattr(self, f"stop_{effect_name}"):
            stop_method = getattr(self, f"stop_{effect_name}")
        elif hasattr(self.effects, f"stop_{effect_name}"):
            stop_method = getattr(self.effects, f"stop_{effect_name}")

        # Si c'est une micro-réaction, utiliser la méthode appropriée
        if effect_name.startswith("micro_") or effect_name in [
            "normal_blink",
            "slight_frown",
            "subtle_eye_close",
            "eye_twinkle",
        ]:
            method_name = f"_apply_{effect_name}"
            if hasattr(self, method_name):
                reaction_method = getattr(self, method_name)
                reaction_method(intensity)
                return

        # Appliquer l'effet si la méthode existe
        if start_method and callable(start_method):
            # Appeler avec les bons paramètres selon la signature
            try:
                # Essayer d'abord avec intensité et durée
                if effect_name in ["eye_sparkle", "pleasure_halo", "soft_glow"]:
                    start_method(duration=duration, intensity=intensity)
                else:
                    start_method(intensity=intensity)
            except TypeError:
                # Si ça échoue, essayer sans paramètres
                try:
                    start_method()
                except Exception as e:
                    self.logger.warning(f"Erreur lors du déclenchement de l'effet {effect_name}: {e}")
                    return

            # Programmer l'arrêt de l'effet si une méthode d'arrêt existe
            if stop_method and callable(stop_method):
                Clock.schedule_once(lambda dt: stop_method(), duration)

            # Journaliser le déclenchement
            self.logger.debug(f"Effet {effect_name} déclenché (durée: {duration}s, intensité: {intensity:.2f})")

    def _execute_hooks(self, hook_name, **kwargs):
        """
        Exécute les hooks enregistrés pour un événement spécifique.

        Args:
            hook_name: Nom du hook à exécuter
            **kwargs: Arguments à passer aux fonctions hook
        """
        if hook_name in self.hooks:
            for hook_func in self.hooks[hook_name]:
                try:
                    hook_func(self, **kwargs)
                except Exception as e:
                    self.logger.warning(f"Erreur lors de l'exécution du hook {hook_name}: {e}")

    def register_hook(self, hook_name, hook_func):
        """
        Enregistre une fonction hook pour un événement spécifique.

        Args:
            hook_name: Nom du hook
            hook_func: Fonction à appeler
        """
        if hook_name in self.hooks:
            self.hooks[hook_name].append(hook_func)

    def unregister_hook(self, hook_name, hook_func):
        """
        Supprime une fonction hook.

        Args:
            hook_name: Nom du hook
            hook_func: Fonction à supprimer
        """
        if hook_name in self.hooks and hook_func in self.hooks[hook_name]:
            self.hooks[hook_name].remove(hook_func)

    def connect_to_emotional_engine(self, engine):
        """
        Connecte le widget au moteur émotionnel fourni.

        Args:
            engine: Instance de EmotionalEngine
        """
        self.emotional_engine = engine

        # Connecter le renderer visuel
        if hasattr(engine, "visual_renderer") and engine.visual_renderer is None:
            engine.visual_renderer = self.visual_renderer

        # Journaliser la connexion
        self.logger.info(f"Widget connecté au moteur émotionnel: {engine}")

    def connect_to_learning_system(self, learning_system):
        """
        Connecte le widget au système d'apprentissage émotionnel.

        Args:
            learning_system: Instance de EmotionalLearning
        """
        self.emotional_learning = learning_system
        self.logger.info(f"Widget connecté au système d'apprentissage: {learning_system}")

    def connect_to_recommendation_engine(self, engine):
        """
        Connecte le widget au moteur de recommandation.

        Args:
            engine: Instance du moteur de recommandation
        """
        self.recommendation_engine = engine
        self.logger.info(f"Widget connecté au moteur de recommandation: {engine}")

        # Essayer d'extraire certaines recommandations si disponibles
        if hasattr(engine, "get_visual_recommendations"):
            recommendations = engine.get_visual_recommendations()
            if recommendations:
                # Appliquer les recommandations visuelles
                self._apply_visual_recommendations(recommendations)

    def _apply_visual_recommendations(self, recommendations):
        """
        Applique les recommandations visuelles fournies.

        Args:
            recommendations: Dictionnaire de recommandations
        """
        # Appliquer les niveaux d'immersion et de transparence
        if "immersion_level" in recommendations:
            self.immersion_level = recommendations["immersion_level"]

        if "emotion_transparency" in recommendations:
            self.emotion_transparency = recommendations["emotion_transparency"]

        # Appliquer les recommandations de style ou d'effets spécifiques
        if "highlight_emotions" in recommendations and recommendations["highlight_emotions"]:
            # Activer les indicateurs émotionnels
            for indicator in self.emotional_indicators.values():
                indicator["active"] = True
                indicator["opacity"] = max(0.7, indicator.get("opacity", 0.7))

        # Journaliser l'application
        self.logger.debug(f"Recommandations visuelles appliquées: {recommendations}")

    # Méthodes publiques pour le contrôle externe

    def set_relationship_level(self, level, person_id="unknown"):
        """
        Définit le niveau de relation actuel pour l'affichage.

        Args:
            level: Niveau de relation (0.0 à 1.0)
            person_id: Identifiant de la personne concernée
        """
        # Normaliser le niveau
        level = max(0.0, min(1.0, level))

        # Mettre à jour le niveau stocké
        self.current_relationship_level = level

        # Ajuster l'adaptation relationnelle en fonction du niveau
        self.relationship_adaptation = 0.3 + (level * 0.7)

        # Activer les effets de relation si le niveau est significatif
        if level > 0.6:
            self.relationship_visible = True

        # Journaliser le changement
        self.logger.info(f"Niveau de relation défini à {level:.2f} pour {person_id}")

    def add_emotion_to_display(self, emotion, intensity=0.5, as_secondary=False):
        """
        Ajoute une émotion à l'affichage sans changer l'émotion principale.

        Args:
            emotion: Émotion à ajouter
            intensity: Intensité de l'émotion (0.0 à 1.0)
            as_secondary: Si True, ajoute comme émotion secondaire
        """
        # Si c'est une émotion secondaire, l'ajouter comme telle
        if as_secondary:
            self.emotion_secondary = emotion
            self.emotion_blend = intensity
            return

        # Sinon, l'ajouter aux émotions dominantes
        emotion_entry = {"name": emotion, "intensity": intensity, "timestamp": time.time()}

        # Vérifier si l'émotion existe déjà
        dominants = self.dominant_emotions
        for i, entry in enumerate(dominants):
            if entry.get("name") == emotion:
                # Mettre à jour l'entrée existante
                dominants[i] = emotion_entry
                self.dominant_emotions = dominants
                return

        # Ajouter la nouvelle émotion
        dominants.append(emotion_entry)

        # Trier par intensité décroissante et limiter à 3
        dominants = sorted(dominants, key=lambda x: x.get("intensity", 0.0), reverse=True)[:3]

        self.dominant_emotions = dominants

        # Recalculer la complexité émotionnelle
        if len(dominants) > 1:
            top_intensities = [e.get("intensity", 0.0) for e in dominants[:2]]
            difference = abs(top_intensities[0] - top_intensities[1])
            self.emotional_complexity = 1.0 - difference

    def set_immersion_mode(self, mode="balanced"):
        """
        Définit le mode d'immersion prédéfini.

        Args:
            mode: Mode d'immersion ("minimal", "balanced", "full", "adaptive")
        """
        if mode == "minimal":
            self.immersion_level = 0.3
            self.emotion_transparency = 0.4
            self.micro_animation_intensity = 0.3
            self.emotional_complexity = 0.3

        elif mode == "balanced":
            self.immersion_level = 0.6
            self.emotion_transparency = 0.7
            self.micro_animation_intensity = 0.6
            self.emotional_complexity = 0.5

        elif mode == "full":
            self.immersion_level = 0.9
            self.emotion_transparency = 0.9
            self.micro_animation_intensity = 0.8
            self.emotional_complexity = 0.7

        elif mode == "adaptive":
            # Mode qui s'adapte aux émotions
            self.adaptive_style = True
            # Niveau initial équilibré
            self.immersion_level = 0.7
            self.emotion_transparency = 0.8
            self.micro_animation_intensity = 0.7

        # Journaliser le changement
        self.logger.info(f"Mode d'immersion défini: {mode}")

    def trigger_immersive_scene(self, scene_type, intensity=1.0, duration=5.0):
        """
        Déclenche une scène immersive complète.

        Args:
            scene_type: Type de scène ("emotional", "relationship", "memory", "focus")
            intensity: Intensité de la scène (0.0 à 1.0)
            duration: Durée en secondes
        """
        # Déléguer au renderer visuel
        if self.visual_renderer:
            self.visual_renderer.trigger_immersive_scene(scene_type, intensity)

        # Augmenter temporairement le niveau d'immersion
        original_immersion = self.immersion_level
        self.immersion_level = min(1.0, original_immersion + 0.3)

        # Restaurer après la durée
        def restore_immersion(dt):
            self.immersion_level = original_immersion

        Clock.schedule_once(restore_immersion, duration)

        # Journaliser le déclenchement
        self.logger.info(f"Scène immersive déclenchée: {scene_type} (int: {intensity:.2f}, durée: {duration:.1f}s)")
