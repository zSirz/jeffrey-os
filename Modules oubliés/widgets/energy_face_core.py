#!/usr/bin/env python
"""
energy_face_core.py - Fonctionnalités de base pour le visage de Jeffrey
Partie de la refactorisation du fichier energy_face.py d'origine (PACK 18)

Ce module contient les fonctionnalités de base du visage émotionnel :
- Initialisation
- Boucles de mise à jour principales
- Gestion canvas
- Respiration et affichage de base
"""

import math
import random
import time

from core.visual.visual_emotion_renderer import VisualEmotionRenderer
from kivy.clock import Clock
from kivy.graphics import Color, Ellipse
from kivy.properties import BooleanProperty, DictProperty, NumericProperty, StringProperty
from kivy.uix.widget import Widget

# Imports des modules refactorisés
from ui.face_drawer import FaceDrawer
from ui.face_effects import FaceEffects

# Imports des sous-modules refactorisés
from widgets.energy_face_emotions import EmotionHandler
from widgets.energy_face_memory import MemoryHandler
from widgets.energy_face_movements import MovementHandler
from widgets.energy_face_utils import UtilityFunctions


class EnergyFaceCoreWidget(Widget):
    """Classe de base pour le visage émotionnel de Jeffrey.

    Cette classe est le contrôleur principal qui coordonne :
    - FaceDrawer : responsable du dessin des éléments du visage
    - FaceEffects : responsable des effets émotionnels et animations
    - VisualEmotionRenderer : coordination des effets visuels liés aux émotions
    - Sous-modules spécialisés pour les émotions, mouvements, mémoire, etc.
    """

    # Propriétés principales
    emotion = StringProperty("neutral")
    intensity = NumericProperty(0.5)
    is_speaking = BooleanProperty(False)

    # Propriétés pour l'enrichissement émotionnel
    emotion_secondary = StringProperty(None)
    emotion_blend = NumericProperty(0.0)
    emotion_transition = NumericProperty(0.0)
    emotional_memory = DictProperty({})

    # Propriété pour la respiration
    scale = NumericProperty(1.0)

    # Propriété pour le contexte d'interaction
    context_mode = StringProperty("public")
    eyelid_openness = NumericProperty(1.0)  # Contrôle l'ouverture des paupières

    # PACK 8/9: Propriétés pour le lien affectif
    lien_affectif = NumericProperty(0.0)  # Niveau du lien affectif (0.0 à 1.0)
    etat_lien = StringProperty("stable")  # État du lien: stable, en_croissance, blesse, refroidi

    # PACK 9: Propriétés pour les souvenirs affectifs
    resonance_affective = NumericProperty(0.3)  # Niveau de résonance affective (0.0 à 1.0)
    blessure_active = BooleanProperty(False)  # Présence d'une blessure active

    def __init__(self, **kwargs):
        """Initialise le widget du visage et tous ses sous-composants."""
        super(EnergyFaceCoreWidget, self).__init__(**kwargs)

        # Initialiser les modules de dessin et d'effets
        self.drawer = FaceDrawer(self)
        self.effects = FaceEffects(self)

        # PACK 18: Initialiser le renderer d'émotions visuelles
        self.visual_renderer = VisualEmotionRenderer(self, self.effects)

        # Initialiser les sous-gestionnaires
        self.emotion_handler = EmotionHandler(self)
        self.movement_handler = MovementHandler(self)
        self.memory_handler = MemoryHandler(self)
        self.utils = UtilityFunctions(self)

        # Variables pour les interactions et états de base
        self.particles = []
        self.mouth_phase = 0
        self.halo_animation = 0

        # Variables pour la respiration émotionnelle
        self.breath_phase = 0.0
        self.breath_amplitude = 0.015
        self.breath_frequency = 0.8

        # Pour les micro-expressions et effets émotionnels
        self.touched_recently = False
        self.halo_feedback_timer = None

        # Variables pour l'animation labiale
        self.current_mouth_shape = "X"  # X = neutre/fermée
        self.lip_sync_events = []
        self.speaking_start_time = None

        # Connecter les événements et les propriétés
        self.bind(
            pos=self.update_canvas,
            size=self.update_canvas,
            emotion=self.on_emotion_change,
            intensity=self.update_canvas,
            is_speaking=self.on_speaking_changed,
        )

        # Planifier les mises à jour régulières
        Clock.schedule_interval(self.animate, 1 / 30.0)
        Clock.schedule_interval(self.effects.update_emotional_waves, 1 / 30.0)

        # Initialisation des fonctionnalités de beauté et de clignement
        self.enhance_beauty()
        self.setup_blinking()

    def update_canvas(self, *args):
        """Met à jour le canvas pour le dessin du visage."""
        # Réinitialiser le canvas
        self.canvas.clear()

        # Dessiner le visage via le drawer
        self.drawer.draw_face()

        # Afficher l'effet de toucher récent si actif
        if self.touched_recently:
            with self.canvas:
                Color(1, 0.8, 1, 0.1)
                Ellipse(pos=(self.center_x - 100, self.center_y - 100), size=(200, 200))

    def animate(self, dt):
        """
        Animation principale appelée à chaque frame.
        Coordonne toutes les animations de base du visage.

        Args:
            dt: Delta temps depuis la dernière mise à jour
        """
        # Mise à jour de la phase de respiration
        self.breath_phase += dt * self.breath_frequency

        # Calculer le facteur de respiration
        breath_value = math.sin(self.breath_phase)

        # Appliquer à l'échelle du visage
        self.scale = 1.0 + breath_value * self.breath_amplitude

        # Mise à jour de l'animation de la bouche si en train de parler
        if self.is_speaking:
            self.mouth_phase += dt * 15.0  # Vitesse de l'animation labiale

            # Alternance de formes de bouche durant la parole
            if self.speaking_start_time is None:
                self.speaking_start_time = time.time()

            # Calcul du temps écoulé depuis le début de la parole
            speaking_duration = time.time() - self.speaking_start_time

            # Traiter les événements d'animation labiale
            self._process_lip_sync_events(speaking_duration)

            # Si aucun événement programmé, utiliser une animation générique
            if not self.lip_sync_events:
                # Choisir périodiquement une nouvelle forme
                if random.random() < 0.1:  # 10% de chance à chaque frame
                    # Sélectionner une forme aléatoire adaptée à la parole
                    speaking_shapes = ["A", "E", "O", "I", "OU", "AN", "EN"]
                    self.current_mouth_shape = random.choice(speaking_shapes)
        else:
            self.speaking_start_time = None

            # Bouche fermée ou légèrement ouverte selon l'émotion
            if self.emotion == "surprise":
                self.current_mouth_shape = "O"
            elif self.emotion == "joie":
                self.current_mouth_shape = "AI"  # Sourire léger
            elif self.emotion == "tristesse":
                self.current_mouth_shape = "EN"  # Bouche tombante
            else:
                self.current_mouth_shape = "X"  # Bouche fermée neutre

        # Mettre à jour le canvas
        self.update_canvas()

    def on_speaking_changed(self, instance, value):
        """
        Appelé quand l'état de parole change.

        Args:
            instance: Instance qui a changé
            value: Nouvelle valeur (True/False pour parler/arrêter)
        """
        if value:  # Commence à parler
            # Réinitialiser les événements d'animation labiale
            self.lip_sync_events = []
            self.speaking_start_time = None
            # Activer l'animation de bouche
            self.animate_mouth(True)
        else:  # Arrête de parler
            # Désactiver l'animation de bouche
            self.animate_mouth(False)

    def animate_mouth(self, is_speaking: bool):
        """
        Contrôle l'animation de la bouche pendant la parole.

        Args:
            is_speaking: True si en train de parler, False sinon
        """
        if is_speaking:
            # Réinitialiser l'état de parole
            self.speaking_start_time = time.time()
            self.mouth_phase = 0
        else:
            # Revenir à une forme de bouche neutre
            self.current_mouth_shape = "X"

    def _process_lip_sync_events(self, speaking_duration):
        """
        Traite les événements programmés pour l'animation labiale synchronisée.

        Args:
            speaking_duration: Temps écoulé depuis le début de la parole
        """
        # Parcourir les événements et appliquer ceux qui sont actifs
        for event in list(self.lip_sync_events):
            start_time = event.get("start_time", 0)
            end_time = event.get("end_time", 0)
            shape = event.get("shape", "X")

            # Vérifier si l'événement est actif
            if start_time <= speaking_duration <= end_time:
                self.current_mouth_shape = shape
                return

            # Retirer les événements expirés
            if speaking_duration > end_time:
                self.lip_sync_events.remove(event)

    def on_emotion_change(self, instance, value):
        """
        Appelé quand l'émotion principale change.

        Args:
            instance: Instance qui a changé
            value: Nouvelle valeur de l'émotion
        """
        # Déléguer à l'handler d'émotions
        self.emotion_handler.process_emotion_change(value, self.intensity)

        # Utiliser le renderer visuel pour les effets visuels
        self.visual_renderer.render_emotion(
            emotion=value,
            intensity=self.intensity,
            secondary_emotion=self.emotion_secondary,
            blend=self.emotion_blend,
        )

    def enhance_beauty(self):
        """Initialise les couches de beauté visuelles du visage."""
        # Créer plusieurs couches de "beauté" avec des propriétés différentes
        self.beauty_layers = []

        # Ajouter 3 couches avec des phases, échelles et opacités différentes
        self.beauty_layers.append({"scale": 1.05, "opacity": 0.1, "phase_offset": 0.0, "speed": 0.2})

        self.beauty_layers.append({"scale": 1.12, "opacity": 0.07, "phase_offset": math.pi / 3, "speed": 0.15})

        self.beauty_layers.append({"scale": 1.18, "opacity": 0.05, "phase_offset": 2 * math.pi / 3, "speed": 0.1})

    def setup_blinking(self):
        """Initialise le système de clignement des yeux."""
        # Définir l'intervalle de base pour le clignement
        self.blink_interval = 5.0  # En secondes
        self.next_blink_time = time.time() + random.uniform(3.0, self.blink_interval)

        # Planifier la vérification régulière pour le clignement
        Clock.schedule_interval(self.check_for_blink, 0.1)

    def check_for_blink(self, dt):
        """
        Vérifie s'il est temps de clignoter des yeux.

        Args:
            dt: Delta temps depuis la dernière vérification
        """
        current_time = time.time()

        # S'il est temps de cligner
        if current_time >= self.next_blink_time:
            # Déclencher un clignement
            self.blink()

            # Calculer le prochain temps de clignement avec variation aléatoire
            variation = random.uniform(-1.0, 1.0)
            next_interval = self.blink_interval + variation
            self.next_blink_time = current_time + max(2.0, next_interval)

    def blink(self):
        """Effectue un clignement d'yeux."""
        # Sauvegarder l'ouverture actuelle des paupières
        original_openness = self.eyelid_openness

        # Animation de fermeture et réouverture
        def blink_down(dt):
            # Fermer rapidement
            self.eyelid_openness = max(0.1, self.eyelid_openness - 0.3)

            if self.eyelid_openness <= 0.1:
                # Programmer la réouverture
                Clock.schedule_once(blink_up, 0.05)
                return False
            return True

        def blink_up(dt):
            # Rouvrir progressivement
            self.eyelid_openness = min(original_openness, self.eyelid_openness + 0.15)

            if self.eyelid_openness >= original_openness:
                self.eyelid_openness = original_openness
                return False
            return True

        # Démarrer la séquence de clignement
        Clock.schedule_interval(blink_down, 1 / 30.0)
