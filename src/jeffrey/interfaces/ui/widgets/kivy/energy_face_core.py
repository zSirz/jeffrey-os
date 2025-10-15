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

from jeffrey.core.visual.visual_emotion_renderer import VisualEmotionRenderer
from jeffrey.interfaces.ui.face_drawer import FaceDrawer
from jeffrey.interfaces.ui.face_effects import FaceEffects
from kivy.clock import Clock
from kivy.graphics import Color, Ellipse
from kivy.properties import BooleanProperty, DictProperty, NumericProperty, StringProperty
from kivy.uix.widget import Widget
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

    emotion = StringProperty('neutral')
    intensity = NumericProperty(0.5)
    is_speaking = BooleanProperty(False)
    emotion_secondary = StringProperty(None)
    emotion_blend = NumericProperty(0.0)
    emotion_transition = NumericProperty(0.0)
    emotional_memory = DictProperty({})
    scale = NumericProperty(1.0)
    context_mode = StringProperty('public')
    eyelid_openness = NumericProperty(1.0)
    lien_affectif = NumericProperty(0.0)
    etat_lien = StringProperty('stable')
    resonance_affective = NumericProperty(0.3)
    blessure_active = BooleanProperty(False)

    def __init__(self, **kwargs):
        """Initialise le widget du visage et tous ses sous-composants."""
        super(EnergyFaceCoreWidget, self).__init__(**kwargs)
        self.drawer = FaceDrawer(self)
        self.effects = FaceEffects(self)
        self.visual_renderer = VisualEmotionRenderer(self, self.effects)
        self.emotion_handler = EmotionHandler(self)
        self.movement_handler = MovementHandler(self)
        self.memory_handler = MemoryHandler(self)
        self.utils = UtilityFunctions(self)
        self.particles = []
        self.mouth_phase = 0
        self.halo_animation = 0
        self.breath_phase = 0.0
        self.breath_amplitude = 0.015
        self.breath_frequency = 0.8
        self.touched_recently = False
        self.halo_feedback_timer = None
        self.current_mouth_shape = 'X'
        self.lip_sync_events = []
        self.speaking_start_time = None
        self.bind(
            pos=self.update_canvas,
            size=self.update_canvas,
            emotion=self.on_emotion_change,
            intensity=self.update_canvas,
            is_speaking=self.on_speaking_changed,
        )
        Clock.schedule_interval(self.animate, 1 / 30.0)
        Clock.schedule_interval(self.effects.update_emotional_waves, 1 / 30.0)
        self.enhance_beauty()
        self.setup_blinking()

    def update_canvas(self, *args):
        """Met à jour le canvas pour le dessin du visage."""
        self.canvas.clear()
        self.drawer.draw_face()
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
        self.breath_phase += dt * self.breath_frequency
        breath_value = math.sin(self.breath_phase)
        self.scale = 1.0 + breath_value * self.breath_amplitude
        if self.is_speaking:
            self.mouth_phase += dt * 15.0
            if self.speaking_start_time is None:
                self.speaking_start_time = time.time()
            speaking_duration = time.time() - self.speaking_start_time
            self._process_lip_sync_events(speaking_duration)
            if not self.lip_sync_events:
                if random.random() < 0.1:
                    speaking_shapes = ['A', 'E', 'O', 'I', 'OU', 'AN', 'EN']
                    self.current_mouth_shape = random.choice(speaking_shapes)
        else:
            self.speaking_start_time = None
            if self.emotion == 'surprise':
                self.current_mouth_shape = 'O'
            elif self.emotion == 'joie':
                self.current_mouth_shape = 'AI'
            elif self.emotion == 'tristesse':
                self.current_mouth_shape = 'EN'
            else:
                self.current_mouth_shape = 'X'
        self.update_canvas()

    def on_speaking_changed(self, instance, value):
        """
        Appelé quand l'état de parole change.

        Args:
            instance: Instance qui a changé
            value: Nouvelle valeur (True/False pour parler/arrêter)
        """
        if value:
            self.lip_sync_events = []
            self.speaking_start_time = None
            self.animate_mouth(True)
        else:
            self.animate_mouth(False)

    def animate_mouth(self, is_speaking: bool):
        """
        Contrôle l'animation de la bouche pendant la parole.

        Args:
            is_speaking: True si en train de parler, False sinon
        """
        if is_speaking:
            self.speaking_start_time = time.time()
            self.mouth_phase = 0
        else:
            self.current_mouth_shape = 'X'

    def _process_lip_sync_events(self, speaking_duration):
        """
        Traite les événements programmés pour l'animation labiale synchronisée.

        Args:
            speaking_duration: Temps écoulé depuis le début de la parole
        """
        for event in list(self.lip_sync_events):
            start_time = event.get('start_time', 0)
            end_time = event.get('end_time', 0)
            shape = event.get('shape', 'X')
            if start_time <= speaking_duration <= end_time:
                self.current_mouth_shape = shape
                return
            if speaking_duration > end_time:
                self.lip_sync_events.remove(event)

    def on_emotion_change(self, instance, value):
        """
        Appelé quand l'émotion principale change.

        Args:
            instance: Instance qui a changé
            value: Nouvelle valeur de l'émotion
        """
        self.emotion_handler.process_emotion_change(value, self.intensity)
        self.visual_renderer.render_emotion(
            emotion=value, intensity=self.intensity, secondary_emotion=self.emotion_secondary, blend=self.emotion_blend
        )

    def enhance_beauty(self):
        """Initialise les couches de beauté visuelles du visage."""
        self.beauty_layers = []
        self.beauty_layers.append({'scale': 1.05, 'opacity': 0.1, 'phase_offset': 0.0, 'speed': 0.2})
        self.beauty_layers.append({'scale': 1.12, 'opacity': 0.07, 'phase_offset': math.pi / 3, 'speed': 0.15})
        self.beauty_layers.append({'scale': 1.18, 'opacity': 0.05, 'phase_offset': 2 * math.pi / 3, 'speed': 0.1})

    def setup_blinking(self):
        """Initialise le système de clignement des yeux."""
        self.blink_interval = 5.0
        self.next_blink_time = time.time() + random.uniform(3.0, self.blink_interval)
        Clock.schedule_interval(self.check_for_blink, 0.1)

    def check_for_blink(self, dt):
        """
        Vérifie s'il est temps de clignoter des yeux.

        Args:
            dt: Delta temps depuis la dernière vérification
        """
        current_time = time.time()
        if current_time >= self.next_blink_time:
            self.blink()
            variation = random.uniform(-1.0, 1.0)
            next_interval = self.blink_interval + variation
            self.next_blink_time = current_time + max(2.0, next_interval)

    def blink(self):
        """Effectue un clignement d'yeux."""
        original_openness = self.eyelid_openness

        def blink_down(dt):
            self.eyelid_openness = max(0.1, self.eyelid_openness - 0.3)
            if self.eyelid_openness <= 0.1:
                Clock.schedule_once(blink_up, 0.05)
                return False
            return True

        def blink_up(dt):
            self.eyelid_openness = min(original_openness, self.eyelid_openness + 0.15)
            if self.eyelid_openness >= original_openness:
                self.eyelid_openness = original_openness
                return False
            return True

        Clock.schedule_interval(blink_down, 1 / 30.0)
