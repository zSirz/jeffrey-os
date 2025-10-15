"""
MainEmotionScreen - Interface principale immersive de Jeffrey, l'IA émotionnelle.

Ce module fournit une expérience visuelle et interactive complète représentant
le cœur émotionnel de Jeffrey - son essence vivante et sensible.
"""

import math
import random
import time

from kivy.animation import Animation
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Ellipse, Rectangle
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import BooleanProperty, ListProperty, NumericProperty
from kivy.uix.effectwidget import EffectWidget, HorizontalBlurEffect
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import Screen
from kivy.uix.widget import Widget

try:
    from jeffrey.core.emotions.jeffrey_emotional_traits import JeffreyEmotionalTraits
    from jeffrey.core.voice.jeffrey_voice_emotion_controller import JeffreyVoiceEmotionController
    from jeffrey.interfaces.ui.gesture_detector import MultiTouchGestureDetector
    from jeffrey.interfaces.ui.visual_effects_engine import VisualEffectsEngine

    from jeffrey.core.jeffrey_emotional_core import JeffreyEmotionalCore
except ImportError:
    print('Utilisation des mocks pour les tests')
    from unittest.mock import MagicMock

    JeffreyEmotionalTraits = MagicMock
    JeffreyEmotionalCore = MagicMock
    JeffreyVoiceEmotionController = MagicMock
    if 'MultiTouchGestureDetector' not in globals():

        class MultiTouchGestureDetector(Widget):
            def register_gesture_callback(self, gesture_type, callback):
                pass

    if 'VisualEffectsEngine' not in globals():

        class VisualEffectsEngine:
            def trigger_emotion_effect(self, emotion, intensity=0.5):
                return []


Builder.load_file('ui/kv/main_emotion_screen.kv')


class ParticleSystem(Widget):
    """Système de particules pour les effets d'ambiance et les réactions émotionnelles."""

    def __init__(self, **kwargs):
        super(ParticleSystem, self).__init__(**kwargs)
        self.particles = []
        self.max_particles = 100
        self.emotion_color = (0.5, 0.5, 1.0, 0.7)
        Clock.schedule_interval(self.update, 1 / 60)
        self.bind(size=self.on_size_change)

    def on_size_change(self, *args):
        """Réagit aux changements de taille de l'écran."""
        self.canvas.clear()
        self.particles = []

    def set_emotion_color(self, color):
        """
        Définit la couleur des particules en fonction de l'émotion actuelle.

        Args:
            color: Tuple RGBA représentant la couleur de l'émotion
        """
        self.emotion_color = color

    def add_particles(self, x, y, count=10, velocity_scale=1.0, lifespan_scale=1.0):
        """
        Ajoute de nouvelles particules à partir d'un point d'origine.

        Args:
            x, y: Coordonnées d'origine
            count: Nombre de particules à ajouter
            velocity_scale: Échelle de vitesse des particules
            lifespan_scale: Échelle de durée de vie des particules
        """
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.0) * velocity_scale
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            size = random.uniform(dp(2), dp(8))
            opacity = random.uniform(0.3, 0.9)
            lifespan = random.uniform(1.0, 3.0) * lifespan_scale
            color_variation = 0.1
            color = (
                max(0, min(1, self.emotion_color[0] + random.uniform(-color_variation, color_variation))),
                max(0, min(1, self.emotion_color[1] + random.uniform(-color_variation, color_variation))),
                max(0, min(1, self.emotion_color[2] + random.uniform(-color_variation, color_variation))),
                opacity,
            )
            particle = {
                'x': x,
                'y': y,
                'vx': vx,
                'vy': vy,
                'size': size,
                'color': color,
                'lifespan': lifespan,
                'age': 0,
            }
            if len(self.particles) < self.max_particles:
                self.particles.append(particle)
            else:
                index = random.randint(0, len(self.particles) - 1)
                self.particles[index] = particle

    def update(self, dt):
        """
        Met à jour et dessine les particules.

        Args:
            dt: Delta de temps depuis la dernière mise à jour
        """
        self.canvas.clear()
        particles_to_keep = []
        with self.canvas:
            for particle in self.particles:
                particle['age'] += dt
                if particle['age'] < particle['lifespan']:
                    age_factor = 1 - particle['age'] / particle['lifespan']
                    r, g, b, a = particle['color']
                    current_opacity = a * age_factor
                    Color(r, g, b, current_opacity)
                    particle['x'] += particle['vx'] * dt
                    particle['y'] += particle['vy'] * dt
                    particle['vy'] -= 0.05 * dt
                    size = particle['size'] * age_factor
                    Ellipse(pos=(particle['x'] - size / 2, particle['y'] - size / 2), size=(size, size))
                    particles_to_keep.append(particle)
        self.particles = particles_to_keep


class EmotionalCore(EffectWidget):
    """
    Représentation visuelle du cœur émotionnel de Jeffrey.
    Ce widget affiche une orbe pulsante qui reflète l'état émotionnel actuel.
    """

    pulse_scale = NumericProperty(1.0)
    base_color = ListProperty([0.3, 0.6, 1.0, 0.9])
    halo_color = ListProperty([0.4, 0.7, 1.0, 0.3])
    emotion_intensity = NumericProperty(0.5)
    is_touched = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(EmotionalCore, self).__init__(**kwargs)
        self.pulse_speed = 1.0
        self.emotion = 'neutral'
        self.pulse_animation = None
        self._start_pulse_animation()
        self.effects = [HorizontalBlurEffect(size=2.0)]
        self.touch_start_time = 0
        self.touch_pos = None
        self.long_press_event = None
        self.energy_waves = []
        self.max_waves = 3
        Clock.schedule_interval(self.update_core, 1 / 30)

    def _start_pulse_animation(self):
        """Démarre l'animation de pulsation du cœur."""
        if self.pulse_animation:
            self.pulse_animation.cancel(self)
        duration = 60 / (70 + self.emotion_intensity * 40)
        scale_min = 0.9 + 0.1 * (1.0 - self.emotion_intensity)
        scale_max = 1.0 + 0.2 * self.emotion_intensity
        self.pulse_animation = Animation(
            pulse_scale=scale_max, duration=duration * 0.3, transition='out_quad'
        ) + Animation(pulse_scale=scale_min, duration=duration * 0.7, transition='in_out_sine')
        self.pulse_animation.repeat = True
        self.pulse_animation.start(self)

    def set_emotion(self, emotion, intensity=0.5):
        """
        Définit l'émotion actuelle et ajuste l'apparence du cœur.

        Args:
            emotion: Chaîne représentant l'émotion (happy, sad, etc.)
            intensity: Intensité de l'émotion entre 0.0 et 1.0
        """
        self.emotion = emotion
        self.emotion_intensity = min(1.0, max(0.1, intensity))
        self.base_color = self.get_emotion_color(emotion)
        self.halo_color = [
            min(1.0, self.base_color[0] * 1.3),
            min(1.0, self.base_color[1] * 1.3),
            min(1.0, self.base_color[2] * 1.3),
            0.3 + 0.4 * self.emotion_intensity,
        ]
        self._start_pulse_animation()

    def get_emotion_color(self, emotion):
        """
        Retourne la couleur RGBA correspondant à une émotion.

        Args:
            emotion: Chaîne représentant l'émotion

        Returns:
            Tuple (r, g, b, a) représentant la couleur
        """
        emotion_colors = {
            'neutral': [0.3, 0.6, 1.0, 0.9],
            'calm': [0.4, 0.6, 0.9, 0.9],
            'peaceful': [0.5, 0.7, 0.9, 0.9],
            'relaxed': [0.3, 0.7, 0.8, 0.9],
            'melancholic': [0.3, 0.4, 0.7, 0.9],
            'happy': [1.0, 0.9, 0.3, 0.9],
            'excited': [1.0, 0.7, 0.2, 0.9],
            'joyful': [1.0, 0.8, 0.5, 0.9],
            'angry': [0.9, 0.1, 0.1, 0.9],
            'frustrated': [0.8, 0.3, 0.2, 0.9],
            'annoyed': [0.7, 0.4, 0.3, 0.9],
            'sad': [0.2, 0.3, 0.6, 0.9],
            'disappointed': [0.4, 0.4, 0.5, 0.9],
        }
        return emotion_colors.get(emotion.lower(), [0.3, 0.6, 1.0, 0.9])

    def update_core(self, dt):
        """
        Met à jour l'apparence du cœur émotionnel.

        Args:
            dt: Delta de temps depuis la dernière mise à jour
        """
        waves_to_keep = []
        for wave in self.energy_waves:
            wave['radius'] += wave['speed'] * dt
            wave['opacity'] -= 0.4 * dt
            if wave['opacity'] > 0:
                waves_to_keep.append(wave)
        self.energy_waves = waves_to_keep
        self.canvas.after.clear()
        with self.canvas.after:
            for wave in self.energy_waves:
                Color(*wave['color'][:3], wave['opacity'])
                center_x = self.center_x
                center_y = self.center_y
                radius = wave['radius']
                Ellipse(pos=(center_x - radius, center_y - radius), size=(radius * 2, radius * 2))

    def create_energy_wave(self, wave_type='pulse'):
        """
        Crée une nouvelle vague d'énergie émise par le cœur.

        Args:
            wave_type: Type de vague (pulse, burst, ripple)
        """
        base_speed = dp(30)
        base_opacity = 0.7
        if wave_type == 'pulse':
            speed = base_speed
            color = self.halo_color[:]
            start_radius = self.width * 0.2 * self.pulse_scale
        elif wave_type == 'burst':
            speed = base_speed * 1.5
            color = [
                min(1.0, self.base_color[0] * 1.5),
                min(1.0, self.base_color[1] * 1.5),
                min(1.0, self.base_color[2] * 1.5),
                base_opacity,
            ]
            start_radius = self.width * 0.15 * self.pulse_scale
        elif wave_type == 'ripple':
            speed = base_speed * 0.7
            color = self.base_color[:]
            color[3] = base_opacity * 0.8
            start_radius = self.width * 0.25 * self.pulse_scale
        else:
            speed = base_speed
            color = self.halo_color[:]
            start_radius = self.width * 0.2 * self.pulse_scale
        wave = {'radius': start_radius, 'speed': speed, 'color': color, 'opacity': base_opacity}
        self.energy_waves.append(wave)
        if len(self.energy_waves) > self.max_waves:
            self.energy_waves.pop(0)

    def on_touch_down(self, touch):
        """
        Gère les interactions tactiles avec le cœur.

        Args:
            touch: Objet touch contenant les informations sur le toucher

        Returns:
            True si le toucher est traité, False sinon
        """
        if self.collide_point(*touch.pos):
            self.touch_start_time = time.time()
            self.touch_pos = touch.pos
            self.is_touched = True
            self.create_energy_wave('burst')

            def trigger_long_press(dt):
                elapsed = time.time() - self.touch_start_time
                if elapsed >= 0.7 and self.is_touched:
                    self.react_to_interaction('long_press')

            self.long_press_event = Clock.schedule_once(trigger_long_press, 0.7)
            return True
        return super(EmotionalCore, self).on_touch_down(touch)

    def on_touch_move(self, touch):
        """
        Gère les mouvements tactiles sur le cœur.

        Args:
            touch: Objet touch contenant les informations sur le toucher

        Returns:
            True si le toucher est traité, False sinon
        """
        if self.is_touched and touch.grab_current is None:
            dx = touch.x - self.touch_pos[0]
            dy = touch.y - self.touch_pos[1]
            distance = math.sqrt(dx * dx + dy * dy)
            if distance > dp(50):
                self.react_to_interaction('caress')
                self.touch_pos = touch.pos
                if self.long_press_event:
                    self.long_press_event.cancel()
                    self.long_press_event = None
            return True
        return super(EmotionalCore, self).on_touch_move(touch)

    def on_touch_up(self, touch):
        """
        Gère la fin des interactions tactiles avec le cœur.

        Args:
            touch: Objet touch contenant les informations sur le toucher

        Returns:
            True si le toucher est traité, False sinon
        """
        if self.is_touched:
            elapsed = time.time() - self.touch_start_time
            if self.long_press_event:
                self.long_press_event.cancel()
                self.long_press_event = None
            if elapsed < 0.3:
                self.react_to_interaction('tap')
            self.is_touched = False
            self.touch_pos = None
            return True
        return super(EmotionalCore, self).on_touch_up(touch)

    def react_to_interaction(self, interaction_type):
        """
        Réagit aux différentes interactions tactiles.

        Args:
            interaction_type: Type d'interaction (tap, caress, long_press)
        """
        if interaction_type == 'tap':
            Animation(pulse_scale=1.3, duration=0.1).start(self)
            Clock.schedule_once(lambda dt: self._start_pulse_animation(), 0.15)
            self.create_energy_wave('ripple')
        elif interaction_type == 'caress':
            self.create_energy_wave('pulse')
        elif interaction_type == 'long_press':
            Animation(pulse_scale=1.5, duration=0.3).start(self)
            Clock.schedule_once(lambda dt: self._start_pulse_animation(), 0.35)
            for _ in range(3):
                self.create_energy_wave('burst')


class DreamBackground(Widget):
    """
    Fond animé représentant le monde onirique de Jeffrey.
    Affiche un ciel étoilé ou des motifs abstraits selon l'état émotionnel.
    """

    def __init__(self, **kwargs):
        super(DreamBackground, self).__init__(**kwargs)
        self.dream_mode = False
        self.stars = []
        self.nebulae = []
        self.emotion_tint = [0.1, 0.1, 0.3, 1.0]
        self.bind(size=self._update_background)
        Clock.schedule_interval(self.update, 1 / 30)
        self._generate_stars()
        self._generate_nebulae()

    def _update_background(self, *args):
        """Met à jour le fond quand la taille change."""
        self._generate_stars()
        self._generate_nebulae()

    def _generate_stars(self):
        """Génère un champ d'étoiles aléatoire."""
        self.stars = []
        star_count = int(self.width * self.height / 3000)
        for _ in range(star_count):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            size = random.uniform(dp(1), dp(3))
            brightness = random.uniform(0.3, 1.0)
            twinkle_speed = random.uniform(0.5, 2.0)
            twinkle_phase = random.uniform(0, 2 * math.pi)
            self.stars.append(
                {
                    'x': x,
                    'y': y,
                    'size': size,
                    'brightness': brightness,
                    'twinkle_speed': twinkle_speed,
                    'twinkle_phase': twinkle_phase,
                }
            )

    def _generate_nebulae(self):
        """Génère des nébuleuses abstraites en arrière-plan."""
        self.nebulae = []
        nebula_count = 3
        for _ in range(nebula_count):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            radius = random.uniform(self.width * 0.3, self.width * 0.5)
            opacity = random.uniform(0.05, 0.2)
            drift_x = random.uniform(-0.2, 0.2)
            drift_y = random.uniform(-0.2, 0.2)
            self.nebulae.append(
                {'x': x, 'y': y, 'radius': radius, 'opacity': opacity, 'drift_x': drift_x, 'drift_y': drift_y}
            )

    def set_emotion_tint(self, emotion):
        """
        Définit la teinte de couleur du fond selon l'émotion.

        Args:
            emotion: Émotion actuelle (happy, sad, etc.)
        """
        emotion_tints = {
            'happy': [0.2, 0.2, 0.1, 1.0],
            'excited': [0.2, 0.15, 0.1, 1.0],
            'joyful': [0.2, 0.18, 0.1, 1.0],
            'angry': [0.2, 0.1, 0.1, 1.0],
            'frustrated': [0.2, 0.12, 0.1, 1.0],
            'sad': [0.1, 0.1, 0.2, 1.0],
            'melancholic': [0.1, 0.1, 0.3, 1.0],
            'calm': [0.1, 0.15, 0.2, 1.0],
            'neutral': [0.1, 0.1, 0.15, 1.0],
            'peaceful': [0.15, 0.2, 0.2, 1.0],
        }
        self.emotion_tint = emotion_tints.get(emotion.lower(), [0.1, 0.1, 0.2, 1.0])

    def set_dream_mode(self, enabled):
        """
        Active ou désactive le mode rêve.

        Args:
            enabled: True pour activer le mode rêve, False sinon
        """
        self.dream_mode = enabled

    def update(self, dt):
        """
        Met à jour l'animation du fond.

        Args:
            dt: Delta de temps depuis la dernière mise à jour
        """
        for star in self.stars:
            star['twinkle_phase'] += star['twinkle_speed'] * dt
            if star['twinkle_phase'] > 2 * math.pi:
                star['twinkle_phase'] -= 2 * math.pi
        for nebula in self.nebulae:
            nebula['x'] += nebula['drift_x'] * dt
            nebula['y'] += nebula['drift_y'] * dt
            if nebula['x'] < -nebula['radius'] or nebula['x'] > self.width + nebula['radius']:
                nebula['drift_x'] *= -1
            if nebula['y'] < -nebula['radius'] or nebula['y'] > self.height + nebula['radius']:
                nebula['drift_y'] *= -1
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self.emotion_tint)
            Rectangle(pos=self.pos, size=self.size)
            for nebula in self.nebulae:
                r, g, b, a = self.emotion_tint
                nebula_color = [min(1.0, r * 1.5), min(1.0, g * 1.5), min(1.0, b * 1.5), nebula['opacity']]
                Color(*nebula_color)
                center_x = nebula['x']
                center_y = nebula['y']
                radius = nebula['radius']
                Ellipse(pos=(center_x - radius, center_y - radius), size=(radius * 2, radius * 2))
            for star in self.stars:
                current_brightness = star['brightness'] * (0.7 + 0.3 * math.sin(star['twinkle_phase']))
                r, g, b, _ = self.emotion_tint
                star_color = [0.7 + r * 0.3, 0.7 + g * 0.3, 0.7 + b * 0.3, current_brightness]
                Color(*star_color)
                x, y = (star['x'], star['y'])
                size = star['size']
                Ellipse(pos=(x - size / 2, y - size / 2), size=(size, size))
                if star['brightness'] > 0.8:
                    halo_size = size * 3
                    halo_color = star_color.copy()
                    halo_color[3] *= 0.3
                    Color(*halo_color)
                    Ellipse(pos=(x - halo_size / 2, y - halo_size / 2), size=(halo_size, halo_size))


class MainEmotionScreen(Screen):
    """
    Écran principal représentant Jeffrey comme une entité émotionnelle vivante.
    Offre une interface immersive et fluide pour interagir avec Jeffrey.
    """

    def __init__(self, **kwargs):
        super(MainEmotionScreen, self).__init__(**kwargs)
        self.layout = FloatLayout()
        self.add_widget(self.layout)
        self.dream_background = DreamBackground()
        self.layout.add_widget(self.dream_background)
        self.particle_system = ParticleSystem()
        self.layout.add_widget(self.particle_system)
        self.core = EmotionalCore(size_hint=(None, None))
        self.core.size = (dp(200), dp(200))
        self.core.pos_hint = {'center_x': 0.5, 'center_y': 0.5}
        self.layout.add_widget(self.core)
        self.gesture_detector = MultiTouchGestureDetector()
        self.layout.add_widget(self.gesture_detector)
        self._register_gestures()
        self.visual_effects = VisualEffectsEngine(ui_controller=self)
        try:
            self.jeffrey_emotions = JeffreyEmotionalCore()
            self.voice_controller = JeffreyVoiceEmotionController()
        except:
            self.jeffrey_emotions = MagicMock()
            self.jeffrey_emotions.get_humeur_color = lambda: (0.3, 0.6, 1.0)
            self.jeffrey_emotions.reagir_au_calin = lambda type: None
            self.jeffrey_emotions.get_current_emotion = lambda: 'neutral'
            self.jeffrey_emotions.get_emotional_intensity = lambda: 0.5
            self.voice_controller = None
        self.current_emotion = 'neutral'
        self.emotion_intensity = 0.5
        self.in_dream_mode = False
        self._update_emotional_state()
        Clock.schedule_interval(self._update_emotional_state, 1.0)
        Clock.schedule_interval(self._pulse_energy, 5.0)
        Window.bind(on_resize=self._on_window_resize)

    def _on_window_resize(self, instance, width, height):
        """Gère le redimensionnement de la fenêtre."""
        self.core.size = (min(dp(200), width * 0.3), min(dp(200), width * 0.3))

    def _register_gestures(self):
        """Enregistre les callbacks pour les différents gestes."""
        self.gesture_detector.register_gesture_callback('tap', self._on_tap)
        self.gesture_detector.register_gesture_callback('long_press', self._on_long_press)
        self.gesture_detector.register_gesture_callback('swipe_left', self._on_swipe)
        self.gesture_detector.register_gesture_callback('swipe_right', self._on_swipe)
        self.gesture_detector.register_gesture_callback('swipe_up', self._on_swipe)
        self.gesture_detector.register_gesture_callback('swipe_down', self._on_swipe)
        self.gesture_detector.register_gesture_callback('pinch_in', self._on_pinch)
        self.gesture_detector.register_gesture_callback('pinch_out', self._on_pinch)

    def _update_emotional_state(self, *args):
        """
        Met à jour l'état émotionnel de l'interface en fonction de Jeffrey.
        """
        try:
            new_emotion = self.jeffrey_emotions.get_current_emotion()
            new_intensity = self.jeffrey_emotions.get_emotional_intensity()
            if new_emotion != self.current_emotion or abs(new_intensity - self.emotion_intensity) > 0.1:
                self.current_emotion = new_emotion
                self.emotion_intensity = new_intensity
                self.core.set_emotion(new_emotion, new_intensity)
                self.dream_background.set_emotion_tint(new_emotion)
                self.particle_system.set_emotion_color(self.core.halo_color)
                self.visual_effects.trigger_emotion_effect(new_emotion, new_intensity)
        except:
            self.core.set_emotion('neutral', 0.5)

    def _pulse_energy(self, dt):
        """
        Crée un effet de pulsation d'énergie périodique.

        Args:
            dt: Delta temps
        """
        x, y = self.core.center
        self.particle_system.add_particles(
            x,
            y,
            count=int(5 + 15 * self.emotion_intensity),
            velocity_scale=0.5 + self.emotion_intensity,
            lifespan_scale=1.0,
        )
        if random.random() < 0.3 + 0.5 * self.emotion_intensity:
            self.core.create_energy_wave('pulse')

    def play_visual_effect(self, effect_name, effect_config):
        """
        Joue un effet visuel spécifique.

        Args:
            effect_name: Nom de l'effet
            effect_config: Configuration de l'effet
        """
        if effect_name == 'halo_lumineux':
            self._play_halo_effect(effect_config)
        elif effect_name == 'sparkles':
            self._play_sparkles_effect(effect_config)
        elif effect_name == 'brume':
            self._play_mist_effect(effect_config)
        elif effect_name == 'flash_rouge':
            self._play_flash_effect(effect_config)

    def _play_halo_effect(self, config):
        """Joue un effet de halo lumineux."""
        color = config.get('color', (255, 255, 190))
        r, g, b = color
        r, g, b = (r / 255.0, g / 255.0, b / 255.0)
        opacity = config.get('opacity', 0.7)
        self.core.halo_color = [r, g, b, opacity]
        anim = Animation(halo_color=[r, g, b, opacity], duration=0.3) + Animation(
            halo_color=[r, g, b, opacity * 0.5], duration=0.7
        )
        anim.start(self.core)
        self.core.create_energy_wave('pulse')

    def _play_sparkles_effect(self, config):
        """Joue un effet d'étincelles."""
        color = config.get('color', (255, 255, 255))
        r, g, b = color
        r, g, b = (r / 255.0, g / 255.0, b / 255.0)
        count = config.get('count', 20)
        x, y = self.core.center
        self.particle_system.add_particles(x, y, count=count, velocity_scale=2.0, lifespan_scale=1.5)

    def _play_mist_effect(self, config):
        """Joue un effet de brume."""
        color = config.get('color', (100, 100, 150))
        r, g, b = color
        r, g, b = (r / 255.0, g / 255.0, b / 255.0)
        opacity = config.get('opacity', 0.5)
        self.dream_background.emotion_tint[3] = 1.5
        anim = Animation(
            emotion_tint=[
                self.dream_background.emotion_tint[0],
                self.dream_background.emotion_tint[1],
                self.dream_background.emotion_tint[2],
                1.0,
            ],
            duration=2.0,
        )
        anim.start(self.dream_background)
        for _ in range(3):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            self.particle_system.add_particles(x, y, count=15, velocity_scale=0.3, lifespan_scale=3.0)

    def _play_flash_effect(self, config):
        """Joue un effet de flash."""
        color = config.get('color', (255, 0, 0))
        r, g, b = color
        r, g, b = (r / 255.0, g / 255.0, b / 255.0)
        intensity = config.get('intensity', 0.8)
        original_color = self.core.base_color[:]
        flash_color = [r, g, b, 1.0]
        anim = Animation(base_color=flash_color, duration=0.1) + Animation(base_color=original_color, duration=0.3)
        anim.start(self.core)
        self.core.create_energy_wave('burst')

    def clear_visual_effects(self):
        """Efface tous les effets visuels en cours."""
        self._update_emotional_state()

    def _on_tap(self, touch_data):
        """
        Gère un tapotement sur l'écran.

        Args:
            touch_data: Données du toucher
        """
        x, y = (touch_data[0], touch_data[1])
        core_x, core_y = self.core.center
        distance = math.sqrt((x - core_x) ** 2 + (y - core_y) ** 2)
        if distance < self.core.width:
            self.jeffrey_emotions.reagir_au_calin('bisou')
            self.core.react_to_interaction('tap')
            self._play_sparkles_effect({'count': 10})
        else:
            self.particle_system.add_particles(x, y, count=5)

    def _on_long_press(self, touch_data):
        """
        Gère une pression longue sur l'écran.

        Args:
            touch_data: Données du toucher
        """
        x, y = (touch_data[0], touch_data[1])
        core_x, core_y = self.core.center
        distance = math.sqrt((x - core_x) ** 2 + (y - core_y) ** 2)
        if distance < self.core.width * 1.5:
            self.jeffrey_emotions.reagir_au_calin('câlin')
            self.core.react_to_interaction('long_press')
            self._play_halo_effect(
                {'color': [int(c * 255) for c in self.core.base_color[:3]], 'opacity': 0.9, 'scale': 1.5}
            )

    def _on_swipe(self, touch_data):
        """
        Gère un glissement sur l'écran.

        Args:
            touch_data: Données du toucher (points de début et de fin)
        """
        start_point, end_point = touch_data
        start_x, start_y = start_point
        end_x, end_y = end_point
        dx = end_x - start_x
        dy = end_y - start_y
        core_x, core_y = self.core.center
        midpoint_x = (start_x + end_x) / 2
        midpoint_y = (start_y + end_y) / 2
        distance_to_core = math.sqrt((midpoint_x - core_x) ** 2 + (midpoint_y - core_y) ** 2)
        if distance_to_core < self.core.width * 1.5:
            self.jeffrey_emotions.reagir_au_calin('caresse')
            steps = 10
            for i in range(steps):
                t = i / float(steps - 1)
                x = start_x + dx * t
                y = start_y + dy * t
                self.particle_system.add_particles(x, y, count=2)
        else:
            for i in range(3):
                x = start_x + dx * (i / 2.0)
                y = start_y + dy * (i / 2.0)
                self.particle_system.add_particles(x, y, count=3)

    def _on_pinch(self, touch_data):
        """
        Gère un pincement multi-touch.

        Args:
            touch_data: Données du toucher
        """
        center = touch_data['center']
        scale_factor = touch_data['scale_factor']
        core_x, core_y = self.core.center
        distance_to_core = math.sqrt((center[0] - core_x) ** 2 + (center[1] - core_y) ** 2)
        if distance_to_core < self.core.width:
            self.jeffrey_emotions.reagir_au_calin('main_posée')
            if scale_factor > 1.0:
                self.core.create_energy_wave('burst')
            else:
                self._play_halo_effect(
                    {'color': [int(c * 255) for c in self.core.base_color[:3]], 'opacity': 0.5, 'scale': 0.8}
                )

    def toggle_dream_mode(self, enabled=None):
        """
        Active ou désactive le mode rêve.

        Args:
            enabled: True pour activer, False pour désactiver, None pour basculer
        """
        if enabled is None:
            enabled = not self.in_dream_mode
        self.in_dream_mode = enabled
        self.dream_background.set_dream_mode(enabled)
        if enabled:
            Animation(emotion_intensity=0.2, duration=2.0).start(self)
            self.core.set_emotion('peaceful', 0.2)
        else:
            self._update_emotional_state()

    def on_enter(self):
        """Appelé quand l'écran devient actif."""
        self._update_emotional_state()
        x, y = self.core.center
        self.particle_system.add_particles(x, y, count=30, velocity_scale=1.5)
        self.core.create_energy_wave('burst')

    def on_leave(self):
        """Appelé quand l'écran n'est plus actif."""
        self.clear_visual_effects()
