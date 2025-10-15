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
    from core.emotions.jeffrey_emotional_traits import JeffreyEmotionalTraits
    from core.jeffrey_emotional_core import JeffreyEmotionalCore
    from core.voice.jeffrey_voice_emotion_controller import JeffreyVoiceEmotionController

    from ui.gesture_detector import MultiTouchGestureDetector
    from ui.visual_effects_engine import VisualEffectsEngine
except ImportError:
    # Mode mock pour les tests
    print("Utilisation des mocks pour les tests")
    from unittest.mock import MagicMock

    JeffreyEmotionalTraits = MagicMock
    JeffreyEmotionalCore = MagicMock
    JeffreyVoiceEmotionController = MagicMock
    if "MultiTouchGestureDetector" not in globals():

        class MultiTouchGestureDetector(Widget):
            def register_gesture_callback(self, gesture_type, callback):
                pass

    if "VisualEffectsEngine" not in globals():

        class VisualEffectsEngine:
            def trigger_emotion_effect(self, emotion, intensity=0.5):
                return []


# Chargement du fichier KV pour la conception de l'interface
Builder.load_file("ui/kv/main_emotion_screen.kv")


class ParticleSystem(Widget):
    """Système de particules pour les effets d'ambiance et les réactions émotionnelles."""

    def __init__(self, **kwargs):
        super(ParticleSystem, self).__init__(**kwargs)
        self.particles = []
        self.max_particles = 100
        self.emotion_color = (0.5, 0.5, 1.0, 0.7)  # Couleur de base (bleu doux)
        Clock.schedule_interval(self.update, 1 / 60)
        self.bind(size=self.on_size_change)

    def on_size_change(self, *args):
        """Réagit aux changements de taille de l'écran."""
        self.canvas.clear()
        # Réinitialiser les particules si la taille change
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
            # Angle aléatoire pour le mouvement
            angle = random.uniform(0, 2 * math.pi)
            # Vitesse aléatoire
            speed = random.uniform(0.5, 2.0) * velocity_scale
            # Calcul des composantes de la vitesse
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            # Calcul de la taille et de l'opacité
            size = random.uniform(dp(2), dp(8))
            opacity = random.uniform(0.3, 0.9)
            # Durée de vie aléatoire
            lifespan = random.uniform(1.0, 3.0) * lifespan_scale

            # Léger décalage aléatoire pour la couleur
            color_variation = 0.1
            color = (
                max(
                    0,
                    min(1, self.emotion_color[0] + random.uniform(-color_variation, color_variation)),
                ),
                max(
                    0,
                    min(1, self.emotion_color[1] + random.uniform(-color_variation, color_variation)),
                ),
                max(
                    0,
                    min(1, self.emotion_color[2] + random.uniform(-color_variation, color_variation)),
                ),
                opacity,
            )

            # Création de la particule
            particle = {
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "size": size,
                "color": color,
                "lifespan": lifespan,
                "age": 0,
            }

            # Ajout à la liste des particules actives
            if len(self.particles) < self.max_particles:
                self.particles.append(particle)
            else:
                # Remplacer une particule existante si la limite est atteinte
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
                # Mettre à jour l'âge
                particle["age"] += dt

                # Vérifier si la particule est encore vivante
                if particle["age"] < particle["lifespan"]:
                    # Calculer le facteur de vieillissement (1 au début, 0 à la fin)
                    age_factor = 1 - (particle["age"] / particle["lifespan"])

                    # Appliquer le facteur à l'opacité
                    r, g, b, a = particle["color"]
                    current_opacity = a * age_factor

                    # Dessiner la particule
                    Color(r, g, b, current_opacity)

                    # Calculer la nouvelle position
                    particle["x"] += particle["vx"] * dt
                    particle["y"] += particle["vy"] * dt

                    # Effet de gravité légère
                    particle["vy"] -= 0.05 * dt

                    # Dessiner la particule comme un cercle
                    size = particle["size"] * age_factor
                    Ellipse(pos=(particle["x"] - size / 2, particle["y"] - size / 2), size=(size, size))

                    # Conserver cette particule
                    particles_to_keep.append(particle)

        # Mettre à jour la liste des particules
        self.particles = particles_to_keep


class EmotionalCore(EffectWidget):
    """
    Représentation visuelle du cœur émotionnel de Jeffrey.
    Ce widget affiche une orbe pulsante qui reflète l'état émotionnel actuel.
    """

    pulse_scale = NumericProperty(1.0)
    base_color = ListProperty([0.3, 0.6, 1.0, 0.9])  # Bleu doux par défaut
    halo_color = ListProperty([0.4, 0.7, 1.0, 0.3])  # Halo légèrement plus clair
    emotion_intensity = NumericProperty(0.5)
    is_touched = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(EmotionalCore, self).__init__(**kwargs)
        self.pulse_speed = 1.0  # Vitesse de pulsation (Hz)
        self.emotion = "neutral"

        # Initialiser les animations de pulsation
        self.pulse_animation = None
        self._start_pulse_animation()

        # Créer l'effet de flou pour le halo
        self.effects = [HorizontalBlurEffect(size=2.0)]

        # Initialiser les interactions tactiles
        self.touch_start_time = 0
        self.touch_pos = None
        self.long_press_event = None

        # Paramètres des vagues d'énergie
        self.energy_waves = []
        self.max_waves = 3

        # Programmer la mise à jour
        Clock.schedule_interval(self.update_core, 1 / 30)

    def _start_pulse_animation(self):
        """Démarre l'animation de pulsation du cœur."""
        # Annuler l'animation précédente si elle existe
        if self.pulse_animation:
            self.pulse_animation.cancel(self)

        # Créer la nouvelle animation avec des paramètres basés sur l'émotion
        duration = 60 / (70 + (self.emotion_intensity * 40))  # Entre 0.5 et 1.0 seconde par battement
        scale_min = 0.9 + (0.1 * (1.0 - self.emotion_intensity))  # Entre 0.9 et 1.0
        scale_max = 1.0 + (0.2 * self.emotion_intensity)  # Entre 1.0 et 1.2

        # Animation aller-retour de la pulsation
        self.pulse_animation = Animation(
            pulse_scale=scale_max, duration=duration * 0.3, transition="out_quad"
        ) + Animation(pulse_scale=scale_min, duration=duration * 0.7, transition="in_out_sine")

        # Répéter indéfiniment
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

        # Ajuster la couleur en fonction de l'émotion
        self.base_color = self.get_emotion_color(emotion)

        # Calculer une couleur de halo légèrement plus claire
        self.halo_color = [
            min(1.0, self.base_color[0] * 1.3),
            min(1.0, self.base_color[1] * 1.3),
            min(1.0, self.base_color[2] * 1.3),
            0.3 + (0.4 * self.emotion_intensity),  # Opacité du halo
        ]

        # Ajuster la pulsation en fonction de l'émotion
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
            # Tons bleus/violets
            "neutral": [0.3, 0.6, 1.0, 0.9],  # Bleu doux
            "calm": [0.4, 0.6, 0.9, 0.9],  # Bleu calme
            "peaceful": [0.5, 0.7, 0.9, 0.9],  # Bleu clair apaisant
            "relaxed": [0.3, 0.7, 0.8, 0.9],  # Bleu-vert relaxant
            "melancholic": [0.3, 0.4, 0.7, 0.9],  # Bleu foncé mélancolique
            # Tons jaunes/oranges (joie)
            "happy": [1.0, 0.9, 0.3, 0.9],  # Jaune vif
            "excited": [1.0, 0.7, 0.2, 0.9],  # Orange vif
            "joyful": [1.0, 0.8, 0.5, 0.9],  # Jaune-orange joyeux
            # Tons rouges (colère)
            "angry": [0.9, 0.1, 0.1, 0.9],  # Rouge vif
            "frustrated": [0.8, 0.3, 0.2, 0.9],  # Rouge-orange frustré
            "annoyed": [0.7, 0.4, 0.3, 0.9],  # Rouge-brun agacé
            # Tons bleu foncé/gris (tristesse)
            "sad": [0.2, 0.3, 0.6, 0.9],  # Bleu triste
            "disappointed": [0.4, 0.4, 0.5, 0.9],  # Gris-bleu déçu
        }

        # Retourner la couleur correspondante ou neutre par défaut
        return emotion_colors.get(emotion.lower(), [0.3, 0.6, 1.0, 0.9])

    def update_core(self, dt):
        """
        Met à jour l'apparence du cœur émotionnel.

        Args:
            dt: Delta de temps depuis la dernière mise à jour
        """
        # Mettre à jour les vagues d'énergie
        waves_to_keep = []
        for wave in self.energy_waves:
            wave["radius"] += wave["speed"] * dt
            wave["opacity"] -= 0.4 * dt
            if wave["opacity"] > 0:
                waves_to_keep.append(wave)
        self.energy_waves = waves_to_keep

        # Redessiner le widget
        self.canvas.after.clear()
        with self.canvas.after:
            # Dessiner les vagues d'énergie
            for wave in self.energy_waves:
                Color(*wave["color"][:3], wave["opacity"])
                center_x = self.center_x
                center_y = self.center_y
                radius = wave["radius"]
                Ellipse(pos=(center_x - radius, center_y - radius), size=(radius * 2, radius * 2))

    def create_energy_wave(self, wave_type="pulse"):
        """
        Crée une nouvelle vague d'énergie émise par le cœur.

        Args:
            wave_type: Type de vague (pulse, burst, ripple)
        """
        base_speed = dp(30)  # Vitesse de base en pixels par seconde
        base_opacity = 0.7  # Opacité de base

        if wave_type == "pulse":
            speed = base_speed
            color = self.halo_color[:]
            start_radius = self.width * 0.2 * self.pulse_scale
        elif wave_type == "burst":
            speed = base_speed * 1.5
            color = [
                min(1.0, self.base_color[0] * 1.5),
                min(1.0, self.base_color[1] * 1.5),
                min(1.0, self.base_color[2] * 1.5),
                base_opacity,
            ]
            start_radius = self.width * 0.15 * self.pulse_scale
        elif wave_type == "ripple":
            speed = base_speed * 0.7
            color = self.base_color[:]
            color[3] = base_opacity * 0.8
            start_radius = self.width * 0.25 * self.pulse_scale
        else:
            # Vague par défaut
            speed = base_speed
            color = self.halo_color[:]
            start_radius = self.width * 0.2 * self.pulse_scale

        # Créer la vague
        wave = {"radius": start_radius, "speed": speed, "color": color, "opacity": base_opacity}

        # Ajouter à la liste des vagues actives
        self.energy_waves.append(wave)

        # Limiter le nombre de vagues
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
            # Marquer le début du toucher
            self.touch_start_time = time.time()
            self.touch_pos = touch.pos
            self.is_touched = True

            # Créer une vague d'énergie
            self.create_energy_wave("burst")

            # Programmer un événement de longue pression
            def trigger_long_press(dt):
                elapsed = time.time() - self.touch_start_time
                if elapsed >= 0.7 and self.is_touched:  # Si toujours touché après 0.7s
                    self.react_to_interaction("long_press")

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
            # Calculer la distance parcourue
            dx = touch.x - self.touch_pos[0]
            dy = touch.y - self.touch_pos[1]
            distance = math.sqrt(dx * dx + dy * dy)

            # Si le mouvement est significatif, c'est une caresse
            if distance > dp(50):
                self.react_to_interaction("caress")
                self.touch_pos = touch.pos

                # Annuler l'événement de longue pression
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
            # Calculer la durée du toucher
            elapsed = time.time() - self.touch_start_time

            # Annuler l'événement de longue pression
            if self.long_press_event:
                self.long_press_event.cancel()
                self.long_press_event = None

            # Tap rapide
            if elapsed < 0.3:
                self.react_to_interaction("tap")

            # Réinitialiser l'état
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
        # Créer différents effets visuels selon l'interaction
        if interaction_type == "tap":
            # Tapotement - effet de rebond
            Animation(pulse_scale=1.3, duration=0.1).start(self)
            Clock.schedule_once(lambda dt: self._start_pulse_animation(), 0.15)
            self.create_energy_wave("ripple")

        elif interaction_type == "caress":
            # Caresse - effet de vague douce
            self.create_energy_wave("pulse")

        elif interaction_type == "long_press":
            # Pression longue - effet de rayonnement intense
            Animation(pulse_scale=1.5, duration=0.3).start(self)
            Clock.schedule_once(lambda dt: self._start_pulse_animation(), 0.35)
            for _ in range(3):
                self.create_energy_wave("burst")


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
        self.emotion_tint = [0.1, 0.1, 0.3, 1.0]  # Teinte émotionnelle (bleu foncé par défaut)
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
        star_count = int((self.width * self.height) / 3000)  # Densité d'étoiles

        for _ in range(star_count):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            size = random.uniform(dp(1), dp(3))
            brightness = random.uniform(0.3, 1.0)
            twinkle_speed = random.uniform(0.5, 2.0)
            twinkle_phase = random.uniform(0, 2 * math.pi)

            self.stars.append(
                {
                    "x": x,
                    "y": y,
                    "size": size,
                    "brightness": brightness,
                    "twinkle_speed": twinkle_speed,
                    "twinkle_phase": twinkle_phase,
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
                {
                    "x": x,
                    "y": y,
                    "radius": radius,
                    "opacity": opacity,
                    "drift_x": drift_x,
                    "drift_y": drift_y,
                }
            )

    def set_emotion_tint(self, emotion):
        """
        Définit la teinte de couleur du fond selon l'émotion.

        Args:
            emotion: Émotion actuelle (happy, sad, etc.)
        """
        # Définir une teinte de couleur adaptée à l'émotion
        emotion_tints = {
            "happy": [0.2, 0.2, 0.1, 1.0],  # Jaune doux
            "excited": [0.2, 0.15, 0.1, 1.0],  # Orange doux
            "joyful": [0.2, 0.18, 0.1, 1.0],  # Or doux
            "angry": [0.2, 0.1, 0.1, 1.0],  # Rouge sombre
            "frustrated": [0.2, 0.12, 0.1, 1.0],  # Rouge-orange sombre
            "sad": [0.1, 0.1, 0.2, 1.0],  # Bleu sombre
            "melancholic": [0.1, 0.1, 0.3, 1.0],  # Bleu profond
            "calm": [0.1, 0.15, 0.2, 1.0],  # Bleu-vert doux
            "neutral": [0.1, 0.1, 0.15, 1.0],  # Bleu-gris neutre
            "peaceful": [0.15, 0.2, 0.2, 1.0],  # Turquoise doux
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
        # Mettre à jour les étoiles (scintillement)
        for star in self.stars:
            star["twinkle_phase"] += star["twinkle_speed"] * dt
            if star["twinkle_phase"] > 2 * math.pi:
                star["twinkle_phase"] -= 2 * math.pi

        # Faire dériver lentement les nébuleuses
        for nebula in self.nebulae:
            nebula["x"] += nebula["drift_x"] * dt
            nebula["y"] += nebula["drift_y"] * dt

            # Rebondir sur les bords
            if nebula["x"] < -nebula["radius"] or nebula["x"] > self.width + nebula["radius"]:
                nebula["drift_x"] *= -1
            if nebula["y"] < -nebula["radius"] or nebula["y"] > self.height + nebula["radius"]:
                nebula["drift_y"] *= -1

        # Redessiner le fond
        self.canvas.before.clear()
        with self.canvas.before:
            # Fond de couleur sombre avec la teinte émotionnelle
            Color(*self.emotion_tint)
            Rectangle(pos=self.pos, size=self.size)

            # Dessiner les nébuleuses
            for nebula in self.nebulae:
                # Créer un dégradé pour la nébuleuse
                r, g, b, a = self.emotion_tint
                # Légèrement plus lumineux que le fond
                nebula_color = [
                    min(1.0, r * 1.5),
                    min(1.0, g * 1.5),
                    min(1.0, b * 1.5),
                    nebula["opacity"],
                ]
                Color(*nebula_color)

                # Dessiner la nébuleuse comme un cercle flou
                center_x = nebula["x"]
                center_y = nebula["y"]
                radius = nebula["radius"]
                Ellipse(pos=(center_x - radius, center_y - radius), size=(radius * 2, radius * 2))

            # Dessiner les étoiles
            for star in self.stars:
                # Calculer la luminosité actuelle (scintillement)
                current_brightness = star["brightness"] * (0.7 + 0.3 * math.sin(star["twinkle_phase"]))

                # Couleur de l'étoile (blanc légèrement teinté)
                r, g, b, _ = self.emotion_tint
                star_color = [0.7 + (r * 0.3), 0.7 + (g * 0.3), 0.7 + (b * 0.3), current_brightness]

                Color(*star_color)

                # Dessiner l'étoile
                x, y = star["x"], star["y"]
                size = star["size"]
                Ellipse(pos=(x - size / 2, y - size / 2), size=(size, size))

                # Étoiles plus brillantes ont un halo
                if star["brightness"] > 0.8:
                    halo_size = size * 3
                    halo_color = star_color.copy()
                    halo_color[3] *= 0.3  # Halo plus transparent
                    Color(*halo_color)
                    Ellipse(pos=(x - halo_size / 2, y - halo_size / 2), size=(halo_size, halo_size))


class MainEmotionScreen(Screen):
    """
    Écran principal représentant Jeffrey comme une entité émotionnelle vivante.
    Offre une interface immersive et fluide pour interagir avec Jeffrey.
    """

    def __init__(self, **kwargs):
        super(MainEmotionScreen, self).__init__(**kwargs)

        # Créer le layout principal
        self.layout = FloatLayout()
        self.add_widget(self.layout)

        # Créer le fond animé
        self.dream_background = DreamBackground()
        self.layout.add_widget(self.dream_background)

        # Créer le système de particules
        self.particle_system = ParticleSystem()
        self.layout.add_widget(self.particle_system)

        # Créer le cœur émotionnel
        self.core = EmotionalCore(size_hint=(None, None))
        self.core.size = (dp(200), dp(200))
        self.core.pos_hint = {"center_x": 0.5, "center_y": 0.5}
        self.layout.add_widget(self.core)

        # Initialiser le détecteur de gestes
        self.gesture_detector = MultiTouchGestureDetector()
        self.layout.add_widget(self.gesture_detector)
        self._register_gestures()

        # Initiliser le moteur d'effets visuels
        self.visual_effects = VisualEffectsEngine(ui_controller=self)

        # Essayer d'initialiser les composants émotionnels
        try:
            self.jeffrey_emotions = JeffreyEmotionalCore()
            self.voice_controller = JeffreyVoiceEmotionController()
        except:
            # Mode mock si les composants ne sont pas disponibles
            self.jeffrey_emotions = MagicMock()
            self.jeffrey_emotions.get_humeur_color = lambda: (0.3, 0.6, 1.0)
            self.jeffrey_emotions.reagir_au_calin = lambda type: None
            self.jeffrey_emotions.get_current_emotion = lambda: "neutral"
            self.jeffrey_emotions.get_emotional_intensity = lambda: 0.5

            self.voice_controller = None

        # État interne
        self.current_emotion = "neutral"
        self.emotion_intensity = 0.5
        self.in_dream_mode = False

        # Initialiser l'état émotionnel
        self._update_emotional_state()

        # Programmer les mises à jour régulières
        Clock.schedule_interval(self._update_emotional_state, 1.0)
        Clock.schedule_interval(self._pulse_energy, 5.0)

        # Gérer le redimensionnement de l'écran
        Window.bind(on_resize=self._on_window_resize)

    def _on_window_resize(self, instance, width, height):
        """Gère le redimensionnement de la fenêtre."""
        self.core.size = (min(dp(200), width * 0.3), min(dp(200), width * 0.3))

    def _register_gestures(self):
        """Enregistre les callbacks pour les différents gestes."""
        # Gestes de base
        self.gesture_detector.register_gesture_callback("tap", self._on_tap)
        self.gesture_detector.register_gesture_callback("long_press", self._on_long_press)
        self.gesture_detector.register_gesture_callback("swipe_left", self._on_swipe)
        self.gesture_detector.register_gesture_callback("swipe_right", self._on_swipe)
        self.gesture_detector.register_gesture_callback("swipe_up", self._on_swipe)
        self.gesture_detector.register_gesture_callback("swipe_down", self._on_swipe)

        # Gestes multi-touch
        self.gesture_detector.register_gesture_callback("pinch_in", self._on_pinch)
        self.gesture_detector.register_gesture_callback("pinch_out", self._on_pinch)

    def _update_emotional_state(self, *args):
        """
        Met à jour l'état émotionnel de l'interface en fonction de Jeffrey.
        """
        try:
            # Récupérer l'émotion actuelle et son intensité
            new_emotion = self.jeffrey_emotions.get_current_emotion()
            new_intensity = self.jeffrey_emotions.get_emotional_intensity()

            # Si l'émotion a changé, mettre à jour l'interface
            if new_emotion != self.current_emotion or abs(new_intensity - self.emotion_intensity) > 0.1:
                self.current_emotion = new_emotion
                self.emotion_intensity = new_intensity

                # Mettre à jour les composants visuels
                self.core.set_emotion(new_emotion, new_intensity)
                self.dream_background.set_emotion_tint(new_emotion)
                self.particle_system.set_emotion_color(self.core.halo_color)

                # Déclencher des effets visuels
                self.visual_effects.trigger_emotion_effect(new_emotion, new_intensity)
        except:
            # En cas d'erreur, revenir à l'état neutre
            self.core.set_emotion("neutral", 0.5)

    def _pulse_energy(self, dt):
        """
        Crée un effet de pulsation d'énergie périodique.

        Args:
            dt: Delta temps
        """
        # Ajouter quelques particules autour du cœur
        x, y = self.core.center
        self.particle_system.add_particles(
            x,
            y,
            count=int(5 + (15 * self.emotion_intensity)),
            velocity_scale=0.5 + self.emotion_intensity,
            lifespan_scale=1.0,
        )

        # Créer une vague d'énergie
        if random.random() < 0.3 + (0.5 * self.emotion_intensity):
            self.core.create_energy_wave("pulse")

    def play_visual_effect(self, effect_name, effect_config):
        """
        Joue un effet visuel spécifique.

        Args:
            effect_name: Nom de l'effet
            effect_config: Configuration de l'effet
        """
        if effect_name == "halo_lumineux":
            self._play_halo_effect(effect_config)
        elif effect_name == "sparkles":
            self._play_sparkles_effect(effect_config)
        elif effect_name == "brume":
            self._play_mist_effect(effect_config)
        elif effect_name == "flash_rouge":
            self._play_flash_effect(effect_config)

    def _play_halo_effect(self, config):
        """Joue un effet de halo lumineux."""
        color = config.get("color", (255, 255, 190))
        r, g, b = color
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        opacity = config.get("opacity", 0.7)

        # Créer une animation de halo
        self.core.halo_color = [r, g, b, opacity]
        anim = Animation(halo_color=[r, g, b, opacity], duration=0.3) + Animation(
            halo_color=[r, g, b, opacity * 0.5], duration=0.7
        )
        anim.start(self.core)

        # Créer une vague d'énergie
        self.core.create_energy_wave("pulse")

    def _play_sparkles_effect(self, config):
        """Joue un effet d'étincelles."""
        color = config.get("color", (255, 255, 255))
        r, g, b = color
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        count = config.get("count", 20)

        # Créer des particules autour du cœur
        x, y = self.core.center
        self.particle_system.add_particles(x, y, count=count, velocity_scale=2.0, lifespan_scale=1.5)

    def _play_mist_effect(self, config):
        """Joue un effet de brume."""
        color = config.get("color", (100, 100, 150))
        r, g, b = color
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        opacity = config.get("opacity", 0.5)

        # Assombrir le fond
        self.dream_background.emotion_tint[3] = 1.5  # Plus sombre
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

        # Ajouter des particules à mouvement lent
        for _ in range(3):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            self.particle_system.add_particles(x, y, count=15, velocity_scale=0.3, lifespan_scale=3.0)

    def _play_flash_effect(self, config):
        """Joue un effet de flash."""
        color = config.get("color", (255, 0, 0))
        r, g, b = color
        r, g, b = r / 255.0, g / 255.0, b / 255.0
        intensity = config.get("intensity", 0.8)

        # Créer un flash sur le cœur
        original_color = self.core.base_color[:]
        flash_color = [r, g, b, 1.0]

        # Animation du flash
        anim = Animation(base_color=flash_color, duration=0.1) + Animation(base_color=original_color, duration=0.3)
        anim.start(self.core)

        # Créer une vague d'énergie intense
        self.core.create_energy_wave("burst")

    def clear_visual_effects(self):
        """Efface tous les effets visuels en cours."""
        # Réinitialiser l'état visuel
        self._update_emotional_state()

    def _on_tap(self, touch_data):
        """
        Gère un tapotement sur l'écran.

        Args:
            touch_data: Données du toucher
        """
        # Vérifier si le tap est sur ou près du cœur
        x, y = touch_data[0], touch_data[1]
        core_x, core_y = self.core.center
        distance = math.sqrt((x - core_x) ** 2 + (y - core_y) ** 2)

        if distance < self.core.width:
            # Tap sur le cœur - effet de bisou
            self.jeffrey_emotions.reagir_au_calin("bisou")
            self.core.react_to_interaction("tap")
            self._play_sparkles_effect({"count": 10})
        else:
            # Tap ailleurs - créer des particules
            self.particle_system.add_particles(x, y, count=5)

    def _on_long_press(self, touch_data):
        """
        Gère une pression longue sur l'écran.

        Args:
            touch_data: Données du toucher
        """
        # Vérifier si la pression est sur ou près du cœur
        x, y = touch_data[0], touch_data[1]
        core_x, core_y = self.core.center
        distance = math.sqrt((x - core_x) ** 2 + (y - core_y) ** 2)

        if distance < self.core.width * 1.5:
            # Pression longue sur le cœur - effet de câlin
            self.jeffrey_emotions.reagir_au_calin("câlin")
            self.core.react_to_interaction("long_press")

            # Effet visuel plus intense
            self._play_halo_effect(
                {
                    "color": [int(c * 255) for c in self.core.base_color[:3]],
                    "opacity": 0.9,
                    "scale": 1.5,
                }
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

        # Calculer la direction
        dx = end_x - start_x
        dy = end_y - start_y

        # Vérifier si le swipe passe par le cœur
        core_x, core_y = self.core.center
        midpoint_x = (start_x + end_x) / 2
        midpoint_y = (start_y + end_y) / 2
        distance_to_core = math.sqrt((midpoint_x - core_x) ** 2 + (midpoint_y - core_y) ** 2)

        if distance_to_core < self.core.width * 1.5:
            # Swipe près du cœur - effet de caresse
            self.jeffrey_emotions.reagir_au_calin("caresse")

            # Ajouter des particules le long du chemin du swipe
            steps = 10
            for i in range(steps):
                t = i / float(steps - 1)
                x = start_x + (dx * t)
                y = start_y + (dy * t)
                self.particle_system.add_particles(x, y, count=2)
        else:
            # Swipe ailleurs - effet d'ondulation du fond
            for i in range(3):
                x = start_x + (dx * (i / 2.0))
                y = start_y + (dy * (i / 2.0))
                self.particle_system.add_particles(x, y, count=3)

    def _on_pinch(self, touch_data):
        """
        Gère un pincement multi-touch.

        Args:
            touch_data: Données du toucher
        """
        center = touch_data["center"]
        scale_factor = touch_data["scale_factor"]

        # Vérifier si le pincement est centré sur le cœur
        core_x, core_y = self.core.center
        distance_to_core = math.sqrt((center[0] - core_x) ** 2 + (center[1] - core_y) ** 2)

        if distance_to_core < self.core.width:
            # Pincement sur le cœur - effet "main posée"
            self.jeffrey_emotions.reagir_au_calin("main_posée")

            # Effet visuel en fonction du type de pincement
            if scale_factor > 1.0:  # Pinch out (écartement)
                self.core.create_energy_wave("burst")
            else:  # Pinch in (rapprochement)
                self._play_halo_effect(
                    {
                        "color": [int(c * 255) for c in self.core.base_color[:3]],
                        "opacity": 0.5,
                        "scale": 0.8,
                    }
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
            # Transition vers l'état de rêve
            Animation(emotion_intensity=0.2, duration=2.0).start(self)
            self.core.set_emotion("peaceful", 0.2)
        else:
            # Revenir à l'état normal
            self._update_emotional_state()

    def on_enter(self):
        """Appelé quand l'écran devient actif."""
        # Mettre à jour l'état émotionnel
        self._update_emotional_state()

        # Déclencher un effet d'entrée
        x, y = self.core.center
        self.particle_system.add_particles(x, y, count=30, velocity_scale=1.5)
        self.core.create_energy_wave("burst")

    def on_leave(self):
        """Appelé quand l'écran n'est plus actif."""
        # Créer un effet de sortie
        self.clear_visual_effects()
