"""
Widgets émotionnels pour l'interface utilisateur de Jeffrey.

Ce module contient des widgets visuels représentant différents aspects
des émotions et réactions de Jeffrey.
"""

import math
import random

from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import Color, Ellipse, Line
from kivy.properties import ColorProperty, ListProperty, NumericProperty
from kivy.uix.widget import Widget


class EmotionalLightPulse(Widget):
    """Widget qui affiche une pulsation lumineuse représentant l'état émotionnel."""

    opacity = NumericProperty(0.8)
    color = ColorProperty([1, 1, 1, 1])
    pulse_size = NumericProperty(1.0)

    def __init__(self, **kwargs):
        super(EmotionalLightPulse, self).__init__(**kwargs)
        self.pulse_anim = None
        self.bind(size=self.update_pulse, pos=self.update_pulse)
        Clock.schedule_once(self.start_pulsing, 0.1)

    def update_pulse(self, *args):
        self.canvas.clear()
        with self.canvas:
            Color(*self.color, mode="rgba")
            Ellipse(pos=self.pos, size=(self.width * self.pulse_size, self.height * self.pulse_size))

    def start_pulsing(self, dt):
        self.pulse_anim = Animation(pulse_size=0.8, opacity=0.4, duration=1.5) + Animation(
            pulse_size=1.0, opacity=0.8, duration=1.5
        )
        self.pulse_anim.repeat = True
        self.pulse_anim.start(self)

    def change_emotion(self, emotion):
        """Change la couleur du pulse selon l'émotion."""
        emotion_colors = {
            "joie": [1, 0.9, 0.2, 0.8],
            "tristesse": [0.2, 0.5, 0.9, 0.8],
            "colere": [0.9, 0.2, 0.2, 0.8],
            "peur": [0.7, 0.3, 0.9, 0.8],
            "surprise": [0.3, 0.9, 0.9, 0.8],
            "degout": [0.5, 0.8, 0.2, 0.8],
            "confiance": [0.3, 0.8, 0.5, 0.8],
            "anticipation": [0.9, 0.6, 0.3, 0.8],
        }
        self.color = emotion_colors.get(emotion, [1, 1, 1, 0.8])
        self.update_pulse()


class HeartBeatPulse(Widget):
    """Widget qui affiche une pulsation cardiaque émotionnelle."""

    line_points = ListProperty([])
    line_color = ColorProperty([0.9, 0.2, 0.3, 0.8])
    beat_frequency = NumericProperty(1.0)  # Battements par seconde

    def __init__(self, **kwargs):
        super(HeartBeatPulse, self).__init__(**kwargs)
        self.time = 0
        self.bind(size=self.update_points)
        Clock.schedule_interval(self.update_heartbeat, 1 / 30)

    def update_points(self, *args):
        self.time = 0
        self.line_points = []

    def update_heartbeat(self, dt):
        self.time += dt

        # Générer les points de la ligne
        points = []
        width = self.width
        height = self.height
        center_y = height / 2

        for i in range(int(width)):
            x = i
            # Fonction sinusoïdale avec un pic périodique
            t = (i / width + self.time * self.beat_frequency) % 1.0

            if 0.1 < t < 0.2:
                # Premier pic du battement
                y = center_y + math.sin(t * 20) * (height * 0.4)
            elif 0.2 < t < 0.25:
                # Creux entre les pics
                y = center_y - (height * 0.2)
            elif 0.25 < t < 0.35:
                # Second pic du battement
                y = center_y + math.sin((t - 0.25) * 10) * (height * 0.3)
            else:
                # Ligne de base
                y = center_y + math.sin(t * 2) * (height * 0.05)

            points.extend([x, y])

        self.line_points = points

        # Redessiner
        self.canvas.clear()
        with self.canvas:
            Color(*self.line_color)
            Line(points=self.line_points, width=2)

    def change_emotion(self, emotion):
        """Change les caractéristiques du battement selon l'émotion."""
        emotion_settings = {
            "joie": {"color": [0.9, 0.7, 0.3, 0.8], "frequency": 1.2},
            "tristesse": {"color": [0.3, 0.5, 0.9, 0.8], "frequency": 0.7},
            "colere": {"color": [0.9, 0.2, 0.2, 0.8], "frequency": 1.5},
            "peur": {"color": [0.7, 0.3, 0.9, 0.8], "frequency": 1.8},
            "surprise": {"color": [0.3, 0.9, 0.9, 0.8], "frequency": 1.4},
            "degout": {"color": [0.5, 0.8, 0.2, 0.8], "frequency": 0.9},
            "confiance": {"color": [0.3, 0.8, 0.5, 0.8], "frequency": 1.0},
            "anticipation": {"color": [0.9, 0.6, 0.3, 0.8], "frequency": 1.1},
        }

        settings = emotion_settings.get(emotion, {"color": [0.9, 0.2, 0.3, 0.8], "frequency": 1.0})
        self.line_color = settings["color"]
        self.beat_frequency = settings["frequency"]


class FireflyField(Widget):
    """Widget qui affiche un champ de lucioles dont le comportement varie selon l'émotion."""

    def __init__(self, **kwargs):
        super(FireflyField, self).__init__(**kwargs)
        self.fireflies = []
        self.emotion = "neutre"
        self.max_fireflies = 20
        self.bind(size=self.update_field)
        Clock.schedule_once(self.create_fireflies, 0.1)
        Clock.schedule_interval(self.update_fireflies, 1 / 30)

    def create_fireflies(self, dt):
        self.fireflies = []
        for i in range(self.max_fireflies):
            self.add_firefly()

    def add_firefly(self):
        firefly = {
            "x": random.random() * self.width,
            "y": random.random() * self.height,
            "size": random.uniform(3, 8),
            "color": [
                random.uniform(0.7, 1.0),
                random.uniform(0.7, 1.0),
                random.uniform(0.2, 0.5),
                random.uniform(0.5, 0.9),
            ],
            "speed": random.uniform(0.5, 2.0),
            "direction": random.uniform(0, 2 * math.pi),
            "pulse": random.uniform(0, 1.0),
            "pulse_speed": random.uniform(0.5, 1.5),
        }
        self.fireflies.append(firefly)

    def update_field(self, *args):
        # Ajustement quand la taille du widget change
        for fly in self.fireflies:
            fly["x"] = min(max(fly["x"], 0), self.width)
            fly["y"] = min(max(fly["y"], 0), self.height)

    def update_fireflies(self, dt):
        # Mettre à jour la position et la luminosité des lucioles
        for fly in self.fireflies:
            # Mise à jour de la pulsation
            fly["pulse"] = (fly["pulse"] + dt * fly["pulse_speed"]) % 1.0
            brightness = 0.5 + 0.5 * math.sin(fly["pulse"] * 2 * math.pi)

            # Mise à jour de la position
            fly["x"] += math.cos(fly["direction"]) * fly["speed"] * dt * 10
            fly["y"] += math.sin(fly["direction"]) * fly["speed"] * dt * 10

            # Rebondir sur les bords
            if fly["x"] < 0 or fly["x"] > self.width:
                fly["direction"] = math.pi - fly["direction"]
            if fly["y"] < 0 or fly["y"] > self.height:
                fly["direction"] = -fly["direction"]

            # Changement aléatoire de direction
            if random.random() < 0.01:
                fly["direction"] += random.uniform(-0.5, 0.5)

        # Redessiner
        self.canvas.clear()
        with self.canvas:
            for fly in self.fireflies:
                brightness = 0.5 + 0.5 * math.sin(fly["pulse"] * 2 * math.pi)
                color = fly["color"].copy()
                color[3] = color[3] * brightness
                Color(*color)
                size = fly["size"] * (0.8 + 0.4 * brightness)
                Ellipse(pos=(fly["x"] - size / 2, fly["y"] - size / 2), size=(size, size))

    def change_emotion(self, emotion):
        """Change le comportement des lucioles selon l'émotion."""
        self.emotion = emotion
        emotion_settings = {
            "joie": {
                "color_base": [1.0, 0.9, 0.2],
                "speed_factor": 1.5,
                "size_factor": 1.2,
                "count_factor": 1.2,
            },
            "tristesse": {
                "color_base": [0.2, 0.5, 0.9],
                "speed_factor": 0.7,
                "size_factor": 0.9,
                "count_factor": 0.8,
            },
            "colere": {
                "color_base": [0.9, 0.3, 0.2],
                "speed_factor": 1.8,
                "size_factor": 1.3,
                "count_factor": 1.0,
            },
            "peur": {
                "color_base": [0.7, 0.3, 0.9],
                "speed_factor": 2.0,
                "size_factor": 0.8,
                "count_factor": 1.5,
            },
            "surprise": {
                "color_base": [0.3, 0.9, 0.9],
                "speed_factor": 1.6,
                "size_factor": 1.1,
                "count_factor": 1.3,
            },
            "confiance": {
                "color_base": [0.3, 0.8, 0.5],
                "speed_factor": 0.9,
                "size_factor": 1.0,
                "count_factor": 1.1,
            },
            "anticipation": {
                "color_base": [0.9, 0.6, 0.3],
                "speed_factor": 1.2,
                "size_factor": 1.0,
                "count_factor": 1.0,
            },
        }

        settings = emotion_settings.get(
            emotion,
            {
                "color_base": [0.8, 0.8, 0.5],
                "speed_factor": 1.0,
                "size_factor": 1.0,
                "count_factor": 1.0,
            },
        )

        # Ajuster les lucioles existantes
        for fly in self.fireflies:
            # Ajuster la couleur
            color_base = settings["color_base"]
            fly["color"] = [
                color_base[0] * random.uniform(0.9, 1.1),
                color_base[1] * random.uniform(0.9, 1.1),
                color_base[2] * random.uniform(0.9, 1.1),
                random.uniform(0.6, 0.9),
            ]

            # Ajuster la vitesse
            fly["speed"] = random.uniform(0.5, 2.0) * settings["speed_factor"]

            # Ajuster la taille
            fly["size"] = random.uniform(3, 8) * settings["size_factor"]

            # Ajuster la vitesse de pulsation
            fly["pulse_speed"] = random.uniform(0.5, 1.5)

        # Ajuster le nombre de lucioles
        target_count = int(self.max_fireflies * settings["count_factor"])
        current_count = len(self.fireflies)

        if current_count < target_count:
            # Ajouter des lucioles
            for i in range(current_count, target_count):
                self.add_firefly()
        elif current_count > target_count:
            # Supprimer des lucioles
            self.fireflies = self.fireflies[:target_count]


class VoiceWaveVisualizer(Widget):
    """Widget qui visualise les ondes sonores de la voix."""

    line_points = ListProperty([])
    line_color = ColorProperty([0.2, 0.6, 0.9, 0.8])

    def __init__(self, **kwargs):
        super(VoiceWaveVisualizer, self).__init__(**kwargs)
        self.amplitude = 0.5
        self.frequency = 1.0
        self.time = 0
        self.bind(size=self.update_points)
        self.active = False

    def update_points(self, *args):
        self.generate_wave()

    def generate_wave(self):
        if not self.active:
            self.line_points = []
            return

        points = []
        width = self.width
        height = self.height
        center_y = height / 2

        for i in range(int(width)):
            x = i
            # Fonction sinusoïdale avec variations aléatoires
            y = center_y + math.sin(i * 0.05 * self.frequency + self.time) * (height * 0.4 * self.amplitude)

            # Ajouter des micro-variations pour un effet plus naturel
            if i % 3 == 0:  # Un point sur trois
                y += random.uniform(-5, 5)

            points.extend([x, y])

        self.line_points = points

        # Redessiner
        self.canvas.clear()
        with self.canvas:
            Color(*self.line_color)
            Line(points=self.line_points, width=1.5)

    def activate(self, active=True):
        """Active ou désactive la visualisation des ondes sonores."""
        self.active = active
        if active:
            Clock.schedule_interval(self.update_wave, 1 / 30)
        else:
            Clock.unschedule(self.update_wave)
            self.line_points = []
            self.canvas.clear()

    def update_wave(self, dt):
        self.time += dt * 10
        self.amplitude = 0.3 + 0.7 * random.random()  # Variation d'amplitude
        self.generate_wave()

    def change_emotion(self, emotion):
        """Change les caractéristiques de l'onde selon l'émotion."""
        emotion_settings = {
            "joie": {"color": [0.9, 0.7, 0.3, 0.8], "frequency": 1.2},
            "tristesse": {"color": [0.3, 0.5, 0.9, 0.8], "frequency": 0.7},
            "colere": {"color": [0.9, 0.2, 0.2, 0.8], "frequency": 1.5},
            "peur": {"color": [0.7, 0.3, 0.9, 0.8], "frequency": 1.8},
            "surprise": {"color": [0.3, 0.9, 0.9, 0.8], "frequency": 1.4},
            "degout": {"color": [0.5, 0.8, 0.2, 0.8], "frequency": 0.9},
            "confiance": {"color": [0.3, 0.8, 0.5, 0.8], "frequency": 1.0},
            "anticipation": {"color": [0.9, 0.6, 0.3, 0.8], "frequency": 1.1},
        }

        settings = emotion_settings.get(emotion, {"color": [0.2, 0.6, 0.9, 0.8], "frequency": 1.0})
        self.line_color = settings["color"]
        self.frequency = settings["frequency"]


class EmotionalNotificationHalo(Widget):
    """Widget qui affiche un halo de notification émotionnelle."""

    color = ColorProperty([0.9, 0.9, 0.9, 0])
    size_hint = ListProperty([1, 1])

    def __init__(self, **kwargs):
        super(EmotionalNotificationHalo, self).__init__(**kwargs)
        self.bind(size=self.update_halo, pos=self.update_halo)

    def update_halo(self, *args):
        self.canvas.clear()
        with self.canvas:
            Color(*self.color)
            Ellipse(pos=self.pos, size=self.size)

    def pulse(self, emotion="neutre"):
        """Affiche une pulsation du halo selon l'émotion spécifiée."""
        emotion_colors = {
            "joie": [1, 0.9, 0.2, 0],
            "tristesse": [0.2, 0.5, 0.9, 0],
            "colere": [0.9, 0.2, 0.2, 0],
            "peur": [0.7, 0.3, 0.9, 0],
            "surprise": [0.3, 0.9, 0.9, 0],
            "degout": [0.5, 0.8, 0.2, 0],
            "confiance": [0.3, 0.8, 0.5, 0],
            "anticipation": [0.9, 0.6, 0.3, 0],
            "neutre": [0.9, 0.9, 0.9, 0],
        }

        self.color = emotion_colors.get(emotion, [0.9, 0.9, 0.9, 0])

        # Animation de pulsation
        anim = Animation(color=[self.color[0], self.color[1], self.color[2], 0.7], d=0.3) + Animation(
            color=[self.color[0], self.color[1], self.color[2], 0], d=0.7
        )

        anim.start(self)
