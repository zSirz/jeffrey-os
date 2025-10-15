"""
Effets atmosphériques pour l'interface utilisateur.

Ce module contient des widgets d'effets atmosphériques qui peuvent être
utilisés pour créer des ambiances visuelles liées aux émotions.
"""

import math
import random

from kivy.clock import Clock
from kivy.graphics import Color, RoundedRectangle
from kivy.properties import ColorProperty, NumericProperty
from kivy.uix.widget import Widget


class EmotionalMist(Widget):
    """Widget qui affiche un effet de brume dont la densité et la couleur varient selon l'émotion."""

    color = ColorProperty([1, 1, 1, 0.2])
    particle_count = NumericProperty(30)

    def __init__(self, **kwargs):
        super(EmotionalMist, self).__init__(**kwargs)
        self.particles = []
        self.bind(size=self.update_mist)
        Clock.schedule_once(self.create_particles, 0.1)
        Clock.schedule_interval(self.update_particles, 1 / 30)

    def create_particles(self, dt):
        self.particles = []
        for i in range(self.particle_count):
            self.add_particle()

    def add_particle(self):
        particle = {
            "x": random.random() * self.width,
            "y": random.random() * self.height,
            "size_x": random.uniform(50, 150),
            "size_y": random.uniform(30, 80),
            "opacity": random.uniform(0.05, 0.2),
            "speed": random.uniform(2, 8),
            "direction": random.uniform(-0.2, 0.2),
        }
        self.particles.append(particle)

    def update_mist(self, *args):
        # Ajustement quand la taille du widget change
        for particle in self.particles:
            particle["x"] = min(max(particle["x"], -particle["size_x"]), self.width)
            particle["y"] = min(max(particle["y"], -particle["size_y"]), self.height)

    def update_particles(self, dt):
        # Mettre à jour la position des particules
        for particle in self.particles:
            # Mouvement
            particle["x"] += particle["speed"] * dt * math.cos(particle["direction"])
            particle["y"] += particle["speed"] * dt * math.sin(particle["direction"])

            # Recycler les particules qui sortent de l'écran
            if particle["x"] > self.width:
                particle["x"] = -particle["size_x"]
                particle["y"] = random.random() * self.height

            # Variation légère d'opacité
            particle["opacity"] = max(0.05, min(0.2, particle["opacity"] + random.uniform(-0.01, 0.01)))

        # Redessiner
        self.canvas.clear()
        with self.canvas:
            for particle in self.particles:
                Color(self.color[0], self.color[1], self.color[2], particle["opacity"])
                RoundedRectangle(
                    pos=(particle["x"], particle["y"]),
                    size=(particle["size_x"], particle["size_y"]),
                    radius=[
                        10,
                    ],
                )

    def change_emotion(self, emotion):
        """Change les caractéristiques de la brume selon l'émotion."""
        emotion_settings = {
            "joie": {"color": [1.0, 0.9, 0.6, 0.15], "count": 40, "speed_factor": 1.2},
            "tristesse": {"color": [0.5, 0.6, 0.9, 0.2], "count": 50, "speed_factor": 0.7},
            "colere": {"color": [0.9, 0.3, 0.3, 0.15], "count": 35, "speed_factor": 1.5},
            "peur": {"color": [0.5, 0.3, 0.6, 0.2], "count": 60, "speed_factor": 1.3},
            "surprise": {"color": [0.7, 0.9, 1.0, 0.15], "count": 45, "speed_factor": 1.1},
            "confiance": {"color": [0.6, 0.9, 0.7, 0.15], "count": 30, "speed_factor": 0.9},
            "anticipation": {"color": [0.9, 0.7, 0.5, 0.15], "count": 40, "speed_factor": 1.0},
            "neutre": {"color": [0.8, 0.8, 0.8, 0.1], "count": 30, "speed_factor": 1.0},
        }

        settings = emotion_settings.get(emotion, emotion_settings["neutre"])

        # Appliquer les changements
        self.color = settings["color"]

        # Ajuster le nombre de particules
        target_count = settings["count"]
        current_count = len(self.particles)

        if current_count < target_count:
            # Ajouter des particules
            for i in range(current_count, target_count):
                self.add_particle()
        elif current_count > target_count:
            # Supprimer des particules
            self.particles = self.particles[:target_count]

        # Ajuster la vitesse des particules
        speed_factor = settings["speed_factor"]
        for particle in self.particles:
            particle["speed"] = random.uniform(2, 8) * speed_factor
