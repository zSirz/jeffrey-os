"""
Effets de traînées lumineuses pour l'interface utilisateur.

Ce module contient des widgets d'effets de traînées lumineuses qui créent
des mouvements fluides et élégants dans l'interface.
"""

import math
import random

from kivy.clock import Clock
from kivy.graphics import Color, Line
from kivy.properties import ColorProperty, NumericProperty
from kivy.uix.widget import Widget


class StarlightTrails(Widget):
    """Widget qui affiche des traînées lumineuses qui suivent des trajectoires fluides."""

    trail_color = ColorProperty([1, 1, 1, 0.7])
    trail_count = NumericProperty(5)

    def __init__(self, **kwargs):
        super(StarlightTrails, self).__init__(**kwargs)
        self.trails = []
        self.bind(size=self.update_trails)
        Clock.schedule_once(self.create_trails, 0.1)
        Clock.schedule_interval(self.update_trail_positions, 1 / 30)

    def create_trails(self, dt):
        self.trails = []
        for i in range(self.trail_count):
            self.add_trail()

    def add_trail(self):
        # Crée une nouvelle traînée avec des propriétés aléatoires
        trail = {
            "points": [],
            "max_points": 15,
            "width": random.uniform(1.5, 3.0),
            "x": random.random() * self.width,
            "y": random.random() * self.height,
            "angle": random.uniform(0, 2 * math.pi),
            "speed": random.uniform(2, 5),
            "turn_rate": random.uniform(-0.1, 0.1),
            "color": [
                self.trail_color[0],
                self.trail_color[1],
                self.trail_color[2],
                self.trail_color[3],
            ],
        }

        # Initialiser la liste des points
        for i in range(trail["max_points"]):
            trail["points"].append((trail["x"], trail["y"]))

        self.trails.append(trail)

    def update_trails(self, *args):
        # Ajustement quand la taille du widget change
        for trail in self.trails:
            trail["x"] = min(max(trail["x"], 0), self.width)
            trail["y"] = min(max(trail["y"], 0), self.height)

            # Réinitialiser les points
            for i in range(len(trail["points"])):
                trail["points"][i] = (trail["x"], trail["y"])

    def update_trail_positions(self, dt):
        # Mettre à jour la position des traînées
        for trail in self.trails:
            # Mettre à jour l'angle avec un petit changement aléatoire pour des mouvements plus naturels
            trail["angle"] += trail["turn_rate"] + random.uniform(-0.05, 0.05)

            # Calculer la nouvelle position
            trail["x"] += math.cos(trail["angle"]) * trail["speed"]
            trail["y"] += math.sin(trail["angle"]) * trail["speed"]

            # Rebondir sur les bords
            if trail["x"] < 0:
                trail["x"] = 0
                trail["angle"] = math.pi - trail["angle"]
            elif trail["x"] > self.width:
                trail["x"] = self.width
                trail["angle"] = math.pi - trail["angle"]

            if trail["y"] < 0:
                trail["y"] = 0
                trail["angle"] = -trail["angle"]
            elif trail["y"] > self.height:
                trail["y"] = self.height
                trail["angle"] = -trail["angle"]

            # Déplacer les points précédents
            for i in range(len(trail["points"]) - 1, 0, -1):
                trail["points"][i] = trail["points"][i - 1]

            # Ajouter la nouvelle position
            trail["points"][0] = (trail["x"], trail["y"])

        # Redessiner
        self.canvas.clear()
        with self.canvas:
            for trail in self.trails:
                points = []
                colors = []

                # Créer les points et couleurs pour le mesh
                for i, point in enumerate(trail["points"]):
                    # Calculer l'opacité qui diminue avec la distance
                    opacity = max(0.1, trail["color"][3] * (1 - i / len(trail["points"])))

                    # Ajouter le point et sa couleur
                    points.append(point[0])
                    points.append(point[1])
                    colors.extend([trail["color"][0], trail["color"][1], trail["color"][2], opacity])

                # Dessiner la ligne avec dégradé d'opacité
                Color(1, 1, 1, 1)  # Couleur de base (sera remplacée par les couleurs des vertex)
                Line(points=points, width=trail["width"], vertex_mode="position")

    def change_emotion(self, emotion):
        """Change les caractéristiques des traînées selon l'émotion."""
        emotion_settings = {
            "joie": {
                "color": [1.0, 0.9, 0.4, 0.7],
                "count": 8,
                "speed_factor": 1.2,
                "turn_rate_factor": 1.5,
            },
            "tristesse": {
                "color": [0.4, 0.6, 0.9, 0.6],
                "count": 3,
                "speed_factor": 0.7,
                "turn_rate_factor": 0.8,
            },
            "colere": {
                "color": [0.9, 0.3, 0.2, 0.7],
                "count": 6,
                "speed_factor": 1.5,
                "turn_rate_factor": 2.0,
            },
            "peur": {
                "color": [0.6, 0.3, 0.8, 0.6],
                "count": 10,
                "speed_factor": 1.3,
                "turn_rate_factor": 1.8,
            },
            "surprise": {
                "color": [0.3, 0.8, 0.9, 0.7],
                "count": 7,
                "speed_factor": 1.4,
                "turn_rate_factor": 1.6,
            },
            "confiance": {
                "color": [0.4, 0.8, 0.6, 0.7],
                "count": 5,
                "speed_factor": 1.0,
                "turn_rate_factor": 0.9,
            },
            "anticipation": {
                "color": [0.9, 0.7, 0.3, 0.7],
                "count": 6,
                "speed_factor": 1.1,
                "turn_rate_factor": 1.2,
            },
            "neutre": {
                "color": [0.8, 0.8, 0.9, 0.5],
                "count": 5,
                "speed_factor": 1.0,
                "turn_rate_factor": 1.0,
            },
        }

        settings = emotion_settings.get(emotion, emotion_settings["neutre"])

        # Appliquer les changements
        self.trail_color = settings["color"]

        # Ajuster le nombre de traînées
        target_count = settings["count"]
        current_count = len(self.trails)

        if current_count < target_count:
            # Ajouter des traînées
            for i in range(current_count, target_count):
                self.add_trail()
        elif current_count > target_count:
            # Supprimer des traînées
            self.trails = self.trails[:target_count]

        # Ajuster la vitesse et le taux de virage des traînées
        speed_factor = settings["speed_factor"]
        turn_rate_factor = settings["turn_rate_factor"]

        for trail in self.trails:
            trail["speed"] = random.uniform(2, 5) * speed_factor
            trail["turn_rate"] = random.uniform(-0.1, 0.1) * turn_rate_factor
            trail["color"] = [
                settings["color"][0],
                settings["color"][1],
                settings["color"][2],
                settings["color"][3],
            ]
