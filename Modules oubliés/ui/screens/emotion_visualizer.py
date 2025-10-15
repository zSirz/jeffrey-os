"""
EmotionVisualizer - Visualisation des émotions sous forme de particules

Ce module crée une visualisation dynamique en temps réel de l'état émotionnel
de Jeffrey en utilisant un système de particules colorées qui réagissent
à l'émotion dominante actuelle.
"""

import math
import random

from kivy.clock import Clock
from kivy.graphics import Color, Ellipse
from kivy.properties import ListProperty, NumericProperty, ObjectProperty
from kivy.uix.widget import Widget


class EmotionVisualizer(Widget):
    """
    Classe qui visualise les émotions sous forme de système de particules dynamiques
    qui s'adaptent à l'état émotionnel dominant de Jeffrey.
    """

    # Propriétés observables pour réagir aux changements d'émotion
    emotion_color = ListProperty([0.3, 0.6, 1.0, 0.7])  # Couleur par défaut (bleu doux)
    particle_speed = NumericProperty(1.0)
    particle_size = NumericProperty(1.0)
    particle_count = NumericProperty(100)
    emotional_state = ObjectProperty(None)
    user_id = "default_user"
    trust_level = 0.0
    warmth_level = 0.0
    proximity_level = 0.0
    is_anchor_user = False
    fatigue_level = 0.0
    emotional_capacity = 1.0

    def __init__(self, **kwargs):
        """Initialise le visualiseur d'émotions."""
        super(EmotionVisualizer, self).__init__(**kwargs)

        # Initialiser le système de particules
        self.particles = []
        self.max_particles = 150

        # Démarrer la boucle de mise à jour
        Clock.schedule_interval(self.update, 1 / 60)

        # Réagir aux changements de taille du widget
        self.bind(size=self._on_size_change, pos=self._on_size_change)

        # Réagir aux changements d'état émotionnel
        self.bind(emotional_state=self._on_emotional_state_change)

        # Initialiser les particules
        self._generate_particles()

    def _on_size_change(self, *args):
        """Réagit aux changements de taille en réinitialisant les particules."""
        self._generate_particles()

    def _on_emotional_state_change(self, *args):
        """Réagit aux changements d'état émotionnel."""
        if self.emotional_state:
            # Mettre à jour la couleur en fonction de l'émotion
            self.set_emotion(self.emotional_state.current, self.emotional_state.intensity)

    def set_emotion(self, emotion, intensity=0.5):
        """
        Définit l'émotion actuelle pour ajuster la visualisation.

        Args:
            emotion: Chaîne représentant l'émotion (happy, sad, etc.)
            intensity: Intensité de l'émotion entre 0.0 et 1.0
        """
        # Convertir l'émotion en couleur
        self.emotion_color = self.get_emotion_color(emotion)

        # Ajuster les paramètres des particules selon l'intensité
        self.particle_speed = 0.5 + (intensity * 1.5)
        self.particle_size = 0.7 + (intensity * 0.6)
        self.particle_count = int(50 + (intensity * 100))

        # Limiter le nombre de particules au maximum
        self.particle_count = min(self.particle_count, self.max_particles)

        # Régénérer les particules avec les nouvelles propriétés
        self._regenerate_particles()

    def get_emotion_color(self, emotion):
        """
        Convertit une émotion en couleur RGBA.

        Args:
            emotion: Chaîne représentant l'émotion

        Returns:
            Tuple (r, g, b, a) représentant la couleur
        """
        emotion_colors = {
            # Tons bleus/violets
            "neutral": [0.3, 0.6, 1.0, 0.7],  # Bleu doux
            "calm": [0.4, 0.6, 0.9, 0.7],  # Bleu calme
            "peaceful": [0.5, 0.7, 0.9, 0.7],  # Bleu clair apaisant
            "relaxed": [0.3, 0.7, 0.8, 0.7],  # Bleu-vert relaxant
            "melancholic": [0.3, 0.4, 0.7, 0.6],  # Bleu foncé mélancolique
            # Tons jaunes/oranges (joie)
            "happy": [1.0, 0.9, 0.3, 0.8],  # Jaune vif
            "excited": [1.0, 0.7, 0.2, 0.8],  # Orange vif
            "joyful": [1.0, 0.8, 0.5, 0.8],  # Jaune-orange joyeux
            # Tons rouges (colère)
            "angry": [0.9, 0.1, 0.1, 0.7],  # Rouge vif
            "frustrated": [0.8, 0.3, 0.2, 0.7],  # Rouge-orange frustré
            "annoyed": [0.7, 0.4, 0.3, 0.7],  # Rouge-brun agacé
            # Tons bleu foncé/gris (tristesse)
            "sad": [0.2, 0.3, 0.6, 0.6],  # Bleu triste
            "disappointed": [0.4, 0.4, 0.5, 0.6],  # Gris-bleu déçu
        }

        # Retourner la couleur correspondante ou neutre par défaut
        return emotion_colors.get(emotion.lower(), [0.3, 0.6, 1.0, 0.7])

    def _generate_particles(self):
        """Génère les particules initiales."""
        self.particles = []

        for _ in range(self.particle_count):
            # Position aléatoire dans les limites du widget
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)

            # Angle et vitesse aléatoires
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.0) * self.particle_speed

            # Calculer les composantes de la vitesse
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed

            # Taille aléatoire
            size = random.uniform(2, 8) * self.particle_size

            # Légère variation de couleur
            color_variation = 0.1
            r, g, b, a = self.emotion_color
            color = (
                max(0, min(1, r + random.uniform(-color_variation, color_variation))),
                max(0, min(1, g + random.uniform(-color_variation, color_variation))),
                max(0, min(1, b + random.uniform(-color_variation, color_variation))),
                max(0, min(1, a + random.uniform(-0.2, 0))),
            )

            # Ajouter la particule
            self.particles.append(
                {
                    "x": x,
                    "y": y,
                    "vx": vx,
                    "vy": vy,
                    "size": size,
                    "color": color,
                    "age": 0,
                    "lifespan": random.uniform(2.0, 5.0),
                }
            )

    def _regenerate_particles(self):
        """Régénère progressivement les particules pour une transition fluide."""
        # Générer de nouvelles particules avec les nouveaux paramètres
        target_count = int(self.particle_count)
        current_count = len(self.particles)

        if target_count > current_count:
            # Ajouter des particules
            for _ in range(target_count - current_count):
                self.add_particle()
        elif target_count < current_count:
            # Marquer des particules pour suppression en réduisant leur durée de vie
            particles_to_remove = current_count - target_count
            for i in range(particles_to_remove):
                if i < len(self.particles):
                    self.particles[i]["lifespan"] = min(self.particles[i]["lifespan"], self.particles[i]["age"] + 0.5)

    def add_particle(self, x=None, y=None):
        """
        Ajoute une nouvelle particule à une position donnée ou aléatoire.

        Args:
            x: Position X (si None, position aléatoire)
            y: Position Y (si None, position aléatoire)
        """
        if x is None:
            x = random.uniform(0, self.width)
        if y is None:
            y = random.uniform(0, self.height)

        # Angle et vitesse aléatoires
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(0.5, 2.0) * self.particle_speed

        # Calculer les composantes de la vitesse
        vx = math.cos(angle) * speed
        vy = math.sin(angle) * speed

        # Taille aléatoire
        size = random.uniform(2, 8) * self.particle_size

        # Légère variation de couleur
        color_variation = 0.1
        r, g, b, a = self.emotion_color
        color = (
            max(0, min(1, r + random.uniform(-color_variation, color_variation))),
            max(0, min(1, g + random.uniform(-color_variation, color_variation))),
            max(0, min(1, b + random.uniform(-color_variation, color_variation))),
            max(0, min(1, a + random.uniform(-0.2, 0))),
        )

        # Créer la particule
        particle = {
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "size": size,
            "color": color,
            "age": 0,
            "lifespan": random.uniform(2.0, 5.0),
        }

        # Ajouter à la liste
        if len(self.particles) < self.max_particles:
            self.particles.append(particle)
        else:
            # Remplacer une particule aléatoire existante
            index = random.randint(0, len(self.particles) - 1)
            self.particles[index] = particle

    def update(self, dt):
        """
        Met à jour l'état et dessine les particules.

        Args:
            dt: Delta temps entre deux mises à jour
        """
        # Vider le canvas
        self.canvas.clear()

        # Liste pour les particules à conserver
        particles_to_keep = []

        # Mettre à jour et dessiner les particules
        with self.canvas:
            for particle in self.particles:
                # Mettre à jour l'âge
                particle["age"] += dt

                # Vérifier si la particule est encore vivante
                if particle["age"] < particle["lifespan"]:
                    # Facteur de vieillissement (1 au début, 0 à la fin)
                    age_factor = 1 - (particle["age"] / particle["lifespan"])

                    # Appliquer le facteur à l'opacité
                    r, g, b, a = particle["color"]
                    current_opacity = a * age_factor

                    # Définir la couleur
                    Color(r, g, b, current_opacity)

                    # Mettre à jour la position
                    particle["x"] += particle["vx"] * dt
                    particle["y"] += particle["vy"] * dt

                    # Gérer les rebonds sur les bords
                    if particle["x"] < 0:
                        particle["x"] = 0
                        particle["vx"] *= -0.8
                    elif particle["x"] > self.width:
                        particle["x"] = self.width
                        particle["vx"] *= -0.8

                    if particle["y"] < 0:
                        particle["y"] = 0
                        particle["vy"] *= -0.8
                    elif particle["y"] > self.height:
                        particle["y"] = self.height
                        particle["vy"] *= -0.8

                    # Appliquer une légère gravité
                    particle["vy"] -= 0.03 * dt

                    # Dessiner la particule
                    size = particle["size"] * age_factor
                    Ellipse(pos=(particle["x"] - size / 2, particle["y"] - size / 2), size=(size, size))

                    # Conserver cette particule
                    particles_to_keep.append(particle)

        # Mettre à jour la liste des particules
        self.particles = particles_to_keep

        # Ajouter de nouvelles particules pour maintenir le nombre souhaité
        while len(self.particles) < self.particle_count:
            self.add_particle()

    def add_burst(self, x, y, count=20, energy=1.0):
        """
        Ajoute une explosion de particules à un point donné.

        Args:
            x: Position X de l'explosion
            y: Position Y de l'explosion
            count: Nombre de particules à créer
            energy: Niveau d'énergie (affecte la vitesse et la taille)
        """
        for _ in range(count):
            # Angle aléatoire pour le mouvement
            angle = random.uniform(0, 2 * math.pi)

            # Vitesse basée sur l'énergie
            speed = random.uniform(1.0, 3.0) * energy * self.particle_speed

            # Calculer les composantes de la vitesse
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed

            # Taille basée sur l'énergie
            size = random.uniform(3, 10) * energy * self.particle_size

            # Durée de vie plus courte pour les explosions
            lifespan = random.uniform(0.5, 1.5)

            # Variation de couleur plus importante
            color_variation = 0.2
            r, g, b, a = self.emotion_color
            color = (
                max(0, min(1, r + random.uniform(-color_variation, color_variation))),
                max(0, min(1, g + random.uniform(-color_variation, color_variation))),
                max(0, min(1, b + random.uniform(-color_variation, color_variation))),
                min(1.0, a + 0.2),  # Plus opaque
            )

            # Créer la particule
            particle = {
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "size": size,
                "color": color,
                "age": 0,
                "lifespan": lifespan,
            }

            # Ajouter à la liste
            if len(self.particles) < self.max_particles:
                self.particles.append(particle)
            else:
                # Remplacer une particule aléatoire existante
                index = random.randint(0, len(self.particles) - 1)
                self.particles[index] = particle

    def display_affective_link(self):
        """
        Affiche un résumé textuel du lien affectif actuel.
        """
        bond = self.trust_level * 0.35 + self.warmth_level * 0.35 + self.proximity_level * 0.3

        print(f"\n💞 Lien affectif avec {self.user_id}")
        print(f"  Confiance :  {self.trust_level:.2f}")
        print(f"  Chaleur :    {self.warmth_level:.2f}")
        print(f"  Proximité :  {self.proximity_level:.2f}")
        print(f"  Lien global : {bond:.2f}")
        print(f"  Fatigue :    {self.fatigue_level:.2f}")
        print(f"  Capacité émotionnelle : {self.emotional_capacity:.2f}")
        if self.is_anchor_user:
            print("  ⚓ Utilisateur d’ancrage principal")
