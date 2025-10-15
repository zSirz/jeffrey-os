#!/usr/bin/env python
"""
emotional_memory_visualizer.py - Visualisation de la mémoire émotionnelle

Ce module fournit des outils de visualisation pour représenter graphiquement
la mémoire émotionnelle de Jeffrey. Il permet d'afficher des chronologies
d'émotions, des distributions et des représentations faciales animées.

Sprint 12: Implémentation de visualisations avancées pour l'analyse émotionnelle
"""

import logging
from datetime import datetime, timedelta

try:
    import matplotlib

    matplotlib.use("Agg")  # Backend non-interactif
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    logging.warning("matplotlib non disponible. Fonctionnalités de visualisation réduites.")
    HAS_MATPLOTLIB = False

try:
    from kivy.animation import Animation
    from kivy.clock import Clock
    from kivy.graphics import Color, Ellipse, Line, Rectangle
    from kivy.properties import BooleanProperty, ListProperty, NumericProperty, ObjectProperty, StringProperty
    from kivy.uix.widget import Widget

    HAS_KIVY = True
except ImportError:
    logging.warning("kivy non disponible. Fonctionnalités d'animation réduites.")
    HAS_KIVY = False


class EmotionalMemoryVisualizer:
    """
    Classe qui génère différentes visualisations de la mémoire émotionnelle.
    Peut fonctionner en mode console ou avec des capacités graphiques selon les dépendances.
    """

    def __init__(self, emotional_memory=None, output_dir="output/visualizations"):
        """
        Initialise le visualisateur de mémoire émotionnelle.

        Args:
            emotional_memory: Instance de EmotionalMemory à visualiser
            output_dir: Répertoire de sortie pour les visualisations générées
        """
        self.logger = logging.getLogger("jeffrey.emotional_memory_visualizer")
        self.emotional_memory = emotional_memory
        self.output_dir = output_dir
        self.last_update = datetime.now()

    def set_emotional_memory(self, emotional_memory):
        """
        Définit la source de mémoire émotionnelle à visualiser.

        Args:
            emotional_memory: Instance de EmotionalMemory
        """
        self.emotional_memory = emotional_memory
        self.logger.info("Source de mémoire émotionnelle mise à jour")

    def create_emotion_timeline(self, days_back: int = 30, output_file: str | None = None) -> str | None:
        """
        Crée une visualisation chronologique des émotions sur une période donnée.

        Args:
            days_back: Nombre de jours à inclure dans la chronologie
            output_file: Chemin du fichier de sortie (PNG)

        Returns:
            Chemin du fichier généré ou None si échec
        """
        if not HAS_MATPLOTLIB or not self.emotional_memory:
            self.logger.warning("Impossible de créer la chronologie: matplotlib manquant ou mémoire non définie")
            return None

        try:
            memories = self.emotional_memory.data["memories"]

            if not memories:
                self.logger.warning("Aucun souvenir trouvé pour générer la chronologie")
                return None

            # Filtrer les souvenirs dans la plage de temps demandée
            start_date = datetime.now() - timedelta(days=days_back)
            filtered_memories = []

            for memory in memories:
                memory_date = datetime.fromisoformat(memory["timestamp"].split("T")[0])
                if memory_date >= start_date:
                    filtered_memories.append(memory)

            if not filtered_memories:
                self.logger.warning(f"Aucun souvenir dans les {days_back} derniers jours")
                return None

            # Préparation des données pour le graphique
            dates = [datetime.fromisoformat(m["timestamp"]) for m in filtered_memories]
            emotions = [m["emotion"] for m in filtered_memories]
            intensities = [m["intensity"] for m in filtered_memories]

            # Créer un mapping des émotions vers des couleurs
            unique_emotions = list(set(emotions))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_emotions)))
            emotion_colors = {emotion: colors[i] for i, emotion in enumerate(unique_emotions)}

            # Créer la figure
            fig, ax = plt.subplots(figsize=(12, 6))

            # Tracer les points avec couleur selon l'émotion et taille selon l'intensité
            for date, emotion, intensity in zip(dates, emotions, intensities):
                ax.scatter(date, emotion, color=emotion_colors[emotion], s=intensity * 100, alpha=0.7)

            # Personnaliser le graphique
            ax.set_title("Chronologie des émotions")
            ax.set_xlabel("Date")
            ax.set_ylabel("Émotion")

            # Configurer l'axe des x pour afficher les dates correctement
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m/%Y"))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days_back // 10)))
            plt.xticks(rotation=45)

            # Ajouter une légende
            for emotion, color in emotion_colors.items():
                ax.scatter([], [], color=color, label=emotion)
            ax.legend(title="Émotions", loc="upper right")

            # Ajuster la mise en page
            plt.tight_layout()

            # Enregistrer ou afficher
            if output_file:
                plt.savefig(output_file, dpi=150)
                self.logger.info(f"Chronologie des émotions enregistrée dans {output_file}")
                plt.close(fig)
                return output_file
            else:
                # Générer un nom de fichier par défaut
                default_file = f"{self.output_dir}/emotion_timeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(default_file, dpi=150)
                self.logger.info(f"Chronologie des émotions enregistrée dans {default_file}")
                plt.close(fig)
                return default_file

        except Exception as e:
            self.logger.error(f"Erreur lors de la création de la chronologie des émotions: {e}")
            if "fig" in locals():
                plt.close(fig)
            return None

    def create_emotion_distribution(self, output_file: str | None = None) -> str | None:
        """
        Crée une visualisation de la distribution des émotions dans la mémoire.

        Args:
            output_file: Chemin du fichier de sortie (PNG)

        Returns:
            Chemin du fichier généré ou None si échec
        """
        if not HAS_MATPLOTLIB or not self.emotional_memory:
            self.logger.warning("Impossible de créer la distribution: matplotlib manquant ou mémoire non définie")
            return None

        try:
            # Obtenir les statistiques des émotions
            if hasattr(self.emotional_memory, "get_emotional_stats"):
                stats = self.emotional_memory.get_emotional_stats()
                emotion_counts = stats.get("most_frequent", {})
            else:
                emotion_counts = self.emotional_memory.data["stats"]["emotion_counts"]

            if not emotion_counts:
                self.logger.warning("Aucune donnée pour générer la distribution")
                return None

            # Préparation des données
            emotions = list(emotion_counts.keys())
            counts = list(emotion_counts.values())

            # Créer la figure
            fig, ax = plt.subplots(figsize=(10, 8))

            # Créer un graphique camembert si moins de 8 émotions,
            # sinon utiliser un histogramme horizontal
            if len(emotions) <= 8:
                # Graphique camembert
                wedges, texts, autotexts = ax.pie(
                    counts,
                    labels=emotions,
                    autopct="%1.1f%%",
                    startangle=90,
                    shadow=True,
                    explode=[0.05] * len(emotions),
                    colors=plt.cm.tab10(np.linspace(0, 1, len(emotions))),
                )

                # Personnaliser l'apparence
                plt.setp(autotexts, size=10, weight="bold")
                ax.set_title("Distribution des émotions")

            else:
                # Histogramme horizontal pour beaucoup d'émotions
                y_pos = np.arange(len(emotions))
                ax.barh(
                    y_pos,
                    counts,
                    align="center",
                    color=plt.cm.tab10(np.linspace(0, 1, len(emotions))),
                )
                ax.set_yticks(y_pos)
                ax.set_yticklabels(emotions)
                ax.invert_yaxis()  # Les entrées les plus élevées en haut
                ax.set_xlabel("Nombre d'occurrences")
                ax.set_title("Distribution des émotions")

            # Ajuster la mise en page
            plt.tight_layout()

            # Enregistrer ou afficher
            if output_file:
                plt.savefig(output_file, dpi=150)
                self.logger.info(f"Distribution des émotions enregistrée dans {output_file}")
                plt.close(fig)
                return output_file
            else:
                # Générer un nom de fichier par défaut
                default_file = f"{self.output_dir}/emotion_distribution_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                plt.savefig(default_file, dpi=150)
                self.logger.info(f"Distribution des émotions enregistrée dans {default_file}")
                plt.close(fig)
                return default_file

        except Exception as e:
            self.logger.error(f"Erreur lors de la création de la distribution des émotions: {e}")
            if "fig" in locals():
                plt.close(fig)
            return None

    def create_emotional_face(
        self, emotion: str = None, intensity: float = 0.7, output_file: str | None = None
    ) -> str | None:
        """
        Crée une représentation faciale simplifiée basée sur l'émotion dominante.

        Args:
            emotion: Émotion à représenter (ou None pour utiliser la dominante dans la mémoire)
            intensity: Intensité de l'émotion (0.0 à 1.0)
            output_file: Chemin du fichier de sortie (PNG)

        Returns:
            Chemin du fichier généré ou None si échec
        """
        if not HAS_MATPLOTLIB:
            self.logger.warning("Impossible de créer le visage émotionnel: matplotlib manquant")
            return None

        try:
            # Si aucune émotion n'est spécifiée, utiliser la plus fréquente dans la mémoire
            if emotion is None and self.emotional_memory:
                if hasattr(self.emotional_memory, "get_emotional_stats"):
                    stats = self.emotional_memory.get_emotional_stats()
                    emotions = stats.get("most_frequent", {})
                    if emotions:
                        emotion = list(emotions.keys())[0]
                        intensity = min(1.0, max(0.3, stats["intensity_stats"]["avg"]))
                else:
                    emotion_counts = self.emotional_memory.data["stats"]["emotion_counts"]
                    if emotion_counts:
                        emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]

            # Utiliser une émotion par défaut si toujours non définie
            if not emotion:
                emotion = "neutral"

            # Créer la figure
            fig, ax = plt.subplots(figsize=(6, 6))

            # Désactiver les axes
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 10)
            ax.axis("off")

            # Dessiner le visage (cercle)
            face = plt.Circle((5, 5), 4, fill=True, color="#FFD700", alpha=0.7)
            ax.add_patch(face)

            # Dessiner les yeux
            eye_l = plt.Circle((3.5, 6), 0.6, fill=True, color="#4169E1")
            eye_r = plt.Circle((6.5, 6), 0.6, fill=True, color="#4169E1")
            ax.add_patch(eye_l)
            ax.add_patch(eye_r)

            # Dessiner la bouche selon l'émotion
            if emotion.lower() in ["happy", "excited", "joyful"]:
                # Bouche souriante
                arc = np.linspace(3 * np.pi / 4, 7 * np.pi / 4, 100)
                x = 5 + 2.5 * np.cos(arc)
                y = 5 + 2.5 * np.sin(arc) * intensity
                plt.plot(x, y, "k-", linewidth=3)

            elif emotion.lower() in ["sad", "disappointed", "melancholic"]:
                # Bouche triste
                arc = np.linspace(np.pi / 4, 3 * np.pi / 4, 100)
                x = 5 + 2.5 * np.cos(arc)
                y = 3 + 2.5 * np.sin(arc) * intensity
                plt.plot(x, y, "k-", linewidth=3)

            elif emotion.lower() in ["angry", "frustrated", "annoyed"]:
                # Bouche en colère (ligne droite inclinée vers le bas)
                plt.plot([3, 7], [4, 3.7], "k-", linewidth=3)
                # Sourcils froncés
                plt.plot([2.5, 4.5], [7, 6.5], "k-", linewidth=2)
                plt.plot([5.5, 7.5], [6.5, 7], "k-", linewidth=2)

            elif emotion.lower() in ["calm", "peaceful", "relaxed"]:
                # Bouche légèrement souriante
                arc = np.linspace(3 * np.pi / 4, 7 * np.pi / 4, 100)
                x = 5 + 2.5 * np.cos(arc)
                y = 5 + 1.5 * np.sin(arc) * intensity
                plt.plot(x, y, "k-", linewidth=2)

            else:  # neutral ou autre
                # Bouche neutre
                plt.plot([3.5, 6.5], [4, 4], "k-", linewidth=2)

            # Ajouter le titre avec l'émotion
            plt.title(f"Émotion: {emotion.capitalize()} ({intensity:.2f})")

            # Ajuster la mise en page
            plt.tight_layout()

            # Enregistrer ou afficher
            if output_file:
                plt.savefig(output_file, dpi=150)
                self.logger.info(f"Visage émotionnel enregistré dans {output_file}")
                plt.close(fig)
                return output_file
            else:
                # Générer un nom de fichier par défaut
                default_file = (
                    f"{self.output_dir}/emotional_face_{emotion}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plt.savefig(default_file, dpi=150)
                self.logger.info(f"Visage émotionnel enregistré dans {default_file}")
                plt.close(fig)
                return default_file

        except Exception as e:
            self.logger.error(f"Erreur lors de la création du visage émotionnel: {e}")
            if "fig" in locals():
                plt.close(fig)
            return None


if HAS_KIVY:

    class AnimatedEmotionalFace(Widget):
        """
        Widget Kivy qui affiche un visage animé reflétant l'état émotionnel.
        Utilisé pour la visualisation interactive.
        """

        emotion = StringProperty("neutral")
        intensity = NumericProperty(0.5)
        face_color = ListProperty([1, 0.85, 0.2, 0.8])  # Jaune doux
        eye_color = ListProperty([0.25, 0.41, 0.88, 1])  # Bleu
        mouth_color = ListProperty([0, 0, 0, 1])  # Noir
        animated = BooleanProperty(True)

        def __init__(self, **kwargs):
            super(AnimatedEmotionalFace, self).__init__(**kwargs)
            self.eye_scale = 1.0
            self.blink_event = None

            # Démarrer les animations
            if self.animated:
                self.blink_event = Clock.schedule_interval(self._blink_eyes, 5)
                Clock.schedule_interval(self._update_face, 1 / 30)

        def on_emotion(self, instance, value):
            """Réagit aux changements d'émotion."""
            if value.lower() in ["happy", "excited", "joyful"]:
                self.face_color = [1, 0.85, 0.2, 0.8]  # Jaune vif
            elif value.lower() in ["sad", "disappointed", "melancholic"]:
                self.face_color = [0.9, 0.85, 0.6, 0.8]  # Jaune pâle
            elif value.lower() in ["angry", "frustrated", "annoyed"]:
                self.face_color = [1, 0.7, 0.2, 0.8]  # Orange
            elif value.lower() in ["calm", "peaceful", "relaxed"]:
                self.face_color = [0.8, 0.9, 0.8, 0.8]  # Vert pâle

        def _blink_eyes(self, dt):
            """Anime un clignotement des yeux."""

            def _blink_down(dt):
                anim = Animation(eye_scale=0.1, duration=0.1)
                anim.bind(on_complete=lambda *args: _blink_up(0))
                anim.start(self)

            def _blink_up(dt):
                anim = Animation(eye_scale=1.0, duration=0.1)
                anim.start(self)

            _blink_down(0)

        def _update_face(self, dt):
            """Met à jour l'affichage du visage."""
            self.canvas.clear()

            with self.canvas:
                # Dessiner le visage (cercle)
                Color(*self.face_color)
                face_size = min(self.width, self.height) * 0.8
                face_x = self.center_x - face_size / 2
                face_y = self.center_y - face_size / 2
                Ellipse(pos=(face_x, face_y), size=(face_size, face_size))

                # Dessiner les yeux
                Color(*self.eye_color)
                eye_size = face_size * 0.12 * self.eye_scale
                eye_y = self.center_y + face_size * 0.1

                # Œil gauche
                eye_l_x = self.center_x - face_size * 0.25
                Ellipse(pos=(eye_l_x - eye_size / 2, eye_y - eye_size / 2), size=(eye_size, eye_size))

                # Œil droit
                eye_r_x = self.center_x + face_size * 0.25
                Ellipse(pos=(eye_r_x - eye_size / 2, eye_y - eye_size / 2), size=(eye_size, eye_size))

                # Dessiner la bouche selon l'émotion
                Color(*self.mouth_color)

                mouth_y = self.center_y - face_size * 0.15
                mouth_width = face_size * 0.5

                if self.emotion.lower() in ["happy", "excited", "joyful"]:
                    # Bouche souriante
                    self._draw_smile(self.center_x, mouth_y, mouth_width, self.intensity, smile=True)

                elif self.emotion.lower() in ["sad", "disappointed", "melancholic"]:
                    # Bouche triste
                    self._draw_smile(self.center_x, mouth_y, mouth_width, self.intensity, smile=False)

                elif self.emotion.lower() in ["angry", "frustrated", "annoyed"]:
                    # Bouche en colère
                    points = [
                        self.center_x - mouth_width / 2,
                        mouth_y,
                        self.center_x + mouth_width / 2,
                        mouth_y - face_size * 0.05,
                    ]
                    Line(points=points, width=2)

                    # Sourcils froncés
                    brow_y = eye_y + face_size * 0.15
                    brow_width = face_size * 0.2

                    # Sourcil gauche
                    points = [
                        eye_l_x - brow_width / 2,
                        brow_y + face_size * 0.05,
                        eye_l_x + brow_width / 2,
                        brow_y,
                    ]
                    Line(points=points, width=2)

                    # Sourcil droit
                    points = [
                        eye_r_x - brow_width / 2,
                        brow_y,
                        eye_r_x + brow_width / 2,
                        brow_y + face_size * 0.05,
                    ]
                    Line(points=points, width=2)

                else:  # neutral ou autre
                    # Bouche neutre
                    points = [
                        self.center_x - mouth_width / 2,
                        mouth_y,
                        self.center_x + mouth_width / 2,
                        mouth_y,
                    ]
                    Line(points=points, width=2)

        def _draw_smile(self, center_x, center_y, width, intensity, smile=True):
            """
            Dessine une bouche souriante ou triste.

            Args:
                center_x: Position X du centre
                center_y: Position Y du centre
                width: Largeur de la bouche
                intensity: Intensité de la courbure
                smile: True pour un sourire, False pour une bouche triste
            """
            points = []
            segments = 20
            height = width * 0.3 * intensity

            if not smile:
                height = -height
                center_y -= height

            for i in range(segments + 1):
                # Paramètre t entre -1 et 1
                t = 2 * (i / segments) - 1

                # Équation d'une parabole
                x = center_x + (width / 2) * t
                y = center_y + height * (1 - t * t)

                points.extend([x, y])

            Line(points=points, width=2)

        def set_emotion(self, emotion, intensity=None):
            """
            Définit l'émotion à afficher.

            Args:
                emotion: Nom de l'émotion
                intensity: Intensité de l'émotion (0.0 à 1.0)
            """
            self.emotion = emotion
            if intensity is not None:
                self.intensity = max(0.0, min(1.0, intensity))
