"""
EmotionGardenScreen - Jardin émotionnel statique pour Jeffrey

Ce module crée une représentation visuelle des émotions passées de Jeffrey
sous forme de jardin où chaque émotion est représentée par une fleur ou un élément visuel.
"""

import math
import random
from datetime import datetime, timedelta

from jeffrey.core.emotions.emotion_visual_engine import EmotionVisualEngine
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.properties import BooleanProperty, ListProperty, NumericProperty, ObjectProperty
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.screenmanager import Screen

try:
    from jeffrey.core.emotional_memory import EmotionalMemory
    from jeffrey.core.io_manager import IOManager
except ImportError:
    # Mock pour les tests
    print("Utilisation de mocks pour le jardin émotionnel")

    class EmotionalMemory:
        def __init__(self, *args, **kwargs):
            pass

        def get_journal(self):
            # Simuler des souvenirs pour les tests
            return [
                {
                    "emotion": "happy",
                    "date": (datetime.now() - timedelta(days=1)).isoformat(),
                    "description": "Un moment de joie",
                },
                {
                    "emotion": "sad",
                    "date": (datetime.now() - timedelta(days=2)).isoformat(),
                    "description": "Un moment de tristesse",
                },
                {
                    "emotion": "excited",
                    "date": (datetime.now() - timedelta(days=3)).isoformat(),
                    "description": "Un moment d'excitation",
                },
                {
                    "emotion": "peaceful",
                    "date": (datetime.now() - timedelta(days=4)).isoformat(),
                    "description": "Un moment de paix",
                },
                {
                    "emotion": "curious",
                    "date": (datetime.now() - timedelta(days=5)).isoformat(),
                    "description": "Un moment de curiosité",
                },
            ]

    class IOManager:
        def __init__(self, *args, **kwargs):
            pass


class EmotionFlower(FloatLayout):
    """Représentation visuelle d'une émotion sous forme de fleur."""

    emotion_color = ListProperty([0.7, 0.7, 1.0, 0.8])
    size_hint = ListProperty([None, None])
    emotion_type = ObjectProperty(None)
    bloom_size = NumericProperty(1.0)
    description = ObjectProperty("")
    date = ObjectProperty("")
    is_selected = BooleanProperty(False)

    def __init__(self, emotion="neutral", description="", date="", **kwargs):
        super(EmotionFlower, self).__init__(**kwargs)
        self.emotion_type = emotion
        self.description = description
        self.date = date

        # Initialiser la taille de la fleur en fonction de l'émotion
        self.size = (dp(60), dp(60))

        # Définir la couleur en fonction de l'émotion
        self.emotion_color = self.get_emotion_color(emotion)

        # Dynamique d'apparition
        self.bloom_size = 0.01
        animation = Animation(bloom_size=1.0, duration=0.8, transition="out_elastic")
        animation.start(self)

        # Démarrer la pulsation
        Clock.schedule_interval(self._pulse, random.uniform(2.0, 5.0))

    def get_emotion_color(self, emotion):
        """Retourne la couleur RGBA correspondant à une émotion."""
        emotion_colors = {
            "happy": [1.0, 0.9, 0.3, 0.8],  # Jaune vif
            "excited": [1.0, 0.7, 0.2, 0.8],  # Orange vif
            "joyful": [1.0, 0.8, 0.5, 0.8],  # Jaune-orange
            "sad": [0.2, 0.3, 0.6, 0.7],  # Bleu triste
            "melancholic": [0.3, 0.4, 0.7, 0.7],  # Bleu mélancolique
            "angry": [0.9, 0.1, 0.1, 0.7],  # Rouge
            "frustrated": [0.8, 0.3, 0.2, 0.7],  # Rouge-orange
            "calm": [0.4, 0.6, 0.9, 0.7],  # Bleu calme
            "peaceful": [0.5, 0.7, 0.9, 0.7],  # Bleu-ciel
            "neutral": [0.6, 0.6, 0.6, 0.7],  # Gris
            "curious": [0.5, 0.3, 0.9, 0.8],  # Violet
            "surprised": [0.9, 0.5, 0.9, 0.8],  # Rose-violet
            "fearful": [0.3, 0.2, 0.5, 0.7],  # Violet foncé
            "disgusted": [0.4, 0.5, 0.2, 0.7],  # Vert-brun
        }

        return emotion_colors.get(emotion.lower(), [0.6, 0.6, 0.6, 0.7])

    def _pulse(self, dt):
        """Crée une légère pulsation de la fleur."""
        current_size = self.bloom_size
        target_size = random.uniform(0.95, 1.05)

        animation = Animation(bloom_size=target_size, duration=1.5, transition="out_sine")
        animation += Animation(bloom_size=1.0, duration=1.5, transition="in_out_sine")
        animation.start(self)

    def on_touch_down(self, touch):
        """Gère les interactions tactiles avec la fleur."""
        if self.collide_point(*touch.pos):
            # Marquer comme sélectionnée
            self.is_selected = True

            # Animer pour montrer la sélection
            Animation(bloom_size=1.2, duration=0.3, transition="out_back").start(self)
            return True
        return super(EmotionFlower, self).on_touch_down(touch)

    def on_touch_up(self, touch):
        """Gère la fin des interactions tactiles."""
        if self.is_selected:
            self.is_selected = False
            Animation(bloom_size=1.0, duration=0.3).start(self)
            return True
        return super(EmotionFlower, self).on_touch_up(touch)


class EmotionGardenScreen(Screen):
    """
    Écran représentant un jardin où les émotions passées sont visualisées
    sous forme de fleurs ou autres éléments visuels.
    """

    memory = ObjectProperty(None)
    flowers = ListProperty([])
    selected_flower = ObjectProperty(None)
    background_emotion_color = ListProperty([0.05, 0.15, 0.05, 1])
    weather_source = ObjectProperty("")

    def __init__(self, **kwargs):
        super(EmotionGardenScreen, self).__init__(**kwargs)

        # Créer le layout principal
        self.garden_layout = FloatLayout()
        self.add_widget(self.garden_layout)

        # Initialiser la mémoire émotionnelle
        self.memory = EmotionalMemory()

        # Initialiser le moteur visuel d'émotions
        self.visual_engine = EmotionVisualEngine()

        # Chargement différé pour laisser le temps à l'interface de s'initialiser
        Clock.schedule_once(self.load_memories, 0.5)

    def load_memories(self, dt):
        """Charge les souvenirs émotionnels et crée les visualisations."""
        # Récupérer le journal émotionnel
        journal = self.memory.get_journal()

        # Créer des fleurs pour chaque entrée du journal (limité aux 50 dernières)
        for index, entry in enumerate(journal[-50:]):
            # Extraire les informations
            emotion = entry.get("emotion", "neutral")
            description = entry.get("description", "")
            date_str = entry.get("date", "")

            # Créer une nouvelle fleur
            flower = EmotionFlower(emotion=emotion, description=description, date=date_str)

            # Positionner la fleur dans le jardin
            self._position_flower(flower, index, len(journal[-50:]))

            # Ajouter au layout
            self.garden_layout.add_widget(flower)
            self.flowers.append(flower)

        # Démarrer l'animation douce des fleurs
        Clock.schedule_interval(self._animate_garden, 0.5)

        # Charger l'émotion la plus récente pour l'ambiance visuelle
        if journal and len(journal) > 0:
            latest_emotion = journal[-1].get("emotion", "neutral")
            self.update_visual_emotion(latest_emotion)
            print(f"[DEBUG] Chargement de l'émotion récente : {latest_emotion}")
        else:
            # Utiliser une émotion par défaut si le journal est vide
            self.update_visual_emotion("neutral")

    def _position_flower(self, flower, index, total):
        """
        Positionne une fleur dans le jardin.

        Args:
            flower: La fleur à positionner
            index: L'index de la fleur
            total: Le nombre total de fleurs
        """
        # Distribuer les fleurs dans une spirale ou un cercle
        if self.width > 0 and self.height > 0:
            # Calculer la position en spirale
            radius = min(self.width, self.height) * 0.4
            angle = (index / total) * 2 * math.pi * 2.5  # 2.5 tours complets

            # La distance du centre augmente progressivement
            distance = (index / total) * radius

            # Coordonnées x et y
            x = self.center_x + math.cos(angle) * distance
            y = self.center_y + math.sin(angle) * distance

            # Appliquer une légère variation aléatoire
            x += random.uniform(-10, 10)
            y += random.uniform(-10, 10)

            # Définir la position
            flower.center = (x, y)
        else:
            # Position par défaut si la taille n'est pas encore définie
            flower.center = (100, 100)

    def _animate_garden(self, dt):
        """
        Anime doucement le jardin pour lui donner vie.

        Args:
            dt: Delta temps
        """
        for flower in self.flowers:
            # Ne pas animer les fleurs sélectionnées
            if flower.is_selected:
                continue

            # Calculer un léger mouvement aléatoire
            dx = random.uniform(-1, 1)
            dy = random.uniform(-1, 1)

            # Créer une animation douce
            current_x, current_y = flower.center
            target_x = current_x + dx
            target_y = current_y + dy

            # S'assurer que la fleur reste dans les limites
            target_x = max(flower.width / 2, min(self.width - flower.width / 2, target_x))
            target_y = max(flower.height / 2, min(self.height - flower.height / 2, target_y))

            # Animer le mouvement
            Animation(center=(target_x, target_y), duration=1.0).start(flower)

    def on_touch_down(self, touch):
        """Gère les interactions tactiles avec le jardin."""
        # Vérifier si une fleur est touchée (géré par EmotionFlower)
        result = super(EmotionGardenScreen, self).on_touch_down(touch)

        # Si aucune fleur n'est touchée et qu'il y a une fleur sélectionnée
        if not result and self.selected_flower:
            self.selected_flower.is_selected = False
            self.selected_flower = None

        return result

    def on_size(self, *args):
        """Gère le redimensionnement de l'écran."""
        # Repositionner toutes les fleurs
        if self.flowers:
            for index, flower in enumerate(self.flowers):
                self._position_flower(flower, index, len(self.flowers))

    def clear_garden(self):
        """Efface toutes les fleurs du jardin."""
        for flower in self.flowers:
            self.garden_layout.remove_widget(flower)
        self.flowers = []
        self.selected_flower = None

    def refresh_garden(self):
        """Rafraîchit le jardin en rechargeant les souvenirs."""
        self.clear_garden()
        self.load_memories(0)

    def update_visual_emotion(self, emotion_str):
        """
        Met à jour les éléments visuels en fonction de l'émotion.

        Args:
            emotion_str: La chaîne représentant l'émotion (ex: "joie", "happy")
        """
        resolved = self.visual_engine.resolve_emotion(emotion_str)
        self.background_emotion_color = self.visual_engine.get_rgba_color(resolved)
        self.weather_source = self.visual_engine.get_weather_asset(resolved)
        print(f"[DEBUG] Émotion mise à jour : {emotion_str} -> {resolved}")

    def trigger_emotion_change(self):
        """Méthode temporaire pour tester un changement d'émotion visuelle."""
        test_emotion = random.choice(["happy", "sad", "excited", "angry", "peaceful", "curious"])
        self.update_visual_emotion(test_emotion)
        print(f"[DEBUG] Changement d'émotion vers : {test_emotion}")
