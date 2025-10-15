"""
EmotionalNavigation - Liaison entre l'état émotionnel et la navigation

Ce module adapte la navigation et les transitions de l'interface utilisateur
en fonction de l'état émotionnel actuel de Jeffrey.
"""

from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from kivy.metrics import dp
from kivy.properties import ListProperty, NumericProperty, ObjectProperty, StringProperty
from kivy.uix.screenmanager import FadeTransition, Screen, ScreenManager, SlideTransition, WipeTransition


class EmotionalNavigation(ScreenManager):
    """
    Gestionnaire d'écrans qui adapte son apparence et ses transitions
    en fonction de l'état émotionnel actuel.
    """

    emotional_state = ObjectProperty(None)
    current_emotion = StringProperty("neutral")
    emotion_intensity = NumericProperty(0.5)
    background_color = ListProperty([0.1, 0.1, 0.2, 1.0])
    transition_duration = NumericProperty(0.5)

    def __init__(self, **kwargs):
        """Initialise le gestionnaire d'écrans émotionnel."""
        # Définir une transition par défaut
        kwargs["transition"] = FadeTransition(duration=0.5)
        super(EmotionalNavigation, self).__init__(**kwargs)

        # Lier la mise à jour émotionnelle
        self.bind(emotional_state=self._on_emotional_state_change)

        # Paramètres selon l'émotion
        self.emotion_params = {
            # Émotions joyeuses - transitions rapides, couleurs vives
            "happy": {
                "transition": FadeTransition,
                "duration": 0.4,
                "background": [0.2, 0.2, 0.1, 1.0],
                "direction": "left",
            },
            "excited": {
                "transition": SlideTransition,
                "duration": 0.3,
                "background": [0.25, 0.15, 0.05, 1.0],
                "direction": "left",
            },
            "joyful": {
                "transition": FadeTransition,
                "duration": 0.35,
                "background": [0.2, 0.25, 0.1, 1.0],
                "direction": "left",
            },
            # Émotions tristes - transitions lentes, teintes bleutées
            "sad": {
                "transition": FadeTransition,
                "duration": 0.8,
                "background": [0.05, 0.1, 0.2, 1.0],
                "direction": "right",
            },
            "melancholic": {
                "transition": FadeTransition,
                "duration": 0.9,
                "background": [0.05, 0.1, 0.25, 1.0],
                "direction": "right",
            },
            "disappointed": {
                "transition": SlideTransition,
                "duration": 0.7,
                "background": [0.1, 0.1, 0.15, 1.0],
                "direction": "right",
            },
            # Émotions calmes - transitions douces, teintes apaisantes
            "calm": {
                "transition": FadeTransition,
                "duration": 0.6,
                "background": [0.1, 0.15, 0.25, 1.0],
                "direction": "up",
            },
            "peaceful": {
                "transition": FadeTransition,
                "duration": 0.7,
                "background": [0.15, 0.2, 0.3, 1.0],
                "direction": "up",
            },
            "relaxed": {
                "transition": FadeTransition,
                "duration": 0.65,
                "background": [0.1, 0.2, 0.2, 1.0],
                "direction": "up",
            },
            # Émotions énervées - transitions brusques, teintes rouges
            "angry": {
                "transition": SlideTransition,
                "duration": 0.25,
                "background": [0.25, 0.05, 0.05, 1.0],
                "direction": "down",
            },
            "frustrated": {
                "transition": SlideTransition,
                "duration": 0.3,
                "background": [0.2, 0.1, 0.05, 1.0],
                "direction": "down",
            },
            "annoyed": {
                "transition": WipeTransition,
                "duration": 0.35,
                "background": [0.15, 0.1, 0.05, 1.0],
                "direction": "down",
            },
            # État neutre
            "neutral": {
                "transition": FadeTransition,
                "duration": 0.5,
                "background": [0.1, 0.1, 0.2, 1.0],
                "direction": "left",
            },
        }

        # Configurer l'arrière-plan
        self.bind(
            size=self._update_background,
            pos=self._update_background,
            background_color=self._update_background,
        )

        # Mettre à jour immédiatement
        Clock.schedule_once(self._update_background, 0)

    def _on_emotional_state_change(self, *args):
        """Réagit aux changements d'état émotionnel."""
        if self.emotional_state:
            self.current_emotion = self.emotional_state.current
            self.emotion_intensity = self.emotional_state.intensity
            self._adapt_to_emotion()

    def _adapt_to_emotion(self):
        """Adapte la navigation à l'émotion actuelle."""
        # Récupérer les paramètres pour cette émotion
        params = self.emotion_params.get(self.current_emotion, self.emotion_params["neutral"])

        # Ajuster la durée en fonction de l'intensité
        adjusted_duration = params["duration"]
        if self.current_emotion in ["happy", "excited", "joyful"]:
            # Plus intense = plus rapide pour les émotions joyeuses
            adjusted_duration *= 1.0 - (self.emotion_intensity * 0.5)
        elif self.current_emotion in ["sad", "melancholic", "disappointed"]:
            # Plus intense = plus lent pour les émotions tristes
            adjusted_duration *= 1.0 + (self.emotion_intensity * 0.5)

        # Mettre à jour la transition
        self.transition = params["transition"](duration=adjusted_duration, direction=params["direction"])
        self.transition_duration = adjusted_duration

        # Animer la couleur d'arrière-plan
        target_color = params["background"]
        Animation(background_color=target_color, duration=1.0).start(self)

    def _update_background(self, *args):
        """Met à jour l'arrière-plan en fonction de la couleur émotionnelle."""
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self.background_color)
            Rectangle(pos=self.pos, size=self.size)

    def switch_screen(self, screen_name):
        """
        Change d'écran avec une transition adaptée à l'émotion actuelle.

        Args:
            screen_name: Nom de l'écran cible
        """
        if screen_name in self.screen_names:
            self.current = screen_name

    def add_screen(self, screen):
        """
        Ajoute un écran au gestionnaire.

        Args:
            screen: Instance de Screen à ajouter
        """
        self.add_widget(screen)

    def display_emotion_message(self, message, duration=3.0):
        """
        Affiche un message émotionnel temporaire.

        Args:
            message: Message à afficher
            duration: Durée d'affichage en secondes
        """
        # Créer un écran temporaire pour le message
        message_screen = Screen(name=f"message_{hash(message)}")

        # Ajouter un label avec le message
        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.label import Label

        layout = BoxLayout(orientation="vertical", padding=dp(20))

        # Couleur et style selon l'émotion
        text_color = [1, 1, 1, 1]  # Blanc par défaut
        font_size = dp(24)

        if self.current_emotion in ["happy", "excited", "joyful"]:
            text_color = [1, 1, 0.7, 1]  # Jaune clair
            font_size = dp(28)
        elif self.current_emotion in ["sad", "melancholic", "disappointed"]:
            text_color = [0.7, 0.8, 1, 1]  # Bleu clair
            font_size = dp(22)
        elif self.current_emotion in ["angry", "frustrated", "annoyed"]:
            text_color = [1, 0.7, 0.7, 1]  # Rouge clair
            font_size = dp(26)

        label = Label(text=message, font_size=font_size, color=text_color, halign="center", valign="middle")
        layout.add_widget(label)
        message_screen.add_widget(layout)

        # Sauvegarder l'écran actuel
        previous_screen = self.current

        # Ajouter et afficher l'écran de message
        self.add_widget(message_screen)
        self.switch_screen(message_screen.name)

        # Programmer le retour à l'écran précédent
        def return_to_previous(*args):
            if message_screen.name in self.screen_names:
                self.switch_screen(previous_screen)
                Clock.schedule_once(lambda dt: self.remove_widget(message_screen), self.transition_duration + 0.1)

        Clock.schedule_once(return_to_previous, duration)
