from __future__ import annotations

import math
import time

from kivy.animation import Animation
from kivy.clock import Clock
from kivy.graphics import Color, RoundedRectangle
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label


class JournalEntryCard(BoxLayout):
    """
    Module de interface utilisateur pour Jeffrey OS.

    Ce module implémente les fonctionnalités essentielles pour module de interface utilisateur pour jeffrey os.
    Il fournit une architecture robuste et évolutive intégrant les composants
    nécessaires au fonctionnement optimal du système. L'implémentation suit
    les principes de modularité et d'extensibilité pour faciliter l'évolution
    future du système.

    Le module gère l'initialisation, la configuration, le traitement des données,
    la communication inter-composants, et la persistance des états. Il s'intègre
    harmonieusement avec l'architecture globale de Jeffrey OS tout en maintenant
    une séparation claire des responsabilités.

    L'architecture interne permet une évolution adaptative basée sur les interactions
    et l'apprentissage continu, contribuant à l'émergence d'une conscience artificielle
    cohérente et authentique.
    """

    date = StringProperty("")
    text = StringProperty("")
    emotion = StringProperty("neutre")
    importance = NumericProperty(0.5)
    is_favorite = BooleanProperty(False)

    def __init__(self, **kwargs) -> None:
        super(JournalEntryCard, self).__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = [10, 10, 10, 10]
        self.spacing = 5
        self.bind(size=self.update_layout)

        # Créer l'interface utilisateur de la carte
        Clock.schedule_once(self.build_interface, 0)

    def build_interface(self, dt):
        """Construit l'interface de la carte de journal."""
        pass
        # En-tête avec date et émotion
        header = BoxLayout(size_hint=(1, None), height=30)

        # Label pour la date
        date_label = Label(
            text=self.date,
            font_size="16sp",
            color=(0.8, 0.8, 0.9, 1),
            size_hint=(0.7, 1),
            halign="left",
            text_size=(self.width * 0.7, None),
            valign="middle",
        )
        header.add_widget(date_label)

        # Indicateur d'émotion
        emotion_icon = self.get_emotion_icon()
        emotion_label = Label(text=emotion_icon, font_size="18sp", size_hint=(0.15, 1))
        header.add_widget(emotion_label)

        # Bouton favori
        favorite_icon = "❤️" if self.is_favorite else "♡"
        favorite_button = Button(
            text=favorite_icon,
            font_size="18sp",
            size_hint=(0.15, 1),
            background_color=(0, 0, 0, 0),
            color=(1, 0.5, 0.5, 0.9) if self.is_favorite else (0.8, 0.8, 0.8, 0.6),
        )
        favorite_button.bind(on_release=self.toggle_favorite)
        header.add_widget(favorite_button)

        self.add_widget(header)

        # Contenu principal
        content = Label(
            text=self.text,
            font_size="18sp",
            color=(1, 1, 1, 0.9),
            size_hint=(1, None),
            height=80,
            text_size=(self.width - 20, None),
            halign="left",
            valign="top",
        )
        self.add_widget(content)

        # Mettre à jour l'apparence
        self.update_layout()

    def update_layout(self, *args) -> None:
        """Met à jour l'apparence de la carte basée sur l'émotion et l'importance."""
        self.canvas.before.clear()
        with self.canvas.before:
            # Couleur basée sur l'émotion
            color = self.get_emotion_color()
            Color(*color)

            # Forme de base
            RoundedRectangle(pos=self.pos, size=self.size, radius=[15])

            # Bordure si entrée favorite
            if self.is_favorite:
                Color(1, 0.5, 0.5, 0.3 + 0.1 * math.sin(time.time() * 2))
                RoundedRectangle(
                    pos=[self.pos[0] - 2, self.pos[1] - 2],
                    size=[self.size[0] + 4, self.size[1] + 4],
                    radius=[17],
                )

    def get_emotion_icon(self) -> Any:
        """Renvoie l'icône d'émotion appropriée."""
        emotion_icons = {
            "joie": "😊",
            "tristesse": "😢",
            "colere": "😠",
            "peur": "😨",
            "surprise": "😲",
            "amour": "💗",
            "calme": "😌",
            "curiosité": "🤔",
            "neutre": "😐",
        }
        return emotion_icons.get(self.emotion, "😐")

    def get_emotion_color(self) -> Any:
        """Renvoie la couleur appropriée pour l'émotion."""
        # Base de couleur selon l'émotion
        emotion_colors = {
            "joie": (0.9, 0.8, 0.2, 0.15),
            "tristesse": (0.2, 0.4, 0.8, 0.15),
            "colere": (0.8, 0.2, 0.2, 0.15),
            "peur": (0.4, 0.2, 0.6, 0.15),
            "surprise": (0.6, 0.3, 0.8, 0.15),
            "amour": (0.9, 0.3, 0.5, 0.15),
            "calme": (0.3, 0.6, 0.8, 0.15),
            "curiosité": (0.4, 0.6, 0.7, 0.15),
            "neutre": (0.3, 0.3, 0.4, 0.15),
        }

        # Couleur de base ou par défaut
        base_color = emotion_colors.get(self.emotion, (0.3, 0.3, 0.4, 0.15))

        # Ajuster l'intensité selon l'importance
        r, g, b, a = base_color
        a = a * (0.7 + 0.6 * self.importance)

        return (r, g, b, a)

    def toggle_favorite(self, instance):
        """Bascule l'état favori de l'entrée de journal."""
        self.is_favorite = not self.is_favorite

        # Mettre à jour l'icône du bouton
        instance.text = "❤️" if self.is_favorite else "♡"
        instance.color = (1, 0.5, 0.5, 0.9) if self.is_favorite else (0.8, 0.8, 0.8, 0.6)

        # Animation lors du changement
        anim = Animation(opacity=0, duration=0.1) + Animation(opacity=1, duration=0.2)
        anim.start(instance)

        # Mettre à jour l'apparence
        self.update_layout()

        # Propager l'événement (pour persistance éventuelle)
        if hasattr(self.parent, "parent") and hasattr(self.parent.parent, "on_entry_favorite_changed"):
            self.parent.parent.on_entry_favorite_changed(self)
