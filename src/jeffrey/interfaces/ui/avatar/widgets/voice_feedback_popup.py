"""
Widget de feedback vocal pour l'interface utilisateur.

Ce module contient des widgets permettant d'afficher un retour visuel
lors de l'utilisation de la voix dans l'application.
"""

from kivy.animation import Animation
from kivy.clock import Clock
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup


class VoiceFeedbackPopup(Popup):
    """Widget qui affiche un popup de feedback lors de l'utilisation de la voix."""

    message = StringProperty("Écoute en cours...")
    active = BooleanProperty(False)
    auto_dismiss = BooleanProperty(True)

    def __init__(self, **kwargs):
        super(VoiceFeedbackPopup, self).__init__(**kwargs)
        self.title = "Voix de Jeffrey"
        self.size_hint = (0.8, 0.3)
        self.content = VoiceFeedbackContent(message=self.message)
        self.bind(message=self._update_message)
        self.bind(active=self._update_active)

    def _update_message(self, instance, value):
        """Met à jour le message affiché dans le popup."""
        if hasattr(self, "content") and isinstance(self.content, VoiceFeedbackContent):
            self.content.message = value

    def _update_active(self, instance, value):
        """Active ou désactive l'animation selon l'état."""
        if hasattr(self, "content") and isinstance(self.content, VoiceFeedbackContent):
            self.content.active = value

    def show_message(self, message, duration=None):
        """Affiche un message pour une durée déterminée."""
        self.message = message
        self.open()

        if duration is not None:
            Clock.schedule_once(self.dismiss, duration)


class VoiceFeedbackContent(BoxLayout):
    """Contenu du popup de feedback vocal avec animations."""

    message = StringProperty("Écoute en cours...")
    active = BooleanProperty(False)
    dot_opacity = NumericProperty(0.5)

    def __init__(self, **kwargs):
        super(VoiceFeedbackContent, self).__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = [20, 20]
        self.spacing = 10

        # Créer le label de message
        self.message_label = Label(
            text=self.message, font_size="18sp", halign="center", valign="middle", size_hint_y=0.6
        )
        self.add_widget(self.message_label)

        # Créer le layout pour les points d'animation
        self.dots_layout = BoxLayout(orientation="horizontal", spacing=10, size_hint_y=0.4)

        # Ajouter les points d'animation
        self.dots = []
        for i in range(3):
            dot = Label(text="•", font_size="30sp", opacity=self.dot_opacity, size_hint_x=1 / 3)
            self.dots.append(dot)
            self.dots_layout.add_widget(dot)

        self.add_widget(self.dots_layout)

        # Démarrer l'animation si actif
        self.bind(message=self._update_message)
        self.bind(active=self._update_active)

        if self.active:
            self._start_animation()

    def _update_message(self, instance, value):
        """Met à jour le texte du message."""
        self.message_label.text = value

    def _update_active(self, instance, value):
        """Active ou désactive l'animation des points."""
        if value:
            self._start_animation()
        else:
            self._stop_animation()

    def _start_animation(self):
        """Démarre l'animation des points."""
        # Animation séquentielle des points
        self.dot_anims = []
        for i, dot in enumerate(self.dots):
            anim = Animation(opacity=1, duration=0.3) + Animation(opacity=0.3, duration=0.3)
            anim.repeat = True

            # Décaler le début de l'animation pour chaque point
            Clock.schedule_once(lambda dt, d=dot, a=anim: a.start(d), i * 0.2)
            self.dot_anims.append(anim)

    def _stop_animation(self):
        """Arrête l'animation des points."""
        for i, dot in enumerate(self.dots):
            if hasattr(self, "dot_anims") and i < len(self.dot_anims):
                self.dot_anims[i].cancel(dot)
            dot.opacity = self.dot_opacity
