#!/usr/bin/env python
"""
Application principale pour le système de guidance émotionnelle et intuitive.
Permet de démarrer l'interface avec la configuration appropriée.
"""

from kivy.app import App
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from ui.guidance_system import GuidanceManager


class MoodSelector(BoxLayout):
    """
    Sélecteur d'ambiance pour le système de guidance.
    Permet à l'utilisateur de choisir entre différents modes émotionnels.
    """

    def __init__(self, guidance_manager, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.size_hint = (1, None)
        self.height = dp(60)
        self.padding = [dp(10), dp(5)]
        self.spacing = dp(10)

        self.guidance_manager = guidance_manager

        # Label descriptif
        self.add_widget(Label(text="Ambiance:", size_hint=(0.3, 1)))

        # Boutons d'ambiance
        self.relax_button = Button(
            text="Relax",
            on_release=lambda x: self.guidance_manager.set_mood("relax"),
            size_hint=(0.23, 1),
            background_color=(0.6, 0.8, 0.6, 1),
        )
        self.energetic_button = Button(
            text="Énergique",
            on_release=lambda x: self.guidance_manager.set_mood("energetic"),
            size_hint=(0.23, 1),
            background_color=(0.5, 0.7, 0.9, 1),
        )
        self.gentle_button = Button(
            text="Douceur",
            on_release=lambda x: self.guidance_manager.set_mood("gentle"),
            size_hint=(0.23, 1),
            background_color=(0.9, 0.7, 0.8, 1),
        )

        self.add_widget(self.relax_button)
        self.add_widget(self.energetic_button)
        self.add_widget(self.gentle_button)


class DemoControls(BoxLayout):
    """
    Contrôles de démonstration pour tester les fonctionnalités du système.
    """

    def __init__(self, guidance_manager, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.size_hint = (1, None)
        self.height = dp(60)
        self.padding = [dp(10), dp(5)]
        self.spacing = dp(10)

        self.guidance_manager = guidance_manager

        # Démonstration de guidance
        guide_button = Button(text="Guider vers...", size_hint=(0.3, 1), on_release=self.show_guidance_demo)
        self.add_widget(guide_button)

        # Simulation hésitation
        hesitate_button = Button(
            text="Simuler hésitation",
            size_hint=(0.3, 1),
            on_release=lambda x: self.guidance_manager.check_hesitation(0),
        )
        self.add_widget(hesitate_button)

        # Réinitialiser tutoriel
        tutorial_button = Button(text="Montrer tutoriel", size_hint=(0.3, 1), on_release=self.show_tutorial)
        self.add_widget(tutorial_button)

    def show_guidance_demo(self, button):
        """Démontre la guidance vers différents points"""
        # Points de démonstration
        demo_points = [(200, 300), (500, 400), (300, 200), (600, 300)]

        # Guide séquentiellement vers ces points
        def guide_step(dt):
            if demo_points:
                point = demo_points.pop(0)
                self.guidance_manager.guide_to_element("demo")

        # Lancer la démo
        guide_step(0)
        for i in range(len(demo_points)):
            Clock.schedule_once(guide_step, i * 2)

    def show_tutorial(self, button):
        """Réinitialise et affiche le tutoriel"""
        self.guidance_manager.first_time_user = True
        self.guidance_manager.add_widget(self.guidance_manager.living_tutorial)
        self.guidance_manager.living_tutorial.start_tutorial()


class GuidanceTestApp(App):
    """
    Application de test pour le système de guidance.
    Inclut des contrôles de démonstration pour visualiser les fonctionnalités.
    """

    def build(self):
        """Construit l'interface de test"""
        root = BoxLayout(orientation="vertical")

        # Créer le gestionnaire de guidance
        self.guidance_manager = GuidanceManager()

        # Ajouter les contrôles
        mood_selector = MoodSelector(self.guidance_manager)
        demo_controls = DemoControls(self.guidance_manager)

        # Assembler l'interface
        root.add_widget(Label(text="Système de Guidance Émotionnelle - Demo", size_hint=(1, None), height=dp(40)))
        root.add_widget(mood_selector)
        root.add_widget(demo_controls)
        root.add_widget(self.guidance_manager)

        return root


if __name__ == "__main__":
    GuidanceTestApp().run()
