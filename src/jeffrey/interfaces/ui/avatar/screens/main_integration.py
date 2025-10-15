"""
MainIntegration - Point d'entrée pour tester les composants visuels émotionnels

Ce module intègre les différents composants émotionnels et constitue un point d'entrée
pour tester le visualiseur d'émotions, le jardin émotionnel et la navigation émotionnelle.
"""

from kivy.app import App
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

Builder.load_file('ui/kv/emotion_visualizer.kv')
Builder.load_file('ui/kv/emotion_garden_screen.kv')
Builder.load_file('ui/kv/emotional_navigation.kv')
Builder.load_file('ui/kv/main_emotion_screen.kv')
Builder.load_file('ui/kv/emotional_profile_screen.kv')
from jeffrey.interfaces.ui.main_emotion_screen import MainEmotionScreen
from jeffrey.interfaces.ui.screens.emotion_garden_screen import EmotionGardenScreen
from jeffrey.interfaces.ui.screens.emotion_visualizer import EmotionVisualizer
from jeffrey.interfaces.ui.screens.emotional_navigation import EmotionalNavigation
from jeffrey.interfaces.ui.screens.emotional_profile_screen import EmotionalProfileScreen


class EmotionalState:
    """Classe simulant l'état émotionnel pour les tests."""

    def __init__(self):
        self.current = 'neutral'
        self.intensity = 0.5
        self.observers = []

    def set_emotion(self, emotion, intensity=0.5):
        """
        Définit l'émotion actuelle.

        Args:
            emotion: Émotion à définir
            intensity: Intensité de l'émotion (0.0 à 1.0)
        """
        self.current = emotion
        self.intensity = intensity
        for observer in self.observers:
            if hasattr(observer, '_on_emotional_state_change'):
                observer._on_emotional_state_change()

    def add_observer(self, observer):
        """
        Ajoute un observateur pour les changements d'état émotionnel.

        Args:
            observer: Widget observant les changements d'émotion
        """
        if observer not in self.observers:
            self.observers.append(observer)
            if hasattr(observer, 'emotional_state'):
                observer.emotional_state = self


class MainIntegrationApp(App):
    """Application de test pour les composants émotionnels."""

    def build(self):
        """Construit l'interface principale."""
        emotional_state = EmotionalState()
        navigation = EmotionalNavigation()
        emotional_state.add_observer(navigation)
        main_screen = MainEmotionScreen(name='main')
        navigation.add_widget(main_screen)
        visualizer_screen = EmotionVisualizer()
        emotional_state.add_observer(visualizer_screen)
        garden_screen = EmotionGardenScreen(name='garden')
        navigation.add_widget(garden_screen)
        profile_screen = EmotionalProfileScreen(name='emotional_profile')
        navigation.add_widget(profile_screen)
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(navigation)
        controls = BoxLayout(size_hint=(1, 0.1))
        btn_main = Button(text='Écran Principal')
        btn_main.bind(on_press=lambda x: navigation.switch_screen('main'))
        controls.add_widget(btn_main)
        btn_garden = Button(text='Jardin Émotionnel')
        btn_garden.bind(on_press=lambda x: navigation.switch_screen('garden'))
        controls.add_widget(btn_garden)
        btn_reset = Button(text='Réinitialiser', background_color=(0.3, 0.6, 1.0, 1.0))
        btn_reset.bind(on_press=lambda x: emotional_state.set_emotion('neutral', 0.5))
        controls.add_widget(btn_reset)
        emotions = ['happy', 'sad', 'angry', 'peaceful', 'neutral']
        for emotion in emotions:
            btn = Button(text=emotion.capitalize())
            btn.emotion = emotion
            btn.bind(on_press=lambda x: emotional_state.set_emotion(x.emotion))
            controls.add_widget(btn)
        layout.add_widget(controls)
        self.emotional_state = emotional_state
        Clock.schedule_once(self.run_demo_sequence, 3)
        return layout

    def run_demo_sequence(self, dt):
        """
        Exécute une séquence de démonstration des différentes émotions.

        Args:
            dt: Delta temps
        """
        emotions = [
            ('neutral', 0.5, 3),
            ('happy', 0.7, 4),
            ('excited', 0.8, 3),
            ('peaceful', 0.6, 4),
            ('sad', 0.5, 4),
            ('melancholic', 0.6, 3),
            ('angry', 0.7, 3),
            ('calm', 0.5, 4),
            ('neutral', 0.5, 3),
        ]

        def set_next_emotion(index=0):
            if index < len(emotions):
                emotion, intensity, duration = emotions[index]
                print(f'Émotion: {emotion}, Intensité: {intensity}')
                self.emotional_state.set_emotion(emotion, intensity)
                Clock.schedule_once(lambda dt: set_next_emotion(index + 1), duration)

        set_next_emotion()


if __name__ == '__main__':
    MainIntegrationApp().run()
