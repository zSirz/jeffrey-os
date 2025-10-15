import math

from kivy.animation import Animation
from kivy.app import App
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.properties import BooleanProperty, NumericProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button


class TouchFloatingMenu(BoxLayout):
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

from __future__ import annotations

    is_open = BooleanProperty(False)
    button_size = NumericProperty(dp(50))

    def __init__(self, **kwargs) -> None:
        super(TouchFloatingMenu, self).__init__(**kwargs)
        self.orientation = 'vertical'
        self.spacing = dp(10)
        self.size_hint = (None, None)
        self.size = (self.button_size, self.button_size)
        self.pos_hint = {'right': 0.95, 'top': 0.95}

        # Bouton principal pour ouvrir/fermer le menu
        self.main_button = Button(
            text='≡',
            font_size=dp(24),
            size_hint=(1, 1),
            background_color=(0.4, 0.5, 0.9, 0.7),
            color=(1, 1, 1, 1),
            border=(0, 0, 0, 0)
        )
        self.main_button.bind(on_release=self.toggle_menu)
        self.add_widget(self.main_button)

        # Création des boutons du menu
        self.menu_buttons = [
            {
                'text': '🏠',
                'screen': 'jeffrey_main',
                'tooltip': 'Écran principal'
            },
            {
                'text': '📔',
                'screen': 'jeffrey_journal',
                'tooltip': 'Journal de Jeffrey'
            },
            {
                'text': '🔮',
                'screen': 'private_sanctuary',
                'tooltip': 'Sanctuaire privé'
            },
            {
                'text': '📊',
                'screen': 'emotional_profile',
                'tooltip': 'Profil émotionnel'
            },
            {
                'text': '💭',
                'screen': 'internal_dialogue',
                'tooltip': 'Dialogue interne'
            }
        ]

        # Les boutons du menu sont masqués par défaut
        self.buttons = []
        for btn_info in self.menu_buttons:
            btn = Button(
                text=btn_info['text'],
                font_size=dp(20),
                size_hint=(1, 1),
                size=(self.button_size, self.button_size),
                background_color=(0.3, 0.4, 0.7, 0.7),
                color=(1, 1, 1, 1),
                opacity=0,
                disabled=True
            )
            btn.screen_name = btn_info['screen']
            btn.tooltip = btn_info['tooltip']
            btn.bind(on_release=self.on_menu_button_press)
            self.buttons.append(btn)
            self.add_widget(btn)

    def toggle_menu(self, instance):
        """Bascule l'état d'ouverture du menu."""
        self.is_open = not self.is_open

        if self.is_open:
            self.open_menu()
        else:
            self.close_menu()

    def open_menu(self):
        """Ouvre le menu en animant les boutons."""
        # Changer la taille du layout
        target_height = self.button_size * (len(self.menu_buttons) + 1) + self.spacing * len(self.menu_buttons)
        anim = Animation(height=target_height, duration=0.3)
        anim.start(self)

        # Animer chaque bouton
        for i, btn in enumerate(self.buttons):
            btn.disabled = False
            anim = Animation(opacity=1, duration=0.2, transition='out_quad')
            anim.start(btn)

    def close_menu(self):
        """Ferme le menu en animant les boutons."""
        # Animer chaque bouton
        for i, btn in enumerate(self.buttons):
            btn.disabled = True
            anim = Animation(opacity=0, duration=0.2, transition='in_quad')
            anim.start(btn)

        # Réduire la taille du layout
        anim = Animation(height=self.button_size, duration=0.3)
        anim.start(self)

    def on_menu_button_press(self, instance):
        """Gère l'appui sur un bouton du menu."""
        # Fermer le menu
        self.is_open = False
        self.close_menu()

        # Changer d'écran
        app = App.get_running_app()
        if hasattr(app, 'root') and app.root is not None and hasattr(app.root, 'current'):
            target_screen = instance.screen_name

            if hasattr(app.root, 'has_screen') and callable(app.root.has_screen) and app.root.has_screen(target_screen):
                app.root.current = target_screen
            else:
                print(f"[TouchFloatingMenu] Écran non disponible: {target_screen}")
