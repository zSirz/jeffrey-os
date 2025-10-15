#!/usr/bin/env python3
"""
Point d'entrée pour l'interface Kivy de Jeffrey
L'UI communique avec le BrainKernel via API/Bus, pas d'import direct
"""

# ICI on peut utiliser Kivy librement car c'est l'UI
import os

os.environ['KIVY_LOG_LEVEL'] = 'info'

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

# Imports Kivy UNIQUEMENT ici

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput


class JeffreyUIApp(App):
    """Interface Kivy pour Jeffrey OS"""

    def build(self):
        # Layout principal
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Titre
        title = Label(text='🧠 Jeffrey OS - Interface UI', size_hint_y=0.1, font_size='24sp')
        layout.add_widget(title)

        # Status
        self.status_label = Label(
            text='État: Déconnecté',
            size_hint_y=0.1,
            color=(1, 0, 0, 1),  # Rouge
        )
        layout.add_widget(self.status_label)

        # Zone de chat
        self.chat_display = TextInput(text='', readonly=True, multiline=True, size_hint_y=0.5)
        layout.add_widget(self.chat_display)

        # Input utilisateur
        self.user_input = TextInput(text='', multiline=False, size_hint_y=0.1, hint_text='Tapez votre message...')
        self.user_input.bind(on_text_validate=self.send_message)
        layout.add_widget(self.user_input)

        # Boutons
        button_layout = BoxLayout(size_hint_y=0.1, spacing=5)

        connect_btn = Button(text='Connecter au Brain')
        connect_btn.bind(on_press=self.connect_to_brain)
        button_layout.add_widget(connect_btn)

        send_btn = Button(text='Envoyer')
        send_btn.bind(on_press=self.send_message)
        button_layout.add_widget(send_btn)

        layout.add_widget(button_layout)

        # Variables d'état
        self.connected = False
        self.api_url = 'http://localhost:8000'

        return layout

    def connect_to_brain(self, instance):
        """Se connecte au BrainKernel via l'API"""
        self.chat_display.text += "\n[SYSTÈME] Connexion au cerveau...\n"

        # ICI : Implémenter la vraie connexion via API REST ou WebSocket
        # Pour l'instant, simulation
        Clock.schedule_once(self.simulate_connection, 1)

    def simulate_connection(self, dt):
        """Simule une connexion réussie"""
        self.connected = True
        self.status_label.text = 'État: Connecté'
        self.status_label.color = (0, 1, 0, 1)  # Vert
        self.chat_display.text += "[SYSTÈME] ✅ Connecté au BrainKernel!\n"

    def send_message(self, instance):
        """Envoie un message au cerveau"""
        message = self.user_input.text.strip()
        if not message:
            return

        if not self.connected:
            self.chat_display.text += "[SYSTÈME] ❌ Non connecté au cerveau\n"
            return

        # Afficher le message de l'utilisateur
        self.chat_display.text += f"\n[VOUS] {message}\n"
        self.user_input.text = ''

        # ICI : Envoyer vraiment au BrainKernel via API
        # Pour l'instant, réponse simulée
        Clock.schedule_once(lambda dt: self.receive_response("Je suis Jeffrey en mode UI. Message reçu!"), 0.5)

    def receive_response(self, response):
        """Affiche la réponse du cerveau"""
        self.chat_display.text += f"[JEFFREY] {response}\n"


if __name__ == "__main__":
    print("Lancement de l'interface UI Jeffrey...")
    JeffreyUIApp().run()
