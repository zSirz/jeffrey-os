#!/usr/bin/env python3
"""
Test d'intégration Kivy + Jeffrey Bridge V3
Simule une conversation dans une interface Kivy simple
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from kivy.app import App
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput

from src.jeffrey.interfaces.ui.kivy_bridge_integration import get_kivy_bridge


class JeffreyChat(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = 10
        self.spacing = 10

        # Bridge Jeffrey
        self.bridge = get_kivy_bridge()
        self.bridge.initialize(on_ready=self.on_bridge_ready)

        # Zone de chat (scrollable)
        self.chat_scroll = ScrollView(size_hint=(1, 0.8))
        self.chat_layout = BoxLayout(orientation="vertical", size_hint_y=None, spacing=5)
        self.chat_layout.bind(minimum_height=self.chat_layout.setter("height"))
        self.chat_scroll.add_widget(self.chat_layout)
        self.add_widget(self.chat_scroll)

        # Zone de saisie
        input_layout = BoxLayout(size_hint=(1, 0.1), spacing=5)

        self.text_input = TextInput(multiline=False, hint_text="Tapez votre message...", size_hint=(0.8, 1))
        self.text_input.bind(on_text_validate=self.send_message)
        input_layout.add_widget(self.text_input)

        self.send_button = Button(
            text="Envoyer",
            size_hint=(0.2, 1),
            disabled=True,  # Désactivé jusqu'à ce que le bridge soit prêt
        )
        self.send_button.bind(on_press=self.send_message)
        input_layout.add_widget(self.send_button)

        self.add_widget(input_layout)

        # Status bar
        self.status_label = Label(text="Initialisation de Jeffrey...", size_hint=(1, 0.05), color=(0.7, 0.7, 0.7, 1))
        self.add_widget(self.status_label)

        # Current streaming message
        self.current_jeffrey_label = None

        # Ajouter message de bienvenue
        self.add_chat_message("🤖 Jeffrey se réveille...", is_jeffrey=True)

    def on_bridge_ready(self):
        """Appelé quand le bridge est prêt"""
        Logger.info("Chat: Jeffrey is ready!")
        self.status_label.text = "Jeffrey est prêt ✅"
        self.send_button.disabled = False

        # Message de bienvenue
        self.add_chat_message("🤖 Bonjour! Je suis Jeffrey, ravi de te rencontrer!", is_jeffrey=True)

    def send_message(self, *args):
        """Envoie le message à Jeffrey"""
        text = self.text_input.text.strip()
        if not text:
            return

        # Ajouter le message utilisateur
        self.add_chat_message(f"👤 {text}", is_jeffrey=False)

        # Clear input
        self.text_input.text = ""
        self.status_label.text = "Jeffrey réfléchit..."

        # Détecter l'émotion simple
        emotion = self.detect_emotion(text)

        # Envoyer à Jeffrey
        self.bridge.send_message(
            text=text,
            emotion=emotion,
            on_response=self.on_jeffrey_response,
            on_error=self.on_jeffrey_error,
            on_chunk=self.on_streaming_chunk,
        )

    def detect_emotion(self, text):
        """Détection d'émotion basique"""
        text_lower = text.lower()

        if any(w in text_lower for w in ["bonjour", "salut", "hello", "merci"]):
            return "friendly"
        elif "?" in text:
            return "curious"
        elif any(w in text_lower for w in ["triste", "mal", "problème"]):
            return "sad"
        elif any(w in text_lower for w in ["super", "génial", "cool", "!"]):
            return "excited"
        else:
            return "neutral"

    def add_chat_message(self, text, is_jeffrey=True):
        """Ajoute un message au chat"""
        label = Label(
            text=text,
            size_hint_y=None,
            height=40,
            text_size=(self.width - 20, None),
            halign="left" if is_jeffrey else "right",
            color=(0.3, 0.7, 1, 1) if is_jeffrey else (1, 1, 1, 1),
        )

        # Bind width pour text wrapping
        label.bind(width=lambda *x: setattr(label, "text_size", (label.width - 20, None)))
        label.bind(texture_size=lambda *x: setattr(label, "height", label.texture_size[1] + 10))

        self.chat_layout.add_widget(label)

        # Auto-scroll to bottom
        Clock.schedule_once(lambda dt: self.scroll_to_bottom(), 0.1)

        # Garder référence pour streaming
        if is_jeffrey:
            self.current_jeffrey_label = label

        return label

    def scroll_to_bottom(self):
        """Scroll vers le bas du chat"""
        self.chat_scroll.scroll_y = 0

    def on_jeffrey_response(self, response, metadata):
        """Réponse complète de Jeffrey"""
        # Si pas de streaming, ajouter la réponse
        if not self.current_jeffrey_label or not hasattr(self.current_jeffrey_label, "_streaming"):
            self.add_chat_message(f"🤖 {response}", is_jeffrey=True)

        # Métriques
        latency = metadata.get("latency_ms", 0)
        from_cache = metadata.get("from_cache", False)

        cache_text = " (cache)" if from_cache else ""
        self.status_label.text = f"Réponse en {latency:.0f}ms{cache_text}"

    def on_jeffrey_error(self, error):
        """Erreur de Jeffrey"""
        Logger.error(f"Jeffrey error: {error}")
        self.add_chat_message(f"⚠️ Erreur: {error}", is_jeffrey=True)
        self.status_label.text = "Erreur ❌"

    def on_streaming_chunk(self, chunk):
        """Chunk de streaming"""
        if not self.current_jeffrey_label:
            self.current_jeffrey_label = self.add_chat_message("🤖 ", is_jeffrey=True)
            self.current_jeffrey_label._streaming = True
            self.current_jeffrey_label._base_text = "🤖 "

        # Ajouter le chunk
        if hasattr(self.current_jeffrey_label, "_streaming"):
            self.current_jeffrey_label._base_text += chunk
            self.current_jeffrey_label.text = self.current_jeffrey_label._base_text

    def on_stop(self):
        """Cleanup à la fermeture"""
        if self.bridge:
            self.bridge.shutdown()


class JeffreyChatApp(App):
    def build(self):
        return JeffreyChat()

    def on_stop(self):
        """Arrêt propre"""
        if self.root:
            self.root.on_stop()


if __name__ == "__main__":
    # Configurer le logging
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Lancer l'app
    app = JeffreyChatApp()
    app.title = "Jeffrey Chat - Bridge V3"
    app.run()
