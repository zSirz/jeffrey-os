"""
Integration between Kivy UI and Jeffrey Bridge V3
Connect chat_screen.py with jeffrey_ui_bridge.py
"""

import asyncio
import threading
from collections.abc import Callable

from kivy.clock import Clock
from kivy.logger import Logger

from .jeffrey_ui_bridge import JeffreyUIBridge


class KivyBridgeIntegration:
    """
    Intègre le Bridge V3 avec les écrans Kivy existants
    Thread-safe et non-bloquant
    """

    def __init__(self):
        self.bridge = None
        self.bridge_thread = None
        self.loop = None
        self._initialized = False

    def initialize(self, on_ready: Callable | None = None):
        """
        Initialise le bridge dans un thread séparé

        Args:
            on_ready: Callback appelé quand le bridge est prêt
        """
        if self._initialized:
            Logger.warning("KivyBridge: Already initialized")
            return

        def run_bridge():
            """Démarre le bridge dans son propre event loop"""
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

            # Créer le bridge
            self.bridge = JeffreyUIBridge()

            # Callback sur le thread principal Kivy
            if on_ready:
                Clock.schedule_once(lambda dt: on_ready(), 0)

            # Maintenir le loop actif
            self.loop.run_forever()

        # Démarrer dans un thread séparé
        self.bridge_thread = threading.Thread(target=run_bridge, daemon=True)
        self.bridge_thread.start()
        self._initialized = True

        Logger.info("KivyBridge: Initialization started")

    def send_message(
        self,
        text: str,
        emotion: str | None = None,
        on_response: Callable | None = None,
        on_error: Callable | None = None,
        on_chunk: Callable | None = None,
    ):
        """
        Envoie un message à Jeffrey de manière non-bloquante

        Args:
            text: Message à envoyer
            emotion: Emotion détectée (optional)
            on_response: Callback(response_text, metadata) sur le thread UI
            on_error: Callback(error_msg) sur le thread UI
            on_chunk: Callback(chunk_text) pour streaming sur le thread UI
        """

        if not self.bridge:
            if on_error:
                Clock.schedule_once(lambda dt: on_error("Bridge not initialized"), 0)
            return

        # Wrappers thread-safe pour callbacks
        def safe_on_complete(response, metadata):
            if on_response:
                Clock.schedule_once(lambda dt: on_response(response, metadata), 0)

        def safe_on_error(error):
            if on_error:
                Clock.schedule_once(lambda dt: on_error(error), 0)

        def safe_on_chunk(chunk):
            if on_chunk:
                Clock.schedule_once(lambda dt: on_chunk(chunk), 0)

        # Envoyer le message sur le loop du bridge
        future = asyncio.run_coroutine_threadsafe(
            self._async_send(text, emotion, safe_on_complete, safe_on_error, safe_on_chunk),
            self.loop,
        )

    async def _async_send(self, text, emotion, on_complete, on_error, on_chunk):
        """Helper async pour envoyer le message"""
        try:
            self.bridge.send_message(
                text=text,
                emotion=emotion,
                on_complete=on_complete,
                on_error=on_error,
                on_chunk=on_chunk if on_chunk else None,
                enable_streaming=bool(on_chunk),
            )
        except Exception as e:
            on_error(str(e))

    def get_metrics(self) -> dict:
        """Récupère les métriques du bridge"""
        if not self.bridge:
            return {}
        return self.bridge.get_metrics()

    def shutdown(self):
        """Arrêt propre"""
        if self.bridge:
            self.bridge.shutdown()

        if self.loop:
            self.loop.call_soon_threadsafe(self.loop.stop)

        if self.bridge_thread:
            self.bridge_thread.join(timeout=2)

        self._initialized = False
        Logger.info("KivyBridge: Shutdown complete")


# Instance singleton
_kivy_bridge = None


def get_kivy_bridge() -> KivyBridgeIntegration:
    """Récupère le bridge singleton"""
    global _kivy_bridge
    if not _kivy_bridge:
        _kivy_bridge = KivyBridgeIntegration()
    return _kivy_bridge


# Exemple d'utilisation dans chat_screen.py:
"""
from src.jeffrey.interfaces.ui.kivy_bridge_integration import get_kivy_bridge

class ChatScreen:
    def __init__(self):
        self.bridge = get_kivy_bridge()
        self.bridge.initialize(on_ready=self.on_bridge_ready)

    def on_bridge_ready(self):
        print("Jeffrey is ready!")

    def send_user_message(self, text):
        # Détecter l'émotion (optionnel)
        emotion = self.detect_emotion(text)

        # Envoyer le message
        self.bridge.send_message(
            text=text,
            emotion=emotion,
            on_response=self.on_jeffrey_response,
            on_error=self.on_jeffrey_error,
            on_chunk=self.on_streaming_chunk  # Pour streaming
        )

    def on_jeffrey_response(self, response, metadata):
        # Afficher la réponse dans l'UI
        self.add_message_to_chat(response, is_jeffrey=True)

        # Optionnel: utiliser les métadonnées
        latency = metadata.get('latency_ms', 0)
        print(f"Response in {latency}ms")

    def on_jeffrey_error(self, error):
        # Afficher l'erreur
        self.show_error_popup(error)

    def on_streaming_chunk(self, chunk):
        # Ajouter le chunk au message en cours
        self.append_to_current_message(chunk)
"""
