import json
import os

from kivy.properties import ListProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.screenmanager import Screen


class HistoryScreen(Screen):
    history_data = ListProperty([])

    def on_pre_enter(self):
        self.load_history()

    def load_history(self):
        # Exemple : charger un fichier JSON avec toutes les conversations
        history_file = "data/conversations_log.json"
        if os.path.exists(history_file):
            with open(history_file, encoding="utf-8") as f:
                try:
                    self.history_data = json.load(f)
                except json.JSONDecodeError:
                    self.history_data = []
        else:
            self.history_data = []

        self.display_history()

    def display_history(self):
        from kivy.utils import escape_markup

        self.ids.container.clear_widgets()
        if not self.history_data:
            self.ids.container.add_widget(HistoryCard(auteur="", contenu="Aucun historique disponible", horodatage=""))
            return

        for session in reversed(self.history_data):
            dt = session.get("timestamp", "")
            messages = session.get("messages", [])
            for msg in messages:
                auteur = msg.get("from", "Jeffrey")
                contenu = msg.get("text", "")
                self.ids.container.add_widget(
                    HistoryCard(auteur=escape_markup(auteur), contenu=escape_markup(contenu), horodatage=dt)
                )


from kivy.properties import StringProperty


class HistoryCard(BoxLayout):
    auteur = StringProperty()
    contenu = StringProperty()
    horodatage = StringProperty()
