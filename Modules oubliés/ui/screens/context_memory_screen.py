import json
import os

from kivy.properties import ListProperty, StringProperty
from kivy.uix.screenmanager import Screen


class ContextMemoryScreen(Screen):
    recent_memories = ListProperty()
    summary_text = StringProperty("")
    filter_keyword = StringProperty("")

    def on_pre_enter(self):
        self.load_recent_context()

    def load_recent_context(self):
        path = "data/context/context_memory.json"
        if os.path.exists(path):
            with open(path, encoding="utf-8") as file:
                data = json.load(file)
                if self.filter_keyword:
                    data = [entry for entry in data if self.filter_keyword.lower() in json.dumps(entry).lower()]
                self.recent_memories = data[-10:] if len(data) > 10 else data
                self.summary_text = self.generate_summary(data)
        else:
            self.recent_memories = []
            self.summary_text = "Aucun souvenir enregistré."

    def generate_summary(self, data):
        if not data:
            return "Mémoire vide."
        emotions = [entry.get("emotion", "neutre") for entry in data]
        dominant = max(set(emotions), key=emotions.count)
        return f"🧠 Émotion dominante récente : {dominant} ({len(data)} entrées)"

    def apply_filter(self, keyword):
        self.filter_keyword = keyword
        self.load_recent_context()
