import json
import os

from kivy.clock import Clock
from kivy.properties import ListProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen

JOURNAL_PATH = "data/emotional_journal.json"


class EmotionEntry(BoxLayout):
    emotion = StringProperty()
    thought = StringProperty()
    surprise = StringProperty()

    def __init__(self, entry, **kwargs):
        super().__init__(orientation="vertical", spacing=5, padding=10, **kwargs)
        self.emotion = entry.get("emotion", "💭")
        self.thought = entry.get("thought", "")
        self.surprise = entry.get("surprise", "")

        with self.canvas.before:
            from kivy.graphics import Color, RoundedRectangle

            emotion_colors = {
                "❤️": (1, 0.6, 0.6, 0.3),
                "😢": (0.6, 0.6, 1, 0.3),
                "😡": (1, 0.4, 0.4, 0.3),
                "😊": (0.6, 1, 0.6, 0.3),
                "💭": (0.9, 0.9, 0.9, 0.2),
            }
            c = emotion_colors.get(self.emotion, (1, 1, 1, 0.1))
            Color(*c)
            self.bg_rect = RoundedRectangle(radius=[10], pos=self.pos, size=self.size)
            self.bind(pos=self.update_bg, size=self.update_bg)

        self.add_widget(
            Label(
                text=f"[b]{self.emotion}[/b] - {self.thought}",
                markup=True,
                font_size=18,
                size_hint_y=None,
                height=40,
            )
        )
        if self.surprise:
            self.add_widget(
                Label(
                    text=f"[i]🎁 {self.surprise}[/i]",
                    markup=True,
                    font_size=14,
                    size_hint_y=None,
                    height=30,
                )
            )

        from datetime import datetime

        timestamp = entry.get("timestamp")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp)
                formatted = dt.strftime("%d/%m/%Y - %H:%M")
                self.add_widget(
                    Label(
                        text=f"🕒 {formatted}",
                        font_size=12,
                        color=(0.5, 0.5, 0.5, 1),
                        size_hint_y=None,
                        height=20,
                    )
                )
            except Exception:
                pass

    def update_bg(self, *args):
        self.bg_rect.pos = self.pos
        self.bg_rect.size = self.size


class EmotionJournalScreen(Screen):
    entries = ListProperty([])

    def on_enter(self):
        self.load_journal()
        from kivy.animation import Animation

        self.ids.journal_box.opacity = 0
        Animation(opacity=1, d=0.8, t="out_cubic").start(self.ids.journal_box)
        self.refresh_event = Clock.schedule_interval(lambda dt: self.load_journal(), 30)

    def on_leave(self):
        if hasattr(self, "refresh_event"):
            self.refresh_event.cancel()

    def load_journal(self):
        self.ids.journal_box.clear_widgets()
        if not os.path.exists(JOURNAL_PATH):
            self.ids.journal_box.add_widget(
                Label(
                    text="Aucun souvenir émotionnel pour l'instant...",
                    italic=True,
                    font_size=16,
                    color=(0.6, 0.6, 0.6, 1),
                )
            )
            return

        try:
            with open(JOURNAL_PATH, encoding="utf-8") as f:
                data = json.load(f)
                entries = sorted(data.get("entries", []), key=lambda x: x.get("timestamp", ""), reverse=True)
                from kivy.animation import Animation

                for entry in entries:
                    widget = EmotionEntry(entry)
                    self.ids.journal_box.add_widget(widget)
                    anim = Animation(opacity=1, d=0.6, t="out_quad")
                    widget.opacity = 0
                    anim.start(widget)
            Animation.cancel_all(self.ids.journal_box)
            pulse = Animation(opacity=0.7, d=0.2) + Animation(opacity=1, d=0.4)
            pulse.start(self.ids.journal_box)
        except Exception as e:
            self.ids.journal_box.add_widget(Label(text=f"Erreur de lecture: {e}", color=(1, 0, 0, 1)))

    def refresh_journal(self):
        self.load_journal()
