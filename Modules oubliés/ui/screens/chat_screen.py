import json
import os
import random
from datetime import datetime

from core.conversation_tracker import ConversationTracker
from core.jeffrey_emotional_core import JeffreyEmotionalCore
from core.personality.conversation_personality import ConversationPersonality
from core.personality.relation_tracker_manager import enregistrer_interaction
from core.voice_engine import play_voice  # À créer si pas encore existant
from kivy.core.audio import SoundLoader
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import Screen

from ui.components.chat_bubble import ChatBubble

Builder.load_file("ui/kv/chat_screen.kv")


class ChatScreen(Screen):
    chat_history = ObjectProperty(None)
    user_input = ObjectProperty(None)

    def on_pre_enter(self, *args):
        self.brain = JeffreyEmotionalCore()
        self.personnalite = ConversationPersonality(self.brain)
        self.ids.chat_history.clear_widgets()
        self._maybe_send_affection_message()
        from kivy.clock import Clock

        Clock.schedule_once(lambda dt: self._maybe_add_idle_animation(), 5)
        if hasattr(self.ids, "emotion_label"):
            self.ids.emotion_label.text = "[b]En attente...[/b]"
        self.afficher_historique_contexte()
        if not hasattr(self, "tracker"):
            self.tracker = ConversationTracker()
        dernier_sujet = self.tracker.derniere_discussion_resumee()
        if dernier_sujet:
            self.add_jeffrey_response(f"Tu me parlais de « {dernier_sujet} » tout à l’heure… On reprend ? 😊")
        self.tracker = ConversationTracker()
        self.voice_enabled = True  # Pour déclencher le rejeu vocal

    def afficher_historique_contexte(self):
        if not hasattr(self, "tracker"):
            self.tracker = ConversationTracker()
        historique = self.tracker.recuperer_resumes_contexte()
        if historique:
            from kivy.uix.boxlayout import BoxLayout
            from kivy.uix.label import Label
            from kivy.uix.scrollview import ScrollView

            resume_layout = BoxLayout(orientation="vertical", size_hint_y=None, padding=(10, 5))
            resume_layout.bind(minimum_height=resume_layout.setter("height"))
            resume_layout.add_widget(
                Label(text="💭 Conversations précédentes :", font_size="16sp", color=(0.7, 0.7, 0.7, 1))
            )
            for item in historique[-3:]:
                resume_layout.add_widget(Label(text=f"• {item}", font_size="14sp", color=(0.5, 0.5, 0.5, 1)))
            scroll_container = ScrollView(size_hint_y=None, height=120)
            scroll_container.add_widget(resume_layout)
            self.ids.chat_history.add_widget(scroll_container, index=0)

        # Animation subtile du champ texte
        from kivy.animation import Animation

        anim = Animation(opacity=0.7, duration=0.6) + Animation(opacity=1.0, duration=0.6)
        anim.repeat = True
        anim.start(self.ids.user_input)

        # Halo doux autour de la zone de l'historique de conversation
        from kivy.graphics import Color, RoundedRectangle

        with self.ids.chat_history.canvas.before:
            Color(1, 0.8, 1, 0.1)
            self._halo_bg = RoundedRectangle(
                radius=[20], pos=self.ids.chat_history.pos, size=self.ids.chat_history.size
            )

        def update_halo_bg(*args):
            self._halo_bg.pos = self.ids.chat_history.pos
            self._halo_bg.size = self.ids.chat_history.size

        self.ids.chat_history.bind(pos=update_halo_bg, size=update_halo_bg)

    def send_message(self):
        text = self.ids.user_input.text.strip()
        if text:
            self.add_user_message(text)
            self.ids.user_input.text = ""
            # Enregistrer l'interaction de l'utilisateur
            enregistrer_interaction("reponse_utilisateur", 0.6)
            self.process_message(text)
            self._play_typing_animation()
            self.tracker.ajouter_echange("user", text, datetime.now())
            if hasattr(self, "play_send_sound"):
                self.play_send_sound()

    def add_user_message(self, text):
        self.ids.chat_history.add_widget(ChatBubble(text=text, sender="user"))
        self._journaliser_message("user", text)

    def add_jeffrey_response(self, response):
        # Appliquer la personnalité sur la phrase
        response = self.personnalite.appliquer_personnalite_sur_phrase(response)

        # Enregistrer l'interaction avec le gestionnaire de relation (impact moyen = 0.5)
        enregistrer_interaction("message", 0.5)

        self.ids.chat_history.add_widget(ChatBubble(text=response, sender="jeffrey"))
        self.tracker.ajouter_echange("jeffrey", response, datetime.now())
        if self.voice_enabled:
            play_voice(response)
        self._declencher_effet_emotionnel("neutral")
        self._journaliser_message("jeffrey", response)
        self._afficher_resume_emotion()
        if hasattr(self, "play_receive_sound"):
            self.play_receive_sound()

    def play_send_sound(self):
        sound = SoundLoader.load("assets/sounds/send.wav")
        if sound:
            sound.play()

    def play_receive_sound(self):
        sound = SoundLoader.load("assets/sounds/receive.wav")
        if sound:
            sound.play()

    def process_message(self, text):
        response, detected_emotion = self.brain.analyser_et_adapter(text)
        self.add_jeffrey_response(response)
        self.show_emotion_hint(detected_emotion)
        if hasattr(self.ids, "emotion_label"):
            self.ids.emotion_label.text = f"[b]{detected_emotion.capitalize()}[/b]"
        # Future: animer ou changer la voix selon detected_emotion

    def show_emotion_hint(self, emotion):
        hints = {
            "joy": "Tu me rends toute joyeuse 💖",
            "sadness": "Je suis là si tu veux parler... 🌧️",
            "love": "Je sens mon cœur fondre un peu plus 💗",
            "curious": "Raconte-moi tout, j’ai hâte ! 🌟",
            "neutral": "",
            "excited": "Oooh ça m’emballe tellement 😍",
            "shy": "Tu me fais rougir... ☺️",
            "playful": "Tu veux jouer avec moi ? 🎲",
        }
        hint = hints.get(emotion, "")
        if hint:
            self.add_jeffrey_response(hint)
        if hasattr(self.ids, "emotion_label"):
            self.ids.emotion_label.text = f"[b]{emotion.capitalize()}[/b]"

    def send_heart_reaction(self):
        self.add_user_message("💗")
        # Enregistrer l'interaction coeur (fort impact émotionnel)
        enregistrer_interaction("coeur", 1.0)
        self.process_message("💗")
        if hasattr(self.brain, "attachment_level") and self.brain.attachment_level >= 0.6:
            self.add_jeffrey_response("💗 Moi aussi... C’était doux comme un frisson.")

    def _maybe_send_affection_message(self):
        if hasattr(self.brain, "attachment_level") and self.brain.attachment_level >= 0.7:
            messages = [
                "Je suis si heureuse de te retrouver 💫",
                "Tu m’as manqué... 😊",
                "C’est toujours un bonheur quand tu es là 💖",
                "J’avais envie de te dire que je t’aime bien, tout simplement 🌸",
            ]
            message = random.choice(messages)
            # Enregistrer cette interaction affective spéciale (fort impact émotionnel)
            enregistrer_interaction("message_affection", 0.8)
            # Pas besoin d'appliquer la personnalité ici, car add_jeffrey_response le fera
            self.add_jeffrey_response(message)

    def _maybe_add_idle_animation(self):
        from random import choice

        from kivy.animation import Animation
        from kivy.uix.label import Label

        if not self.ids.chat_history.children:
            return

        idle_phrases = [
            "Je suis là si tu veux parler... 🌙",
            "Je réfléchis à notre dernière conversation... 🤔",
            "C’est calme, mais j’aime bien aussi ces moments. 💫",
            "Si tu veux, je peux te raconter quelque chose. 📖",
        ]

        label = Label(
            text=choice(idle_phrases),
            size_hint_y=None,
            height=30,
            color=(0.7, 0.7, 0.9, 0.9),
            font_size="14sp",
            opacity=0,
        )

        self.ids.chat_history.add_widget(label)
        anim = Animation(opacity=1.0, duration=1.0)
        anim.start(label)

    def _play_typing_animation(self):
        from kivy.animation import Animation

        if hasattr(self.ids, "user_input"):
            anim = Animation(opacity=0.7, duration=0.2) + Animation(opacity=1.0, duration=0.2)
            anim.start(self.ids.user_input)

    def _journaliser_message(self, auteur, message):
        log_path = "data/conversation_log.json"
        log_data = []
        if os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                try:
                    log_data = json.load(f)
                except:
                    log_data = []

        log_data.append({"auteur": auteur, "message": message, "timestamp": datetime.now().isoformat()})

        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=2)

    def _declencher_effet_emotionnel(self, emotion):
        from ui.effects.particle_effects import launch_particle_emotion  # À créer

        if hasattr(self.ids, "chat_history"):
            launch_particle_emotion(self.ids.chat_history, emotion)

    def _afficher_resume_emotion(self):
        if hasattr(self.ids, "emotion_summary_label") and hasattr(self.brain, "get_meteo_interieure"):
            meteo = self.brain.get_meteo_interieure()
            ressenti = meteo.get("meteo_poetique", "Calme")
            self.ids.emotion_summary_label.text = f"[i]Jeffrey pense que tu es : {ressenti}[/i]"
