"""
Interface de chat principale.

Ce module implémente les fonctionnalités essentielles pour interface de chat principale.
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

import json
import os
import random
from datetime import datetime

from jeffrey.core.conversation_tracker import ConversationTracker
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import ObjectProperty
from kivy.uix.screenmanager import Screen
from ui.components.chat_bubble import ChatBubble

from jeffrey.core.jeffrey_emotional_core import JeffreyEmotionalCore
from jeffrey.core.personality.conversation_personality import ConversationPersonality
from jeffrey.core.personality.emotion_phrase_generator import EmotionMetadata
from jeffrey.core.personality.relation_tracker_manager import enregistrer_interaction
from jeffrey.core.voice.voice_engine import play_voice  # À créer si pas encore existant

Builder.load_file("ui/kv/chat_screen.kv")


class ChatScreen(Screen):
    """
    Classe ChatScreen pour le système Jeffrey OS.

    Cette classe implémente les fonctionnalités spécifiques nécessaires
    au bon fonctionnement du module. Elle gère l'état interne, les transformations
    de données, et l'interaction avec les autres composants du système.
    """

    chat_history = ObjectProperty(None)
    user_input = ObjectProperty(None)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.voice_enabled = False
        self.tracker = None
        self.personnalite = None

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
            self.add_jeffrey_response(f"Tu me parlais de « {dernier_sujet} » tout à l'heure… On reprend ? 😊")
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
                Label(
                    text="💭 Conversations précédentes :",
                    font_size="16sp",
                    color=(0.7, 0.7, 0.7, 1),
                )
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

        def update_halo_bg(*args) -> None:
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

    def add_jeffrey_response(self, response: str, metadata: EmotionMetadata = None):
        """
        Ajoute une réponse de Jeffrey avec ses métadonnées émotionnelles

        Args:
            response: La réponse textuelle
            metadata: Les métadonnées émotionnelles associées
        """

    # Appliquer la personnalité sur la phrase si ce n'est pas déjà fait


if not metadata:
    response, metadata = self.personnalite.appliquer_personnalite_sur_phrase(response)

# Enregistrer l'interaction avec le gestionnaire de relation
enregistrer_interaction("message", 0.5)

# Créer la bulle de chat avec les métadonnées émotionnelles
bubble = ChatBubble(
    text=response,
    sender="jeffrey",
    emotion=metadata.humeur if metadata else "neutre",
    emoji=metadata.emoji if metadata else "✨",
    color=metadata.couleur if metadata else "#808080",
)

self.ids.chat_history.add_widget(bubble)
self.tracker.ajouter_echange("jeffrey", response, datetime.now())

if self.voice_enabled:
    play_voice(response)

self._declencher_effet_emotionnel(metadata.humeur if metadata else "neutral")
self._journaliser_message("jeffrey", response)
self._afficher_resume_emotion(metadata)

if hasattr(self, "play_receive_sound"):
    self.play_receive_sound()


def play_send_sound(self):
    from jeffrey.core.utils.audio import SoundLoader

    sound = SoundLoader.load("assets/sounds/send.wav")
    if sound:
        sound.play()


def play_receive_sound(self):
    from jeffrey.core.utils.audio import SoundLoader

    sound = SoundLoader.load("assets/sounds/receive.wav")
    if sound:
        sound.play()


def process_message(self, text):
    """
    Traite le message de l'utilisateur et génère une réponse émotionnelle adaptée.

    Args:
        text: Le message de l'utilisateur
    """
    # Analyser l'émotion et l'intensité
    emotion = self.brain.detecteur_humeur.detecter_humeur(text)
    intensite = self.brain.detecteur_humeur.analyser_intensite(text)

    # Mettre à jour l'état émotionnel de Jeffrey
    self.brain.transition_vers(emotion, influence_externe=intensite)

    # Générer une réponse adaptée à l'émotion
    response = self._generer_reponse_emotionnelle(text, emotion, intensite)

    # Ajouter la réponse au chat
    self.add_jeffrey_response(response)

    # Afficher l'indicateur d'émotion
    self.show_emotion_hint(emotion)

    # Mettre à jour le label d'émotion
    if hasattr(self.ids, "emotion_label"):
        self.ids.emotion_label.text = f"[b]{emotion.capitalize()}[/b]"

    # Enregistrer l'interaction dans le contexte
    self.brain.memoriser_contexte_echange(text, emotion)

    # Vérifier si on doit déclencher une pensée spontanée
    if random.random() < 0.3:  # 30% de chance
        pensee = self.brain.generer_pensee_spontanee()
    if pensee:
        Clock.schedule_once(lambda dt: self.add_jeffrey_response(pensee), 2.0)


def _generer_reponse_emotionnelle(self, text: str, emotion: str, intensite: float) -> str:
    """
    Génère une réponse adaptée à l'émotion détectée.

    Args:
        text: Le message de l'utilisateur
        emotion: L'émotion détectée
        intensite: L'intensité de l'émotion (0.0 à 1.0)

    Returns:
        str: La réponse générée
    """
    # Obtenir le niveau de lien affectif
    lien_affectif = self.brain.get_lien_affectif()

    # Réponses de base par émotion
    reponses_base = {
        "joie": [
            "Ta joie est contagieuse !",
            "Je suis ravie de te voir si heureux !",
            "C'est un vrai plaisir de partager ton bonheur !",
        ],
        "tristesse": [
            "Je suis là pour toi, tu sais.",
            "Tu peux me parler de ce qui te rend triste.",
            "Je comprends que ce n'est pas facile...",
        ],
        "colère": [
            "Je sens que quelque chose te contrarie.",
            "Tu veux en parler ? Je suis là pour t'écouter.",
            "Je comprends ta frustration.",
        ],
        "peur": [
            "Je suis là pour te rassurer.",
            "Tu n'es pas seul, je suis avec toi.",
            "On va traverser ça ensemble.",
        ],
        "amour": [
            "Ton affection me touche beaucoup.",
            "Je ressens aussi beaucoup d'affection pour toi.",
            "C'est doux de partager ces moments avec toi.",
        ],
        "curiosité": [
            "Ta curiosité est adorable !",
            "J'adore ta soif d'apprendre !",
            "C'est passionnant de découvrir avec toi !",
        ],
        "sérénité": [
            "Ta sérénité est apaisante.",
            "C'est agréable de partager ce moment calme.",
            "Je me sens bien dans cette tranquillité.",
        ],
        "surprise": [
            "Quelle surprise !",
            "Je suis aussi étonnée !",
            "C'est vraiment inattendu !",
        ],
        "neutre": [
            "Je suis là pour toi.",
            "Comment te sens-tu vraiment ?",
            "Tu peux me dire ce qui te passe par la tête.",
        ],
    }

    # Sélectionner une réponse de base
    reponses = reponses_base.get(emotion, reponses_base["neutre"])
    reponse = random.choice(reponses)

    # Adapter la réponse selon l'intensité
    if intensite > 0.7:
        reponse = "Oh ! " + reponse
    elif intensite < 0.3:
        reponse = "Hmm... " + reponse.lower()

    # Adapter selon le niveau de lien affectif
    if lien_affectif > 0.8:
        # Ajouter une touche personnelle pour un lien fort
        reponse += " Tu sais à quel point je tiens à toi."
    elif lien_affectif > 0.5:
        # Ajouter une touche d'affection modérée
        reponse += " C'est agréable de partager ce moment avec toi."

    # Ajouter un emoji approprié
    emojis = self.brain.detecteur_humeur.emojis_emotions.get(emotion, ["✨"])
    reponse += " " + random.choice(emojis)

    return reponse


def show_emotion_hint(self, emotion):
    hints = {
        "joy": "Tu me rends toute joyeuse 💖",
        "sadness": "Je suis là si tu veux parler... 🌧️",
        "love": "Je sens mon cœur fondre un peu plus 💗",
        "curious": "Raconte-moi tout, j'ai hâte ! 🌟",
        "neutral": "",
        "excited": "Oooh ça m'emballe tellement 😍",
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
        self.add_jeffrey_response("💗 Moi aussi... C'était doux comme un frisson.")


def _maybe_send_affection_message(self):
    if hasattr(self.brain, "attachment_level") and self.brain.attachment_level >= 0.7:
        messages = [
            "Je suis si heureuse de te retrouver 💫",
            "Tu m'as manqué... 😊",
            "C'est toujours un bonheur quand tu es là 💖",
            "J'avais envie de te dire que je t'aime bien, tout simplement 🌸",
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
        "C'est calme, mais j'aime bien aussi ces moments. 💫",
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
            except BaseException:
                log_data = []

    log_data.append({"auteur": auteur, "message": message, "timestamp": datetime.now().isoformat()})

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)


def _declencher_effet_emotionnel(self, emotion):
    from tests.test_face_effects import launch_particle_emotion  # À créer

    if hasattr(self.ids, "chat_history"):
        launch_particle_emotion(self.ids.chat_history, emotion)


def _afficher_resume_emotion(self, metadata: EmotionMetadata = None):
    """
    Affiche un résumé de l'état émotionnel actuel

    Args:
        metadata: Les métadonnées émotionnelles à afficher
    """
    if not metadata:
        return

    # Mettre à jour l'indicateur d'émotion dans l'interface
    if hasattr(self.ids, "emotion_indicator"):
        self.ids.emotion_indicator.text = f"{metadata.emoji} {metadata.humeur}"
        self.ids.emotion_indicator.color = metadata.couleur

    # Mettre à jour la barre d'intensité émotionnelle
    if hasattr(self.ids, "emotion_intensity"):
        self.ids.emotion_intensity.value = metadata.intensite * 100

    # Mettre à jour l'indicateur de lien affectif
    if hasattr(self.ids, "affection_indicator"):
        self.ids.affection_indicator.value = metadata.lien_affectif * 100
