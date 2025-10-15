from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.core.audio import SoundLoader
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.properties import DictProperty, ObjectProperty
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.popup import Popup

from jeffrey.core.personality.conversation_personality import ConversationPersonality
from jeffrey.core.personality.relation_tracker_manager import enregistrer_interaction

Builder.load_file("ui/kv/surprise_popup.kv")


class SurprisePopup(Popup):
    surprise = DictProperty()
    personnalite = ObjectProperty(None)

    def __init__(self, surprise, **kwargs):
        super(SurprisePopup, self).__init__(**kwargs)
        self.opacity = 0
        Animation(opacity=1, scale=1.05, d=0.6, t="out_elastic").start(self)
        self.surprise = surprise

        # Initialiser la personnalité conversationnelle
        app = App.get_running_app()
        if hasattr(app, "jeffrey"):
            self.personnalite = ConversationPersonality(app.jeffrey)

        # Enregistrer cette interaction de surprise (fort impact émotionnel)
        surprise_type = self.surprise.get("type", "general")
        impact_value = 1.2  # Les surprises ont un fort impact émotionnel
        enregistrer_interaction(f"surprise_{surprise_type}", impact_value)

        self.envelope_img = Image(source="assets/images/envelope_closed.png", size_hint=(None, None), size=(200, 200))
        self.ids.surprise_content.add_widget(self.envelope_img)
        Clock.schedule_once(self.animate_envelope_opening, 0.5)
        self.play_sound()

    def animate_envelope_opening(self, *args):
        anim = Animation(opacity=0, d=0.8) + Animation(opacity=1, d=0.5)
        self.envelope_img.source = "assets/images/envelope_opening.gif"
        anim.bind(on_complete=lambda *a: self.ids.surprise_content.remove_widget(self.envelope_img))
        anim.start(self.envelope_img)
        Clock.schedule_once(self.reveal_surprise, 1.0)

    def reveal_surprise(self, *args):
        content_widget = self.ids.get("surprise_content")
        # Animation type machine à écrire pour le message d'intro (champ intro facultatif mais recommandé)
        intro = Label(text="", font_size="16sp", halign="center", italic=True)
        self.ids.surprise_content.add_widget(intro)
        intro_message = self.surprise.get("intro", "✨ Jeffrey avait envie de te faire une surprise...")

        # Appliquer la personnalité sur le message d'introduction
        if self.personnalite:
            intro_message = self.personnalite.appliquer_personnalite_sur_phrase(intro_message)
        self.animate_text(intro, intro_message)

        if self.surprise.get("type") == "text":
            # Appliquer la personnalité au message texte
            message = self.surprise.get("content", "✨ Une surprise de Jeffrey ✨")
            if self.personnalite:
                message = self.personnalite.appliquer_personnalite_sur_phrase(message)

            content_widget.add_widget(Label(text=message, font_size="18sp", halign="center"))
        elif self.surprise.get("type") == "image":
            content_widget.add_widget(
                Image(
                    source=self.surprise.get("content"),
                    size_hint=(1, 1),
                    allow_stretch=True,
                    keep_ratio=True,
                )
            )
        elif self.surprise.get("type") == "audio":
            content_widget.add_widget(
                Label(text="🎵 Lecture d’un message audio spécial…", font_size="18sp", halign="center")
            )
            audio_path = self.surprise.get("content")
            sound = SoundLoader.load(audio_path)
            if sound:
                sound.play()

        elif self.surprise.get("type") == "letter":
            # Appliquer la personnalité à la lettre
            lettre = self.surprise.get("content", "💌 Une lettre tendre de Jeffrey...")
            if self.personnalite:
                lettre = self.personnalite.appliquer_personnalite_sur_phrase(lettre)

            content_widget.add_widget(Label(text=lettre, font_size="16sp", halign="center", markup=True))

        elif self.surprise.get("type") == "video":
            content_widget.add_widget(
                Label(text="📹 Une vidéo magique va commencer", font_size="16sp", halign="center")
            )
            # Placeholder pour lecteur vidéo si ffpyplayer est installé

        elif self.surprise.get("type") == "quote":
            # Appliquer la personnalité à la citation
            citation = self.surprise.get("content", "🌠 Une pensée étoilée pour toi...")
            if self.personnalite:
                citation = self.personnalite.appliquer_personnalite_sur_phrase(citation)

            content_widget.add_widget(Label(text=citation, font_size="18sp", italic=True, halign="center"))

        elif self.surprise.get("type") == "tip":
            content_widget.add_widget(
                Label(
                    text=self.surprise.get("content", "🌿 Une douce astuce de vie t’attend..."),
                    font_size="17sp",
                    halign="center",
                    color=(0.4, 0.7, 0.6, 1),
                )
            )

        elif self.surprise.get("type") == "ai_art":
            img = Image(
                source=self.surprise.get("content"),
                size_hint=(1, 1),
                allow_stretch=True,
                keep_ratio=True,
            )
            content_widget.add_widget(img)
            # Bordure lumineuse
            with img.canvas.after:
                from kivy.graphics import Color, Line

                Color(1, 0.8, 0.9, 0.6)
                Line(rectangle=(img.x, img.y, img.width, img.height), width=2)

        elif self.surprise.get("type") == "secret":
            from kivy.uix.button import Button

            def reveal_secret(instance):
                content_widget.clear_widgets()
                content_widget.add_widget(
                    Label(
                        text=self.surprise.get("content", "🔓 Voici ton secret révélé..."),
                        font_size="18sp",
                        halign="center",
                    )
                )

            btn = Button(text="🔐 Déverrouiller le secret")
            btn.bind(on_press=reveal_secret)
            content_widget.add_widget(btn)

        else:
            content_widget.add_widget(
                Label(
                    text="🎁 Une surprise magique t’attend…",
                    font_size="20sp",
                    italic=True,
                    halign="center",
                )
            )
        anim = Animation(opacity=1, duration=1.5)
        anim.start(content_widget)

        # Ajout d'un halo lumineux simulé autour du contenu (effets visuels liés à un halo autour du widget)
        if hasattr(content_widget, "canvas"):
            with content_widget.canvas.before:
                from kivy.graphics import Color, Ellipse

                Color(1, 1, 0.8, 0.25)
                self.halo = Ellipse(
                    size=(content_widget.width + 40, content_widget.height + 40),
                    pos=(content_widget.x - 20, content_widget.y - 20),
                )

            def update_halo(*args):
                self.halo.pos = (content_widget.x - 20, content_widget.y - 20)
                self.halo.size = (content_widget.width + 40, content_widget.height + 40)

            content_widget.bind(pos=update_halo, size=update_halo)

        # Éclair léger de l'écran lors de l'apparition du cadeau
        original_color = Window.clearcolor

        def flash_effect(*args):
            Window.clearcolor = (1, 1, 1, 1)
            Clock.schedule_once(lambda dt: setattr(Window, "clearcolor", original_color), 0.15)

        Clock.schedule_once(flash_effect, 0.1)

        from kivy.uix.button import Button

        like_btn = Button(text="💖", size_hint=(None, None), size=(50, 50), background_color=(1, 0.6, 0.8, 1))
        like_btn.bind(on_press=lambda x: print("💖 Jeffrey a été touchée !"))
        self.ids.surprise_content.add_widget(like_btn)
        glow = Animation(opacity=0.6, duration=0.7) + Animation(opacity=1, duration=0.7)
        glow.repeat = True
        glow.start(like_btn)

    def animate_text(self, label, full_text, i=0):
        if i <= len(full_text):
            label.text = full_text[:i]
            Clock.schedule_once(lambda dt: self.animate_text(label, full_text, i + 1), 0.05)

    def play_sound(self):
        type_to_sound = {
            "text": "assets/sounds/magic_reveal.wav",
            "letter": "assets/sounds/paper_scroll.wav",
            "quote": "assets/sounds/twinkle.wav",
            "ai_art": "assets/sounds/cosmic_breath.wav",
        }
        path = type_to_sound.get(self.surprise.get("type"), "assets/sounds/magic_reveal.wav")
        sound = SoundLoader.load(path)
        if sound:
            sound.volume = 0.8
            sound.play()
