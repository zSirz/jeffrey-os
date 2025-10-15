# Hypothetical import for message handling
from jeffrey.core.messages_manager import messages_manager
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.screenmanager import Screen
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput

# Ce fichier pourrait être enrichi avec une interaction tactile plus fine :
# (ex: appui long sur le cœur pour modifier l’émotion associée)


class MessagesFavorisScreen(Screen):
    """
    Écran des messages favoris enrichi avec :
    - Commentaire contextuel de Jeffrey
    - Affichage émotion + interaction via ❤️
    - Suppression via 💔
    """

    def on_enter(self):
        self.clear_widgets()

        try:
            from kivy.core.audio import SoundLoader

            sound = SoundLoader.load("assets/sounds/jeffrey_favoris_chuchote.mp3")
            if sound:
                sound.play()
        except:
            pass

        from kivy.animation import Animation

        # Animation douce autour du label de commentaire
        def animer_label(label):
            anim = Animation(opacity=0.6, duration=1) + Animation(opacity=1, duration=1)
            anim.repeat = True
            anim.start(label)

        scroll = ScrollView()
        layout = BoxLayout(orientation="vertical", size_hint_y=None)
        layout.bind(minimum_height=layout.setter("height"))

        favoris = messages_manager.get_favoris()
        commentaire = messages_manager.get_jeffrey_commentaire_favoris()
        commentaire_label = Label(
            text=f"✨ Jeffrey murmure : « {commentaire} »",
            italic=True,
            size_hint_y=None,
            height=50,
            color=(0.8, 0.6, 1, 1),
            opacity=1,
        )
        layout.add_widget(commentaire_label)
        animer_label(commentaire_label)
        if not favoris:
            layout.add_widget(Label(text="Aucun message favori pour le moment.", size_hint_y=None, height=40))
        else:
            for message in favoris:
                box = BoxLayout(orientation="horizontal", size_hint_y=None, height=100, padding=10, spacing=10)

                label = Label(
                    text=message.get("texte"),
                    halign="left",
                    valign="middle",
                    text_size=(self.width * 0.8, None),
                    size_hint_x=0.85,
                )

                # Tooltip ou popup simulé pour l’émotion liée
                emotion_label = Label(
                    text=f"{message.get('emotion', '')} 💫",
                    font_size=16,
                    size_hint_x=0.05,
                    color=(1, 0.5, 0.8, 1),
                    markup=True,
                )

                def ouvrir_popup_emotion(instance):
                    popup_layout = BoxLayout(orientation="vertical", spacing=10, padding=10)
                    input_emotion = TextInput(hint_text="Émotion associée", multiline=False)
                    btn_valider = Button(text="Valider", size_hint_y=None, height=40)

                    def enregistrer_emotion(instance_btn):
                        message["emotion"] = input_emotion.text
                        messages_manager.ajouter_favori(message)  # Met à jour
                        popup.dismiss()
                        self.on_enter()  # Recharge

                    btn_valider.bind(on_press=enregistrer_emotion)
                    popup_layout.add_widget(input_emotion)
                    popup_layout.add_widget(btn_valider)

                    popup = Popup(title="Modifier l’émotion", content=popup_layout, size_hint=(0.8, 0.4))
                    popup.open()

                heart_btn = Button(
                    text="❤️",
                    font_size=22,
                    size_hint_x=0.1,
                    background_normal="",
                    background_color=(1, 0, 0, 0.4),
                )
                heart_btn.bind(on_release=ouvrir_popup_emotion)

                def animer_heart(btn):
                    anim = Animation(scale=1.2, duration=0.2) + Animation(scale=1.0, duration=0.2)
                    anim.repeat = True
                    btn.canvas.before.clear()
                    with btn.canvas.before:
                        from kivy.graphics import PopMatrix, PushMatrix, Scale

                        PushMatrix()
                        scale = Scale(x=1.0, y=1.0, origin=btn.center)
                        btn.scale = scale
                        anim.start(scale)
                        PopMatrix()

                animer_heart(heart_btn)

                retirer_btn = Button(
                    text="💔",
                    font_size=20,
                    size_hint_x=0.1,
                    background_normal="",
                    background_color=(0.6, 0, 0, 0.3),
                )

                def retirer_favori(instance_btn, msg=message):
                    messages_manager.retirer_favori(msg)
                    self.on_enter()

                retirer_btn.bind(on_release=retirer_favori)

                box.add_widget(label)
                box.add_widget(emotion_label)
                box.add_widget(heart_btn)
                box.add_widget(retirer_btn)
                layout.add_widget(box)

        scroll.add_widget(layout)
        self.add_widget(scroll)

        from kivy.core.window import Window

        Window.bind(on_resize=self._resize_labels)
        self._resize_labels(None, Window.width, Window.height)

    def _resize_labels(self, instance, width, height):
        for child in self.children:
            if isinstance(child, ScrollView):
                for box in child.children:
                    if isinstance(box, BoxLayout):
                        for widget in box.children:
                            if isinstance(widget, Label):
                                widget.text_size = (self.width * 0.9, None)


# TODO : ajouter une méthode pour taguer un message favori avec une émotion personnalisée

# L’utilisateur peut désormais personnaliser l’émotion d’un message favori via un popup déclenché par un appui sur le ❤️
