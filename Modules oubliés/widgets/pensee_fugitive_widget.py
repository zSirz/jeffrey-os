from kivy.animation import Animation
from kivy.clock import Clock
from kivy.properties import StringProperty
from kivy.uix.floatlayout import FloatLayout


class PenseeFugitiveWidget(FloatLayout):
    texte = StringProperty("")

    def __init__(self, **kwargs):
        self.texte = kwargs.get("texte", "")
        super().__init__(**kwargs)
        self.opacity = 0
        Clock.schedule_once(self.animer_apparition, 0.1)
        Clock.schedule_once(self.disparaitre, 8)

    def animer_apparition(self, *args):
        anim = Animation(opacity=1, duration=1.2, t="out_quad")
        anim.start(self)

    def disparaitre(self, *args):
        anim = Animation(opacity=0, duration=1.2, t="out_quad")
        anim.bind(on_complete=self.supprimer)
        anim.start(self)

    def supprimer(self, *args):
        if self.parent:
            self.parent.remove_widget(self)
