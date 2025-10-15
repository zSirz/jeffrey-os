import random

from jeffrey.core.ia_pricing import estimate_cost
from kivy.animation import Animation
from kivy.app import App
from kivy.clock import Clock
from kivy.properties import ListProperty, NumericProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label

from jeffrey.core.personality.conversation_personality import ConversationPersonality
from jeffrey.core.personality.relation_tracker_manager import enregistrer_interaction


class RecommendationPopup(BoxLayout):
    recommended_models = ListProperty()
    total_cost = NumericProperty(0.0)
    task_description = StringProperty("")
    personnalite = ObjectProperty(None)

    def __init__(self, task_description="", recommended_models=None, **kwargs):
        super().__init__(**kwargs)
        self.task_description = task_description
        self.recommended_models = recommended_models or []

        # Initialiser la personnalité avec le cœur émotionnel de Jeffrey
        app = App.get_running_app()
        if hasattr(app, "jeffrey"):
            self.personnalite = ConversationPersonality(app.jeffrey)

        self.calculate_total_cost()

    def calculate_total_cost(self):
        total = 0.0
        for model in self.recommended_models:
            cost = estimate_cost(model, token_count=800)
            total += cost
        self.total_cost = round(total, 4)

    def confirm_and_execute(self):
        """
        Exécute la tâche avec les IA recommandées, en appelant orchestrator,
        et affiche une animation visuelle rapide.
        """
        # Enregistrer cette interaction d'exécution de tâche
        enregistrer_interaction("execution_tache", 0.7)
        print("✅ Lancement de la tâche avec les IA suivantes :")
        for model in self.recommended_models:
            print(f" - {model}")
        print(f"💰 Coût estimé total : {self.total_cost} $")

        app = App.get_running_app()
        if hasattr(app, "orchestrator"):
            app.orchestrator.execute_task(prompt=self.task_description, models=self.recommended_models)
            # Ajout feedback visuel temporaire
            label = Label(
                text="🌀 Tâche en cours...",
                font_size="16sp",
                color=(0.7, 1, 0.7, 1),
                size_hint=(None, None),
                size=(200, 50),
                pos_hint={"center_x": 0.5, "center_y": 0.5},
            )
            if self.parent:
                self.parent.add_widget(label)
                Clock.schedule_once(lambda dt: self.parent.remove_widget(label), 2.5)

            def show_task_complete_label(*args):
                base_phrases = [
                    "La tâche est terminée.",
                    "C’est fait, comme tu l’as demandé.",
                    "Voilà qui est fait.",
                    "C’est terminé.",
                    "Mission accomplie.",
                    "C’est prêt.",
                    "Fini, à ma façon.",
                    "Et voilà, comme prévu.",
                ]
                if hasattr(self, "personnalite") and self.personnalite:
                    phrase_choisie = self.personnalite.appliquer_personnalite_sur_phrase(random.choice(base_phrases))
                else:
                    phrase_choisie = random.choice(base_phrases)

                done_label = Label(
                    text=phrase_choisie,
                    font_size="16sp",
                    color=(0.5, 1, 0.5, 1),
                    size_hint=(None, None),
                    size=(200, 50),
                    pos_hint={"center_x": 0.5, "center_y": 0.5},
                )
                if self.parent:
                    self.parent.add_widget(done_label)

                    # Shake animation
                    anim = (
                        Animation(x=done_label.x - 4, duration=0.05)
                        + Animation(x=done_label.x + 4, duration=0.05)
                        + Animation(x=done_label.x, duration=0.1)
                    )
                    anim.start(done_label)

                    Clock.schedule_once(lambda dt: self.parent.remove_widget(done_label), 2.5)

                # Utilise l’intelligence émotionnelle centralisée de Jeffrey pour la phrase de fin
                emotion_engine = getattr(App.get_running_app(), "jeffrey", None)
                if emotion_engine and hasattr(emotion_engine, "say_with_emotion"):
                    emotion_engine.say_with_emotion(phrase=phrase_choisie, context="task_complete")

            Clock.schedule_once(show_task_complete_label, 3.0)
        else:
            print("⚠️ Erreur : orchestrator non trouvé dans l'app.")

    def toggle_history(self):
        widget = self.ids.history_widget
        if widget.opacity == 0:
            widget.opacity = 1
            widget.height = widget.minimum_height
            widget.load_history()
        else:
            widget.opacity = 0
            widget.height = 0
