"""
EmotionalProfileScreen - Profil émotionnel personnalisé

Ce module affiche le profil émotionnel calculé par EmotionalLearning,
permettant à l'utilisateur de voir les tendances émotionnelles de Jeffrey
basées sur les interactions passées.
"""

from datetime import datetime

from core.emotions.emotional_learning import EmotionalLearning
from kivy.animation import Animation
from kivy.clock import Clock
from kivy.metrics import dp
from kivy.properties import DictProperty, ListProperty, NumericProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.screenmanager import Screen
from kivy.uix.scrollview import ScrollView


class EmotionCard(BoxLayout):
    """Carte affichant une émotion spécifique et ses statistiques."""

    emotion_name = StringProperty("")
    emotion_count = NumericProperty(0)
    emotion_percentage = NumericProperty(0)
    last_seen = StringProperty("")
    emotion_color = ListProperty([0.5, 0.5, 0.5, 0.8])

    def __init__(self, name, data, total_count, **kwargs):
        """
        Initialise une carte d'émotion.

        Args:
            name: Nom de l'émotion
            data: Données de l'émotion (dict avec count et last_seen)
            total_count: Nombre total d'émotions pour calculer le pourcentage
        """
        super(EmotionCard, self).__init__(**kwargs)
        self.orientation = "vertical"
        self.padding = dp(10)
        self.spacing = dp(5)
        self.size_hint_y = None
        self.height = dp(120)

        self.emotion_name = name
        self.emotion_count = data["count"]

        # Calcul du pourcentage (éviter la division par zéro)
        if total_count > 0:
            self.emotion_percentage = (data["count"] / total_count) * 100
        else:
            self.emotion_percentage = 0

        # Format de la dernière observation
        if data["last_seen"]:
            try:
                last_seen_date = datetime.fromisoformat(data["last_seen"])
                self.last_seen = last_seen_date.strftime("%d/%m/%Y %H:%M")
            except (ValueError, TypeError):
                self.last_seen = "Date inconnue"
        else:
            self.last_seen = "Jamais observée"

        # Couleur associée à l'émotion
        self.emotion_color = self._get_emotion_color(name)

    def _get_emotion_color(self, emotion_name):
        """Retourne une couleur RGBA associée à une émotion."""
        emotion_colors = {
            "joie": [1.0, 0.9, 0.3, 0.8],
            "happy": [1.0, 0.9, 0.3, 0.8],
            "tristesse": [0.2, 0.3, 0.6, 0.7],
            "sad": [0.2, 0.3, 0.6, 0.7],
            "colère": [0.9, 0.1, 0.1, 0.7],
            "angry": [0.9, 0.1, 0.1, 0.7],
            "peur": [0.3, 0.2, 0.5, 0.7],
            "fear": [0.3, 0.2, 0.5, 0.7],
            "surprise": [0.9, 0.5, 0.9, 0.8],
            "surprised": [0.9, 0.5, 0.9, 0.8],
            "dégoût": [0.4, 0.5, 0.2, 0.7],
            "disgust": [0.4, 0.5, 0.2, 0.7],
            "curiosité": [0.5, 0.3, 0.9, 0.8],
            "curious": [0.5, 0.3, 0.9, 0.8],
            "calme": [0.4, 0.6, 0.9, 0.7],
            "calm": [0.4, 0.6, 0.9, 0.7],
            "amour": [1.0, 0.7, 0.7, 0.8],
            "love": [1.0, 0.7, 0.7, 0.8],
            "neutre": [0.6, 0.6, 0.6, 0.7],
            "neutral": [0.6, 0.6, 0.6, 0.7],
        }

        # Recherche par nom exact ou par approximation
        color = emotion_colors.get(emotion_name.lower())
        if not color:
            # Chercher une correspondance partielle
            for key, value in emotion_colors.items():
                if key in emotion_name.lower() or emotion_name.lower() in key:
                    color = value
                    break

        # Couleur par défaut si aucune correspondance
        if not color:
            color = [0.6, 0.6, 0.6, 0.7]

        return color


class EmotionalProfileScreen(Screen):
    """
    Écran affichant le profil émotionnel détaillé de Jeffrey
    basé sur les apprentissages émotionnels.
    """

    profile_data = DictProperty({})
    dominant_emotions = ListProperty([])
    total_emotions = NumericProperty(0)

    def __init__(self, **kwargs):
        super(EmotionalProfileScreen, self).__init__(**kwargs)
        self.name = "emotional_profile"

        # Initialiser le moteur d'apprentissage émotionnel
        self.emotional_learning = EmotionalLearning()

        # Mise en page différée pour éviter les problèmes de rendu
        Clock.schedule_once(self.setup_ui, 0.1)

    def setup_ui(self, dt):
        """Configure les éléments d'interface après l'initialisation."""
        self.load_profile()

    def on_enter(self):
        """Appelé lorsque l'écran devient actif."""
        self.load_profile()
        # Animation d'entrée
        self.opacity = 0
        Animation(opacity=1, duration=0.3).start(self)

    def load_profile(self):
        """Charge les données du profil émotionnel."""
        # Essayer de charger un profil existant, sinon créer un profil de démonstration
        try:
            self.emotional_learning.load_profile()
            profile = self.emotional_learning.get_profile()

            if not profile or not profile.get("detailed_profile"):
                # Si aucun profil trouvé, créer des données de démonstration
                self._create_demo_profile()
                profile = self.emotional_learning.get_profile()
        except Exception as e:
            print(f"Erreur lors du chargement du profil émotionnel: {e}")
            # Créer des données de démonstration en cas d'erreur
            self._create_demo_profile()
            profile = self.emotional_learning.get_profile()

        # Mise à jour des propriétés
        self.profile_data = profile
        self.dominant_emotions = profile.get("dominant_emotions", [])
        self.total_emotions = profile.get("total_emotions_tracked", 0)

        # Générer l'interface en fonction des données
        self.update_ui()

    def update_ui(self):
        """Met à jour l'interface avec les données actuelles du profil."""
        # Réinitialiser le contenu
        if hasattr(self, "content_layout"):
            self.content_layout.clear_widgets()

        # Afficher les émotions dominantes
        dominant_layout = GridLayout(cols=3, spacing=dp(10), size_hint_y=None, height=dp(120))

        # Affichage du nombre total d'émotions
        emotions_count_label = Label(
            text=f"[b]{self.total_emotions}[/b] émotions enregistrées",
            markup=True,
            font_size=dp(16),
            size_hint_y=None,
            height=dp(40),
        )

        # Création de la grille pour les cartes d'émotions
        cards_grid = GridLayout(cols=2, spacing=dp(15), size_hint_y=None, padding=[dp(20), dp(10)])
        cards_grid.bind(minimum_height=cards_grid.setter("height"))

        # Ajouter les cartes d'émotions
        detailed_profile = self.profile_data.get("detailed_profile", {})

        # Trier les émotions par nombre d'occurrences
        sorted_emotions = sorted(detailed_profile.items(), key=lambda x: x[1]["count"], reverse=True)

        for emotion_name, emotion_data in sorted_emotions:
            card = EmotionCard(name=emotion_name, data=emotion_data, total_count=self.total_emotions)
            cards_grid.add_widget(card)

        # Aucune émotion n'a été trouvée
        if not sorted_emotions:
            empty_label = Label(
                text="Aucune émotion enregistrée pour le moment.\nInteragissez avec Jeffrey pour enrichir son profil émotionnel.",
                halign="center",
                valign="middle",
                size_hint_y=None,
                height=dp(100),
            )
            cards_grid.add_widget(empty_label)

        # Assemblage de l'interface
        if not hasattr(self, "content_layout"):
            # Création des layouts principaux
            self.content_layout = BoxLayout(orientation="vertical", spacing=dp(20), padding=dp(20))

            # Scrollview pour le contenu
            scroll_view = ScrollView(do_scroll_x=False)
            scroll_view.add_widget(self.content_layout)

            # Ajout du scrollview au widget principal
            self.add_widget(scroll_view)

        # Ajouter les widgets au layout de contenu
        self.content_layout.add_widget(emotions_count_label)
        self.content_layout.add_widget(cards_grid)

        # Bouton d'exportation
        export_button = Button(
            text="Exporter le profil",
            size_hint=(None, None),
            size=(dp(200), dp(50)),
            pos_hint={"center_x": 0.5},
            background_color=(0.2, 0.6, 0.8, 1.0),
        )
        export_button.bind(on_release=self.export_profile)

        # Bouton retour
        back_button = Button(
            text="Retour",
            size_hint=(None, None),
            size=(dp(100), dp(50)),
            pos_hint={"center_x": 0.5},
            background_color=(0.3, 0.3, 0.3, 1.0),
        )
        back_button.bind(on_release=self.go_back)

        # Layout pour les boutons
        buttons_layout = BoxLayout(
            orientation="horizontal",
            spacing=dp(20),
            padding=dp(20),
            size_hint_y=None,
            height=dp(70),
        )
        buttons_layout.add_widget(back_button)
        buttons_layout.add_widget(export_button)

        self.content_layout.add_widget(buttons_layout)

    def _create_demo_profile(self):
        """Crée un profil de démonstration si aucun profil n'existe."""
        # Réinitialiser
        self.emotional_learning = EmotionalLearning()

        # Ajouter des émotions fictives
        emotions = [
            ("joie", 15),
            ("curiosité", 12),
            ("surprise", 8),
            ("calme", 7),
            ("tristesse", 3),
            ("peur", 2),
            ("neutre", 10),
        ]

        for emotion, count in emotions:
            for _ in range(count):
                self.emotional_learning.observe_emotion(emotion)

    def export_profile(self, instance):
        """Exporte le profil émotionnel dans un fichier JSON."""
        try:
            self.emotional_learning.export_profile("emotional_profile_export.json")
            # Feedback visuel temporaire
            instance.text = "Exporté !"
            instance.background_color = (0.2, 0.8, 0.2, 1.0)
            # Remettre le texte d'origine après un délai
            Clock.schedule_once(lambda dt: setattr(instance, "text", "Exporter le profil"), 1.5)
            Clock.schedule_once(lambda dt: setattr(instance, "background_color", (0.2, 0.6, 0.8, 1.0)), 1.5)
        except Exception as e:
            print(f"Erreur lors de l'exportation: {e}")
            instance.text = "Erreur!"
            instance.background_color = (0.8, 0.2, 0.2, 1.0)
            Clock.schedule_once(lambda dt: setattr(instance, "text", "Exporter le profil"), 1.5)
            Clock.schedule_once(lambda dt: setattr(instance, "background_color", (0.2, 0.6, 0.8, 1.0)), 1.5)

    def go_back(self, instance):
        """Retourne à l'écran principal."""
        self.manager.current = "jeffrey_main"
