# PACK 11: Système de relation
from core.personality.relation_tracker_manager import get_niveau_relation, get_relation_tracker
from kivy.clock import Clock
from kivy.lang import Builder
from kivy.properties import NumericProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout

# PACK 12: Système de lien émotionnel évolutif
try:
    from core.emotions.emotional_affective_touch import get_emotional_affective_touch, obtenir_lien

    PACK12_AVAILABLE = True
except ImportError:
    PACK12_AVAILABLE = False

    # Fonction de fallback si le Pack 12 n'est pas disponible
    def obtenir_lien(user_id: str) -> float:
        # Fallback to Pack 11
        return get_niveau_relation()


Builder.load_string(
    """
<LienAffectifWidget>:
    orientation: 'vertical'
    padding: 10
    spacing: 5
    size_hint: None, None
    size: dp(180), dp(100)

    canvas.before:
        Color:
            rgba: 1, 0.6, 0.8, 0.2
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [10]

    Label:
        id: heart_level
        text: root.link_description
        font_size: '16sp'
        halign: 'center'
        color: 1, 0.2, 0.5, 1

    ProgressBar:
        id: affection_bar
        max: 1.0
        value: root.link_strength
        height: dp(10)
        size_hint_y: None
        pos_hint: {"center_x": 0.5}
"""
)


class LienAffectifWidget(BoxLayout):
    link_strength = NumericProperty(0.0)
    link_description = StringProperty("Lien inconnu...")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        Clock.schedule_once(self.update_link_state, 0.5)

    def update_link_state(self, *args):
        # PACK 12: Utiliser le système de lien émotionnel s'il est disponible
        if PACK12_AVAILABLE:
            try:
                # Obtenir le lien émotionnel pour l'utilisateur actuel (supposé être "David" par défaut)
                user_id = "David"  # L'utilisateur principal par défaut

                # Obtenir le niveau de lien
                niveau_lien = obtenir_lien(user_id)

                # Obtenir le système complet pour des statistiques détaillées
                lien_system = get_emotional_affective_touch()
                stats = lien_system.obtenir_statistiques_lien(user_id)

                # Mettre à jour l'affichage
                self.link_strength = niveau_lien
                self.link_description = stats.get("description", f"Lien {stats.get('categorie', 'modéré')}")

                return
            except Exception as e:
                print(f"Erreur lors de l'accès au système de lien émotionnel: {e}")
                # Fallback to Pack 11 if there's an error

        # PACK 11: Fallback si Pack 12 n'est pas disponible ou en erreur
        try:
            # Obtenir les informations du système de relation (Pack 11)
            niveau = get_niveau_relation()
            profil = get_relation_tracker().get_profil_relation()

            # Mettre à jour l'affichage
            self.link_strength = niveau
            self.link_description = f"Lien {profil}"
        except Exception:
            # Valeurs par défaut en cas d'erreur
            self.link_strength = 0.5
            self.link_description = "Lien en construction..."
