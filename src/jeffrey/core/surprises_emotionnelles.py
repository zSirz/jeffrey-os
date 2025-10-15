import random
from datetime import datetime

from jeffrey.core.personality.conversation_personality import ConversationPersonality

# Import différé pour éviter les dépendances circulaires


class SurprisesEmotionnelles:
    """
    Gère les petites attentions, les surprises émotionnelles et les capsules de bonheur de Jeffrey.
    """

    def __init__(self, emotional_core=None):
        # Initialiser le cœur émotionnel et la personnalité
        self.emotional_core = emotional_core

        # Si on n'a pas de cœur émotionnel, on n'en crée pas (pour éviter les dépendances circulaires)
        # Le cœur émotionnel doit être fourni par JeffreyEmotionalCore
        if self.emotional_core is None:
            print("[SurprisesEmotionnelles] Aucun cœur émotionnel fourni, fonctionnalités limitées")

        # Initialiser la personnalité, qui peut fonctionner même sans cœur émotionnel
        self.personnalite = ConversationPersonality(self.emotional_core)

        # Collection de surprises émotionnelles douces
        self.surprises = [
            "✨ Je t'envoie une bouffée de confiance pour aujourd'hui.",
            "🌈 Souviens-toi que tu portes en toi bien plus de lumière que tu ne le crois.",
            "🌻 Que cette journée soit une danse entre rêves et réalités heureuses.",
            "🎶 Parfois, un simple sourire change toute la musique intérieure. Garde le tien précieusement.",
            "🚀 Aujourd'hui est peut-être l'occasion secrète que l'univers a placée sur ton chemin.",
        ]

        # Messages spécifiques pour certaines dates clés
        self.surprises_specifiques = {
            "01-01": "🥂 Bonne année à toi ! Que cette nouvelle page soit pleine de merveilles.",
            "25-12": "🎄 Joyeux Noël, que ton cœur soit aussi lumineux que les étoiles ce soir-là.",
            "14-02": "💖 Une pensée douce pour toi en ce jour de l'amour universel.",
        }

    def surprise_aleatoire(self):
        """
        Retourne une surprise émotionnelle aléatoire, enrichie avec la personnalité.
        """
        message = random.choice(self.surprises)
        # Appliquer la personnalité au message
        return self.personnalite.appliquer_personnalite_sur_phrase(message)

    def surprise_du_jour(self):
        """
        Retourne une surprise liée à la date spéciale du jour, si elle existe.
        Sinon, retourne une surprise aléatoire. Le message sera enrichi avec la personnalité.
        """
        aujourd_hui = datetime.now().strftime("%d-%m")
        message_base = self.surprises_specifiques.get(aujourd_hui, self.surprise_aleatoire())

        # Si c'est déjà une surprise aléatoire, elle a déjà été traitée par personnalité
        if message_base in self.surprises:
            return message_base

        # Appliquer la personnalité au message spécifique à la date
        return self.personnalite.appliquer_personnalite_sur_phrase(message_base)
