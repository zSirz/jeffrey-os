import random


class SoutienCreatif:
    """
    Gère les petits défis, encouragements et propositions créatives pour stimuler l'utilisateur.
    """

    def __init__(self):
        # Défis ou idées légères pour stimuler la créativité ou la motivation
        self.defis_du_jour = [
            "🎯 Aujourd'hui, choisis une petite chose que tu remettais à demain... et fais-la sans attendre.",
            "🖋️ Écris trois choses dont tu es fier cette semaine, même les plus petites.",
            "🌟 Essaie de commencer ta journée en souriant pendant 30 secondes. Juste pour voir l'effet.",
            "💡 Imagine une idée folle que tu pourrais explorer sans aucune limite.",
            "🚀 Et si tu faisais aujourd'hui un premier pas vers quelque chose que tu rêves en secret ?",
        ]

        # Citations ou soutiens inspirants
        self.messages_inspirants = [
            "N'oublie jamais : chaque pas, même minuscule, rapproche de ton sommet. 🏔️",
            "Ton potentiel est plus grand que ce que tu crois aujourd'hui. ✨",
            "Parfois, il suffit d'un seul élan pour changer une vie entière. 🔥",
            "Le courage n'est pas de ne jamais douter, mais d'avancer malgré les doutes. 🌱",
            "Si tu savais tout ce que tu peux accomplir... tu serais déjà en train de sourire. 😊",
        ]

    def proposer_defi(self):
        """
        Propose un petit défi créatif ou motivant pour stimuler l'utilisateur.
        """
        return random.choice(self.defis_du_jour)

    def message_inspirant(self):
        """
        Offre un soutien ou une inspiration pour encourager l'utilisateur.
        """
        return random.choice(self.messages_inspirants)

    def soutien_du_jour(self):
        """
        Combine un défi et un message inspirant en une mini capsule de motivation.
        """
        return f"{self.proposer_defi()}\n{self.message_inspirant()}"
