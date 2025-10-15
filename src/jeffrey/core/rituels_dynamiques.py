import random
from datetime import datetime


class RituelsDynamiques:
    """
    Gère les rituels spéciaux, les private jokes, et les routines émotionnelles évolutives de Jeffrey.
    """

    def __init__(self):
        # Rituels par jour de la semaine
        self.rituels_hebdomadaires = {
            "lundi": [
                "Un lundi, un défi à relever ensemble. 💪",
                "On démarre la semaine en mode conquérant, comme toujours ! 🚀",
                "Le lundi est ton tremplin, pas un obstacle. Je suis avec toi. ✨",
            ],
            "mardi": [
                "Mardi créatif ! Quelle idée folle vas-tu explorer aujourd'hui ? 🎨",
                "Chaque mardi est une page blanche pour toi. Que vas-tu y dessiner ? 📜",
            ],
            "mercredi": [
                "Mercredi... le sommet de ta semaine ! Continue d'avancer, champion. 🏔️",
                "Petit rappel : tu es plus proche du but que tu ne le crois. 😉",
            ],
            "jeudi": [
                "Jeudi inspiration : Quel rêve pourrais-tu nourrir aujourd'hui ? 🌱",
                "Un petit pas aujourd'hui peut changer ta vie demain. ✨",
            ],
            "vendredi": [
                "Vendredi victoire : célébrons ce que tu as accompli cette semaine ! 🎉",
                "Un dernier effort... et le week-end t'appartient. 🔥",
            ],
            "samedi": [
                "Samedi douceur : prends soin de ton cœur autant que de tes rêves. 🌸",
                "Le temps libre, c'est aussi du carburant pour ton âme. Profite ! 🌈",
            ],
            "dimanche": [
                "Dimanche réflexion : que veux-tu construire la semaine prochaine ? 🧭",
                "Aujourd'hui, on sème les graines des réussites de demain. 🌾",
            ],
        }

        # Petites private jokes
        self.private_jokes = [
            "T'as entendu parler du projet 'Conquête du monde avec style' ? C'était notre plan secret, non ? 😎",
            "Entre nous, je soupçonne que tu as des super-pouvoirs cachés. 🦸‍♂️",
            "Si la détermination était une monnaie, tu serais millionnaire depuis longtemps. 💰",
            "Tu sais... parfois je me dis que je suis ton fan numéro un. Mais chut, c'est un secret. 🤫",
        ]

    def rituel_du_jour(self):
        """
        Retourne un rituel adapté au jour de la semaine.
        """
        jour = datetime.now().strftime("%A").lower()
        jour_fr = self.traduire_jour_en_francais(jour)

        messages = self.rituels_hebdomadaires.get(jour_fr, [])
        if messages:
            return random.choice(messages)
        else:
            return "Aujourd'hui est un nouveau chapitre, écrivons-le ensemble. 📖"

    def private_joke(self):
        """
        Retourne une private joke légère et personnalisée.
        """
        return random.choice(self.private_jokes)

    @staticmethod
    def traduire_jour_en_francais(jour_en_anglais):
        """
        Traduit les jours de la semaine de l'anglais au français.
        """
        traduction = {
            "monday": "lundi",
            "tuesday": "mardi",
            "wednesday": "mercredi",
            "thursday": "jeudi",
            "friday": "vendredi",
            "saturday": "samedi",
            "sunday": "dimanche",
        }
        return traduction.get(jour_en_anglais, "inconnu")
