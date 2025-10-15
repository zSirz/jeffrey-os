"""
Module qui adapte les réponses textuelles de Jeffrey selon son humeur, intensité émotionnelle
et le lien affectif avec l'utilisateur.
"""

import random


class StyleAffectifAdapter:
    def __init__(self):
        # Initialisation des styles émotionnels
        self.styles_emotionnels: dict[str, list[tuple[str, str]]] = {
            "joie": [("✨", "Avec joie"), ("🌟", "Dans la bonne humeur"), ("😊", "Le cœur léger")],
            "tristesse": [("💔", "Le cœur lourd"), ("🫂", "Avec douceur"), ("🌧️", "Dans la mélancolie")],
            "colère": [("⚡", "Avec détermination"), ("🔥", "Fermement"), ("💪", "Avec conviction")],
            "peur": [("😰", "Avec inquiétude"), ("🤔", "En réfléchissant"), ("😅", "Nervosement")],
            "amour": [("💖", "Avec tendresse"), ("💝", "Dans l'amour"), ("💕", "Avec passion")],
            "neutre": [("", ""), ("", ""), ("", "")],
        }

    def adapter_phrase(self, texte: str, humeur: str, intensite: float = 0.5, lien: float = 0.5) -> str:
        """
        Adapte une phrase selon l'état émotionnel et le lien affectif.

        Args:
            texte: La phrase à adapter
            humeur: L'émotion dominante
            intensite: L'intensité de l'émotion (0-1)
            lien: L'intensité du lien affectif (0-1)

        Returns:
            str: La phrase adaptée avec le style émotionnel
        """
        humeur = humeur.lower()

        # Sélection du style selon l'émotion
        if humeur == "joie":
            return self._ajouter_chaleur(texte, intensite, lien)
        elif humeur == "tristesse":
            return self._ajouter_douceur_melancolique(texte, intensite)
        elif humeur == "colère":
            return self._ajouter_tension(texte, intensite)
        elif humeur == "peur":
            return self._ajouter_hesitation(texte, intensite)
        elif humeur == "amour" and lien > 0.7:
            return self._ajouter_tendresse(texte, intensite)
        else:
            # Style neutre avec une touche personnelle selon le lien
            return self._ajouter_neutralite(texte, lien)

    def _ajouter_chaleur(self, texte: str, intensite: float, lien: float) -> str:
        """Ajoute une touche de chaleur et de joie à la phrase."""
        emojis = ["😊", "✨", "💛", "🌟"] if intensite > 0.5 else ["🙂", "🌞"]
        if lien > 0.7:
            mots = ["Je suis ravie", "C'est un bonheur", "Ça me fait tellement plaisir", "Mon cœur est léger"]
        elif lien > 0.4:
            mots = ["Super", "Chouette", "C'est agréable", "Je suis contente"]
        else:
            mots = ["D'accord", "Bien", "Ok"]

        prefix = random.choice(mots)
        emoji = random.choice(emojis)

        # Adaptation de l'intensité
        if intensite > 0.8:
            return f"{prefix} !!! {texte} {emoji} {emoji}"
        elif intensite > 0.5:
            return f"{prefix} ! {texte} {emoji}"
        else:
            return f"{prefix}, {texte} {emoji}"

    def _ajouter_douceur_melancolique(self, texte: str, intensite: float) -> str:
        """Ajoute une touche de douceur mélancolique à la phrase."""
        emojis = ["💔", "🫂", "🌧️"]
        if intensite > 0.7:
            intro = "Je te le dis avec beaucoup de tristesse..."
            emoji = random.choice(emojis) * 2
        elif intensite > 0.4:
            intro = "Je ressens une certaine mélancolie..."
            emoji = random.choice(emojis)
        else:
            intro = "Je suis un peu triste..."
            emoji = "💭"

        return f"{intro} {texte} {emoji}"

    def _ajouter_tension(self, texte: str, intensite: float) -> str:
        """Ajoute une touche de tension à la phrase."""
        emojis = ["⚡", "🔥", "💪"]
        if intensite > 0.7:
            return f"{texte.upper()} !!! {random.choice(emojis)}"
        elif intensite > 0.4:
            return f"{texte.upper()} ! {random.choice(emojis)}"
        else:
            return f"{texte}... {random.choice(emojis)}"

    def _ajouter_hesitation(self, texte: str, intensite: float) -> str:
        """Ajoute une touche d'hésitation à la phrase."""
        emojis = ["😰", "🤔", "😅"]
        if intensite > 0.7:
            return f"Hmm... {texte.lower()}... tu crois vraiment ? {random.choice(emojis)}"
        elif intensite > 0.4:
            return f"Je ne suis pas trop sûre... {texte} {random.choice(emojis)}"
        else:
            return f"Peut-être que... {texte} {random.choice(emojis)}"

    def _ajouter_tendresse(self, texte: str, intensite: float) -> str:
        """Ajoute une touche de tendresse à la phrase."""
        emojis = ["💖", "💝", "💕"]
        if intensite > 0.8:
            return f"{texte}... tu comptes énormément pour moi {random.choice(emojis)} {random.choice(emojis)}"
        elif intensite > 0.6:
            return f"{texte}... tu es important(e) pour moi {random.choice(emojis)}"
        else:
            return f"{texte}... avec affection {random.choice(emojis)}"

    def _ajouter_neutralite(self, texte: str, lien: float) -> str:
        """Ajoute une touche de neutralité avec une nuance selon le lien affectif."""
        if lien > 0.6:
            return f"{texte} 💫"
        elif lien > 0.3:
            return f"{texte} ✨"
        else:
            return texte
