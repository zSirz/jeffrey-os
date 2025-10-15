"""
Module : emotion_phrase_generator.py
Description : Gère la transformation des phrases selon l'état émotionnel
"""

import random
from dataclasses import dataclass


@dataclass
class EmotionMetadata:
    humeur: str
    intensite: float
    couleur: str
    emoji: str
    lien_affectif: float


class EmotionPhraseGenerator:
    def __init__(self):
        self.emotion_prefixes: dict[str, list[tuple[str, str]]] = {
            "heureux": [("✨", "Avec joie"), ("🌟", "Dans la bonne humeur"), ("😊", "Le cœur léger")],
            "triste": [("💔", "Le cœur lourd"), ("🫂", "Avec douceur"), ("🌧️", "Dans la mélancolie")],
            "énervé": [("⚡", "Avec détermination"), ("🔥", "Fermement"), ("💪", "Avec conviction")],
            "amoureux": [("💖", "Avec tendresse"), ("💝", "Dans l'amour"), ("💕", "Avec passion")],
            "neutre": [("", ""), ("", ""), ("", "")],
        }

        self.emotion_colors = {
            "heureux": "#FFD700",  # Or
            "triste": "#4169E1",  # Bleu royal
            "énervé": "#FF4500",  # Rouge-orange
            "amoureux": "#FF69B4",  # Rose vif
            "neutre": "#808080",  # Gris
        }

    def adapter_phrase(
        self, texte: str, humeur: str, intensite: float, lien_affectif: float
    ) -> tuple[str, EmotionMetadata]:
        """
        Adapte une phrase selon l'état émotionnel

        Args:
            texte: La phrase à adapter
            humeur: L'émotion dominante
            intensite: L'intensité de l'émotion (0-1)
            lien_affectif: L'intensité du lien affectif (0-1)

        Returns:
            Tuple[str, EmotionMetadata]: La phrase adaptée et ses métadonnées
        """
        # Sélection du préfixe émotionnel
        prefixes = self.emotion_prefixes.get(humeur, self.emotion_prefixes["neutre"])
        emoji, prefix = random.choice(prefixes)

        # Adaptation de l'intensité
        if intensite > 0.8:
            prefix = f"{prefix} !"
        elif intensite < 0.3:
            prefix = f"{prefix}..."

        # Adaptation selon le lien affectif
        if lien_affectif > 0.7:
            prefix = f"Mon cher ami, {prefix.lower()}"

        # Création des métadonnées
        metadata = EmotionMetadata(
            humeur=humeur,
            intensite=intensite,
            couleur=self.emotion_colors[humeur],
            emoji=emoji,
            lien_affectif=lien_affectif,
        )

        # Construction de la phrase finale
        phrase_adaptee = f"{prefix} {texte}" if prefix else texte

        return phrase_adaptee, metadata
