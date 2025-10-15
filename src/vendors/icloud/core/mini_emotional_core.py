#!/usr/bin/env python

"""
Mini cerveau émotionnel pour Jeffrey.
Cette implémentation comprend la gestion des poids émotionnels
et l'oubli progressif des émotions au fil du temps.
"""

import math
from datetime import datetime
from typing import Dict, Optional


class MiniEmotionalCore:
    """
    Mini cerveau émotionnel de Jeffrey pour affiner son évolution émotionnelle.
    Gère un historique d'émotions avec des poids et une décroissance temporelle.
    """

    def __init__(self):
        """Initialise le mini cerveau émotionnel avec des historiques vides."""
        # Structure: liste d'émotions simples pour compatibilité avec l'ancien code
        self.historique_emotions = []

        # Structure: dictionnaire d'intensités pour compatibilité avec les tests
        self.emotions_apprises = {}

        # Nouvelle structure: liste de tuples (émotion, poids, timestamp)
        self.emotions_ponderees = []

        # Facteur de décroissance (plus grand = oubli plus rapide)
        self.facteur_oubli = 0.01  # 1% de décroissance par jour

        # Taille maximale de l'historique
        self.taille_max_historique = 100

    def enregistrer_emotion(self, emotion: str) -> None:
        """
        Enregistre une émotion dans l'historique (compatibilité).
        """
        self.historique_emotions.append(emotion)
        if len(self.historique_emotions) > self.taille_max_historique:
            self.historique_emotions.pop(0)

        # Utiliser aussi la nouvelle méthode avec poids par défaut
        self.enregistrer_emotion_ponderee(emotion, 1.0)

    def enregistrer_emotion_ponderee(self, emotion: str, poids: float) -> None:
        """
        Enregistre une émotion avec un poids spécifique et un timestamp.
        """
        if not isinstance(emotion, str) or not emotion.strip():
            raise ValueError("L'émotion doit être une chaîne non vide")
        if not isinstance(poids, (int, float)) or poids <= 0:
            raise ValueError("Le poids doit être un nombre positif")

        maintenant = datetime.now().timestamp()
        self.emotions_ponderees.append((emotion, float(poids), maintenant))

        if len(self.emotions_ponderees) > self.taille_max_historique:
            self.emotions_ponderees.pop(0)

    def emotion_predominante(self) -> Optional[str]:
        """
        Détermine l'émotion la plus fréquente récemment (compatibilité).
        """
        compteur: Dict[str, int] = {}
        for emotion in self.historique_emotions:
            compteur[emotion] = compteur.get(emotion, 0) + 1
        if not compteur:
            return None
        return max(compteur, key=compteur.get)

    def emotion_predominante_ponderee(self) -> Optional[str]:
        """
        Détermine l'émotion dominante en tenant compte des poids et de l'oubli progressif.
        """
        if not self.emotions_ponderees:
            return None

        maintenant = datetime.now().timestamp()
        compteur_pondere: Dict[str, float] = {}

        for emotion, poids, timestamp in self.emotions_ponderees:
            age_jours = (maintenant - timestamp) / (24 * 3600)
            facteur_oubli = math.exp(-self.facteur_oubli * age_jours)
            poids_actuel = poids * facteur_oubli
            compteur_pondere[emotion] = compteur_pondere.get(emotion, 0.0) + poids_actuel

        if not compteur_pondere:
            return None
        return max(compteur_pondere, key=compteur_pondere.get)

    def get_emotions_ponderees(self) -> Dict[str, float]:
        """
        Obtient toutes les émotions avec leurs poids actuels après décroissance.
        """
        maintenant = datetime.now().timestamp()
        emotions: Dict[str, float] = {}

        for emotion, poids, timestamp in self.emotions_ponderees:
            age_jours = (maintenant - timestamp) / (24 * 3600)
            facteur_oubli = math.exp(-self.facteur_oubli * age_jours)
            poids_actuel = poids * facteur_oubli
            emotions[emotion] = emotions.get(emotion, 0.0) + poids_actuel

        return emotions

    def ajuster_facteur_oubli(self, facteur: float) -> None:
        """
        Ajuste le facteur d'oubli (0.01 = 1% par jour).
        """
        if facteur < 0:
            raise ValueError("Le facteur d'oubli doit être positif")
        self.facteur_oubli = float(facteur)

    # === Compatibilité tests ===

    def learn_emotion(self, emotion: str, intensity: float) -> None:
        """
        Apprend une émotion avec son intensité (compatibilité tests).
        """
        self.emotions_apprises[emotion] = float(intensity)
        self.enregistrer_emotion_ponderee(emotion, intensity)

    def get_dominant_emotion(self) -> Optional[str]:
        """
        Retourne l'émotion dominante selon l'intensité (compatibilité tests).
        """
        if not self.emotions_apprises:
            return None
        return max(self.emotions_apprises, key=self.emotions_apprises.get)
