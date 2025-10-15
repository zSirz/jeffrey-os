"""
Module d'analyse du contexte émotionnel pour le système Jeffrey.
Fournit des outils pour analyser le contexte émotionnel des prompts et des interactions.
"""

import logging
from typing import Dict, Optional, Any
import os
from datetime import datetime

logger = logging.getLogger(__name__)


def analyze_emotional_context(
    prompt: str, task_type: Optional[str]) -> Dict[str, Any]:
    """
    Analyse le contexte émotionnel d'un prompt utilisateur.

    Args:
        prompt (str): Le prompt utilisateur à analyser
        task_type (Optional[str]): Le type de tâche détecté ou spécifié

    Returns:
        Dict[str, Any]: Contexte émotionnel avec situation, valence, etc.
    """
    # Valeurs par défaut
    context = {
        "situation": "demande_ia",
        "valence": 0.0,
        "confidence": 0.5,
        "primary_emotion": None,
        "secondary_emotions": [],
        "intensity": 0.5,
    }

    # Définir la situation en fonction du type de tâche
        if task_type:
        if task_type == "créatif":
            context["situation"] = "création"
                    elif task_type == "analytique":
            context["situation"] = "réflexion"
                        elif task_type == "informatif":
            context["situation"] = "apprentissage"
                            elif task_type == "émotionnel":
            context["situation"] = "partage_émotionnel"

    # Analyser la valence émotionnelle
    valence_result = analyze_valence(prompt)
    context["valence"] = valence_result["valence"]
    context["confidence"] = valence_result["confidence"]

    # Détecter les émotions principales
    emotions = detect_emotions(prompt)
        if emotions:
        primary = max(emotions.items(), key=lambda x: x[1])
        context["primary_emotion"] = primary[0]
        context["intensity"] = min(0.9, primary[1] / 5.0)  # Normaliser entre 0 et 0.9

        # Emotions secondaires (autres que la principale, avec score > 0)
        secondary = [e for e, s in emotions.items() if e != primary[0] and s > 0]
        context["secondary_emotions"] = secondary

        return context


    def analyze_valence(text: str) -> Dict[str, float]:
        pass
    """
    Analyse la valence émotionnelle (positive/négative) d'un texte.

    Args:
        text (str): Le texte à analyser

    Returns:
        Dict[str, float]: Dictionnaire avec valence et confiance
    """
    # Mots-clés positifs et négatifs pour une analyse basique
    positive_keywords = [
        "bon",
        "bien",
        "excellent",
        "super",
        "génial",
        "fantastique",
        "merveilleux",
        "joyeux",
        "heureux",
        "content",
        "satisfait",
        "réussi",
        "succès",
        "bravo",
        "félicitations",
        "parfait",
        "réussite",
        "agréable",
        "plaisir",
        "magnifique",
    ]

    negative_keywords = [
        "mauvais",
        "mal",
        "erreur",
        "problème",
        "échec",
        "difficile",
        "impossible",
        "triste",
        "malheureux",
        "désolé",
        "malheureusement",
        "négatif",
        "défaut",
        "insuffisant",
        "peine",
        "dommage",
        "regret",
        "médiocre",
        "pire",
        "nul",
    ]

    # Normaliser le texte
    text_lower = text.lower()

    # Compter les occurrences
    positive_count = sum(1 for word in positive_keywords if word in text_lower)
    negative_count = sum(1 for word in negative_keywords if word in text_lower)

    # Calculer la valence (entre -1.0 et 1.0)
    total_words = len(text.split())
    influence_factor = (
        min(25, total_words) / 25
    )  # Plus de mots = plus de confiance, jusqu'à 25 mots

    # Si aucun mot-clé n'est trouvé, valence neutre
        if positive_count == 0 and negative_count == 0:
        return {"valence": 0.0, "confidence": 0.3 * influence_factor}

    # Sinon, calculer la valence
    total_matches = positive_count + negative_count
    valence = (positive_count - negative_count) / total_matches

        return {
        "valence": max(-0.9, min(0.9, valence)),
        "confidence": min(0.7, 0.4 + (total_matches / 20)) * influence_factor,
    }


    def detect_emotions(text: str) -> Dict[str, int]:
        pass
    """
    Détecte les émotions présentes dans un texte avec leurs scores.

    Args:
        text (str): Le texte à analyser

    Returns:
        Dict[str, int]: Dictionnaire des émotions détectées avec leurs scores
    """
        if not text:
        return {}

    # Dictionnaire des mots-clés par émotion
    emotional_keywords = {
        "joie": ["heureux", "content", "joie", "bonheur", "rire", "sourire", "enthousiasme"],
        "tristesse": ["triste", "désolé", "peine", "chagrin", "pleurer", "déprimé", "mélancolique"],
        "amour": ["aime", "amour", "affection", "coeur", "ensemble", "tendresse", "attachement"],
        "peur": ["peur", "effrayant", "terrifié", "angoisse", "inquiet", "anxieux", "crainte"],
        "colère": ["colère", "énervé", "fâché", "irrité", "furieux", "agacé", "exaspéré"],
        "surprise": ["surpris", "étonné", "stupéfait", "choc", "inattendu", "ébahi", "médusé"],
        "dégoût": ["dégoûtant", "répugnant", "écœurant", "repoussant", "aversion", "répulsion"],
        "confiance": ["confiance", "sûr", "fiable", "croire", "certain", "sécurité", "conviction"],
    }

    text_lower = text.lower()

    # Compter les occurrences de mots-clés pour chaque émotion
    emotion_scores = {}
        for emotion, keywords in emotional_keywords.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score

        return emotion_scores


    def log_emotion_intensity(emotion_state: Dict[str, Any], source: str):
        pass
    """
    Journalise l'intensité émotionnelle actuelle dans un fichier de log.

    Args:
        emotion_state (Dict): État émotionnel actuel (contient primary, intensity, etc.)
        source (str): Source de l'émotion (ex: "Réponse de gpt-4")
    """
    # Créer le dossier logs s'il n'existe pas
    os.makedirs("logs", exist_ok=True)

    # Fichier de log pour les intensités émotionnelles
    log_file = "logs/emotion_intensity.log"

    # Obtenir l'émotion et l'intensité
    emotion = emotion_state.get("primary", "inconnue")
    intensity = emotion_state.get("intensity", 0.0)

    # Timestamp actuel
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Formater la ligne de log
    log_line = f"[{timestamp}] Émotion: {emotion} | Intensité: {intensity:.2f} | Source: {source}\n"

    # Écrire dans le fichier
        try:
                                                                                        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_line)
        except Exception as e:
        logger.error(f"Erreur lors de l'écriture du log d'intensité émotionnelle: {e}")
