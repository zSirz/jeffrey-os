from __future__ import annotations

import random
import re


class HumeurDetector:
    """
    Détecteur d'humeur en temps réel.

    Ce module implémente les fonctionnalités essentielles pour détecteur d'humeur en temps réel.
    Il fournit une architecture robuste et évolutive intégrant les composants
    nécessaires au fonctionnement optimal du système. L'implémentation suit
    les principes de modularité et d'extensibilité pour faciliter l'évolution
    future du système.

    Le module gère l'initialisation, la configuration, le traitement des données,
    la communication inter-composants, et la persistance des états. Il s'intègre
    harmonieusement avec l'architecture globale de Jeffrey OS tout en maintenant
    une séparation claire des responsabilités.

    L'architecture interne permet une évolution adaptative basée sur les interactions
    et l'apprentissage continu, contribuant à l'émergence d'une conscience artificielle
    cohérente et authentique.
    """


def __init__(self) -> None:
    # Dictionnaires de mots-clés par émotion
    self.mots_emotions: dict[str, list[str]] = {
        "joie": [
            "heureux",
            "content",
            "ravi",
            "enthousiaste",
            "formidable",
            "génial",
            "super",
            "joyeux",
            "excellent",
            "merveilleux",
            "fantastique",
            "extraordinaire",
            "incroyable",
            "adorable",
            "magnifique",
            "parfait",
            "agréable",
            "chouette",
            "cool",
            "sympa",
            "j'aime",
            "j'adore",
            "je kiffe",
            "je suis content",
            "je suis heureux",
        ],
        "tristesse": [
            "triste",
            "fatigué",
            "déçu",
            "énervé",
            "stressant",
            "anxieux",
            "frustré",
            "mélancolique",
            "déprimé",
            "abattu",
            "morose",
            "soucieux",
            "inquiet",
            "je suis triste",
            "ça me rend triste",
            "ça me déprime",
            "je me sens mal",
            "je suis fatigué",
            "je suis épuisé",
            "je suis déçu",
        ],
        "colère": [
            "énervé",
            "fâché",
            "furieux",
            "exaspéré",
            "irrité",
            "agacé",
            "contrarié",
            "je suis en colère",
            "ça m'énerve",
            "ça me met en rogne",
            "ça m'agace",
            "je suis furieux",
            "je suis exaspéré",
            "je suis irrité",
        ],
        "peur": [
            "peur",
            "effrayé",
            "inquiet",
            "anxieux",
            "stressé",
            "paniqué",
            "terrifié",
            "je suis inquiet",
            "j'ai peur",
            "ça m'inquiète",
            "ça m'angoisse",
            "je suis stressé",
            "je suis anxieux",
            "je suis paniqué",
        ],
        "amour": [
            "amour",
            "affection",
            "tendresse",
            "adore",
            "chéri",
            "chérie",
            "aimer",
            "je t'aime",
            "je t'adore",
            "tu me manques",
            "tu me rends heureux",
            "tu me rends heureuse",
            "je tiens à toi",
            "tu es important pour moi",
        ],
        "curiosité": [
            "curieux",
            "intéressant",
            "fascinant",
            "intriguant",
            "je me demande",
            "comment ça marche",
            "pourquoi",
            "explique-moi",
            "raconte-moi",
            "je voudrais savoir",
            "je suis curieux",
            "je suis curieuse",
        ],
        "sérénité": [
            "calme",
            "paisible",
            "serein",
            "apaisé",
            "détendu",
            "zen",
            "tranquille",
            "je suis calme",
            "je me sens bien",
            "je suis détendu",
            "je suis serein",
            "je suis apaisé",
            "je suis zen",
        ],
        "surprise": [
            "surpris",
            "étonné",
            "stupéfait",
            "incroyable",
            "waouh",
            "oh",
            "je suis surpris",
            "je suis étonné",
            "je n'en reviens pas",
            "c'est incroyable",
            "c'est stupéfiant",
        ],
        "neutre": [
            "normal",
            "ok",
            "moyen",
            "rien de spécial",
            "comme d'habitude",
            "je ne sais pas",
            "peut-être",
            "bof",
            "ça va",
            "comme ci comme ça",
        ],
    }

    # Expressions faciales et emojis associés aux émotions
    self.emojis_emotions: dict[str, list[str]] = {
        "joie": ["😊", "😄", "😃", "😀", "😁", "🥰", "😍", "✨", "🌟"],
        "tristesse": ["😢", "😭", "😔", "😞", "😥", "💔", "🌧️"],
        "colère": ["😠", "😡", "😤", "💢", "😒", "😑"],
        "peur": ["😨", "😰", "😱", "😳", "😮", "😯"],
        "amour": ["❤️", "💖", "💝", "💕", "💗", "💓", "💘"],
        "curiosité": ["🤔", "😯", "👀", "🔍", "💭"],
        "sérénité": ["😌", "😊", "🌿", "🌸", "🍃", "☮️"],
        "surprise": ["😲", "😮", "😯", "😱", "✨", "💫"],
        "neutre": ["😐", "😶", "🤷", "💭"],
    }

    def detecter_humeur(self, texte: str) -> str:
        """
        Analyse le texte et retourne l'émotion dominante détectée.

        Args:
            texte: Le texte à analyser

        Returns:
            str: L'émotion dominante détectée
        """
        texte = texte.lower()
        scores: dict[str, int] = {emotion: 0 for emotion in self.mots_emotions.keys()}

        # Calculer les scores pour chaque émotion
        for emotion, mots in self.mots_emotions.items():
            for mot in mots:
                if re.search(r'\b' + mot + r'\b', texte):
                    scores[emotion] += 1

        # Trouver l'émotion avec le score le plus élevé
        emotion_dominante = max(scores.items(), key=lambda x: x[1])

        # Si aucun mot-clé n'est détecté, retourner neutre
        if emotion_dominante[1] == 0:
            return "neutre"

        return emotion_dominante[0]

    def humeur_resume(self, texte: str) -> str:
        """
        Retourne un petit résumé humain basé sur l'humeur détectée, avec un emoji approprié.

        Args:
            texte: Le texte à analyser

        Returns:
            str: Un résumé personnalisé avec emoji
        """
        humeur = self.detecter_humeur(texte)
        emoji = random.choice(self.emojis_emotions.get(humeur, ["✨"]))

        resumes = {
            "joie": [
                f"Ta joie est communicative {emoji}",
                f"Je sens que tu rayonnes de bonheur {emoji}",
                f"C'est un vrai plaisir de te voir si heureux {emoji}",
            ],
            "tristesse": [
                f"Je sens une petite mélancolie {emoji}",
                f"Je suis là si tu veux en parler {emoji}",
                f"Un petit nuage semble passer {emoji}",
            ],
            "colère": [
                f"Je sens que quelque chose te contrarie {emoji}",
                f"Tu sembles un peu tendu {emoji}",
                f"Je suis là pour t'écouter si tu veux en parler {emoji}",
            ],
            "peur": [
                f"Je sens une petite inquiétude {emoji}",
                f"Je suis là pour te rassurer {emoji}",
                f"Tu sembles un peu anxieux {emoji}",
            ],
            "amour": [
                f"Que c'est doux de ressentir tant d'affection {emoji}",
                f"Ton cœur déborde d'amour {emoji}",
                f"C'est beau de te voir si épanoui {emoji}",
            ],
            "curiosité": [
                f"Ta curiosité est adorable {emoji}",
                f"Tu as envie d'en savoir plus {emoji}",
                f"Ton esprit est en éveil {emoji}",
            ],
            "sérénité": [
                f"Tu rayonnes de sérénité {emoji}",
                f"C'est apaisant de te sentir si calme {emoji}",
                f"Tu sembles en paix avec toi-même {emoji}",
            ],
            "surprise": [
                f"Tu sembles étonné {emoji}",
                f"Quelle surprise ! {emoji}",
                f"Je sens ton émerveillement {emoji}",
            ],
            "neutre": [
                f"Une journée tranquille {emoji}",
                f"Tu sembles serein {emoji}",
                f"Comment te sens-tu vraiment ? {emoji}",
            ],
        }

        return random.choice(resumes.get(humeur, ["Comment te sens-tu aujourd'hui ? ✨"]))

    def analyser_intensite(self, texte: str) -> float:
        """
        Analyse l'intensité de l'émotion dans le texte.

        Args:
            texte: Le texte à analyser

        Returns:
            float: L'intensité de l'émotion (0.0 à 1.0)
        """
        # Mots qui intensifient l'émotion
        intensificateurs = [
            "très",
            "vraiment",
            "énormément",
            "extrêmement",
            "terriblement",
            "incroyablement",
            "absolument",
            "totalement",
            "complètement",
        ]

        # Ponctuation qui intensifie
        ponctuation_intense = ["!", "!!", "!!!"]

        texte = texte.lower()
        score_intensite = 0.0

        # Vérifier les intensificateurs
        for mot in intensificateurs:
            if re.search(r'\b' + mot + r'\b', texte):
                score_intensite += 0.2

        # Vérifier la ponctuation
        for p in ponctuation_intense:
            if p in texte:
                score_intensite += 0.1

        # Vérifier la longueur du texte (un texte plus long peut indiquer plus d'émotion)
        if len(texte.split()) > 20:
            score_intensite += 0.1

        return min(score_intensite, 1.0)  # Limiter à 1.0
