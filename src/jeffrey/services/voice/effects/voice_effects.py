"""
Moteur d'effets vocaux pour Jeffrey - ajoute des effets spéciaux aux voix générées.

Ce module permet d'ajouter des effets vocaux réalistes comme des hésitations,
des rires légers, des soupirs, et d'autres nuances vocales pour rendre
les interactions plus naturelles et expressives.
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any

# Liste des marqueurs d'effets vocaux
VOCAL_EFFECTS = {
    "sigh": "💨",  # soupir
    "pause": "...",
    "laugh": "*petit rire*",
    "sparkle": "*sourire dans la voix*",
    "tremble": "*voix tremblante*",
    "whisper": "*chuchote*",
    "growl": "*voix grondante*",
    "soft": "*voix posée*",
    "soft_breath": "*respiration légère*",
    "hesitation": "hmm...",
    "thinking": "euh...",
    "surprise": "*gasp*",
    "excited": "wow!",
    "sad_sigh": "*soupir mélancolique*",
    "happy_sigh": "*soupir heureux*",
    "nervous": "*nerveux*",
}

# Variations des effets vocaux
EFFECT_VARIATIONS = {
    "laugh": ["*petit rire*", "*haha*", "*rit doucement*", "*rire léger*"],
    "sigh": ["*soupir*", "*souffle*", "*long soupir*", "*soupire*"],
    "hesitation": ["hmm...", "euh...", "mmm...", "hem..."],
    "pause": ["...", ". . .", "…", "   "],
    "thinking": ["euh...", "hmm, voyons...", "alors...", "laissez-moi réfléchir..."],
    "soft_breath": ["*respire doucement*", "*respiration légère*", "*inspire légèrement*"],
    "surprise": ["*gasp*", "*oh!*", "*inspiration surprise*", "*surprise*"],
    "whisper": ["*chuchote*", "*baisse la voix*", "*en chuchotant*"],
}

# Patrons d'effets automatiques basés sur la ponctuation
AUTO_EFFECT_PATTERNS = [
    (r"(?<!\.)\.\.\.(?!\.)", "pause"),  # Trois points consécutifs
    (r"\!\?", "surprise"),  # Point d'exclamation suivi d'un point d'interrogation
    (r"\!\!+", "excited"),  # Plusieurs points d'exclamation consécutifs
    (r"--", "hesitation"),  # Double tiret
    (r"euh\s|hum\s|hmm\s", "thinking"),  # Marqueurs explicites d'hésitation
    (r"haha|hihi|lol", "laugh"),  # Marqueurs de rire
]


class VoiceEffects:
    """
    Gestionnaire d'effets vocaux qui ajoute et traite des effets vocaux réalistes
    pour enrichir la synthèse vocale de Jeffrey.
    """

    def __init__(self, effect_intensity: float = 0.5) -> None:
        """
        Initialise le gestionnaire d'effets vocaux.

        Args:
            effect_intensity: Intensité globale des effets (0.0-1.0)
        """
        self.logger = logging.getLogger(__name__)
        self.effect_intensity = max(0.0, min(1.0, effect_intensity))

        # Dictionnaire des effets activés/désactivés
        self.enabled_effects = {effect: True for effect in VOCAL_EFFECTS}

        # Compteurs pour éviter la répétition excessive des effets
        self.effect_usage_count = {effect: 0 for effect in VOCAL_EFFECTS}

    def set_effect_intensity(self, intensity: float) -> None:
        """
        Définit l'intensité globale des effets vocaux.

        Args:
            intensity: Nouvelle intensité (0.0-1.0)
        """
        self.effect_intensity = max(0.0, min(1.0, intensity))
        self.logger.debug(f"Intensité des effets vocaux définie à {self.effect_intensity}")

    def enable_effect(self, effect_name: str, enabled: bool = True) -> bool:
        """
        Active ou désactive un effet vocal spécifique.

        Args:
            effect_name: Nom de l'effet
            enabled: True pour activer, False pour désactiver

        Returns:
            bool: True si le changement a réussi, False si l'effet est inconnu
        """
        if effect_name not in VOCAL_EFFECTS:
            self.logger.warning(f"Effet vocal inconnu: {effect_name}")
            return False

        self.enabled_effects[effect_name] = enabled
        self.logger.debug(f"Effet '{effect_name}' {'activé' if enabled else 'désactivé'}")
        return True

    def get_all_effects(self) -> dict[str, dict[str, Any]]:
        """
        Obtient la liste de tous les effets vocaux disponibles avec leur statut.

        Returns:
            dict: Dictionnaire des effets vocaux disponibles avec leur statut
        """
        result = {}
        for effect_name, symbol in VOCAL_EFFECTS.items():
            result[effect_name] = {
                "symbol": symbol,
                "enabled": self.enabled_effects.get(effect_name, True),
                "variations": EFFECT_VARIATIONS.get(effect_name, [symbol]),
                "usage_count": self.effect_usage_count.get(effect_name, 0),
            }
        return result

    def add_effect_to_text(self, text: str, effect_name: str) -> str:
        """
        Ajoute un effet vocal spécifique au texte.

        Args:
            text: Texte original
            effect_name: Nom de l'effet à ajouter

        Returns:
            str: Texte avec effet ajouté
        """
        if effect_name not in VOCAL_EFFECTS or not self.enabled_effects.get(effect_name, True):
            return text

        # Obtenir les variations possibles de l'effet
        variations = EFFECT_VARIATIONS.get(effect_name, [VOCAL_EFFECTS[effect_name]])

        # Sélectionner une variation aléatoire
        effect_symbol = random.choice(variations)

        # Mettre à jour le compteur d'utilisation
        self.effect_usage_count[effect_name] = self.effect_usage_count.get(effect_name, 0) + 1

        # Ajouter l'effet soit au début, au milieu ou à la fin du texte
        position = random.random()
        if position < 0.2:  # Début (20% des cas)
            return f"{effect_symbol} {text}"
        elif position < 0.7:  # Milieu (50% des cas)
            # Diviser le texte en phrases et insérer l'effet entre deux phrases
            sentences = re.split(r"([.!?])", text)
            if len(sentences) <= 2:  # Pas assez de phrases
                mid_point = len(text) // 2
                return f"{text[:mid_point]} {effect_symbol} {text[mid_point:]}"
            else:
                # Trouver un point d'insertion après une ponctuation
                insertion_point = random.randint(1, len(sentences) // 2) * 2
                return (
                    "".join(sentences[:insertion_point]) + f" {effect_symbol} " + "".join(sentences[insertion_point:])
                )
        else:  # Fin (30% des cas)
            return f"{text} {effect_symbol}"

    def enhance_text_with_effects(
        self,
        text: str,
        selected_effects: list[str] | None = None,
        emotional_state: dict[str, Any] | None = None,
    ) -> str:
        """
        Améliore un texte avec des effets vocaux adaptés au contenu et à l'état émotionnel.

        Args:
            text: Texte à améliorer
            selected_effects: Liste d'effets spécifiques à appliquer (optionnel)
            emotional_state: État émotionnel pour adapter les effets (optionnel)

        Returns:
            str: Texte enrichi d'effets vocaux
        """
        if not text:
            return text

        # Si l'intensité est nulle, ne rien faire
        if self.effect_intensity <= 0.0:
            return text

        # Appliquer les effets explicitement sélectionnés
        enhanced_text = text
        if selected_effects:
            for effect in selected_effects:
                if effect in VOCAL_EFFECTS and self.enabled_effects.get(effect, True):
                    # Probabilité d'application basée sur l'intensité globale
                    if random.random() < self.effect_intensity:
                        enhanced_text = self.add_effect_to_text(enhanced_text, effect)

        # Appliquer les effets automatiques basés sur le contenu
        else:
            # Effets basés sur la ponctuation et les motifs
            for pattern, effect in AUTO_EFFECT_PATTERNS:
                if self.enabled_effects.get(effect, True) and random.random() < self.effect_intensity:
                    matches = list(re.finditer(pattern, enhanced_text))
                    if matches:
                        # Limiter le nombre de remplacements pour éviter la surcharge
                        max_replacements = max(1, int(len(matches) * self.effect_intensity))
                        for match in random.sample(matches, min(max_replacements, len(matches))):
                            start, end = match.span()
                            effect_symbol = random.choice(EFFECT_VARIATIONS.get(effect, [VOCAL_EFFECTS[effect]]))
                            enhanced_text = (
                                enhanced_text[:start]
                                + enhanced_text[start:end]
                                + f" {effect_symbol} "
                                + enhanced_text[end:]
                            )

            # Effets aléatoires basés sur l'état émotionnel (si fourni)
            if emotional_state:
                emotion = emotional_state.get("dominant", "neutre")
                intensity = emotional_state.get("intensity", 0.5)

                # Sélectionner des effets appropriés selon l'émotion
                emotion_effects = self._get_emotion_appropriate_effects(emotion)

                # Appliquer les effets avec une probabilité basée sur l'intensité
                effect_probability = self.effect_intensity * intensity * 0.7  # Facteur pour éviter trop d'effets

                # Limiter le nombre total d'effets émotionnels
                max_emotional_effects = 1 if len(enhanced_text) < 100 else 2
                added_effects = 0

                for effect in emotion_effects:
                    if added_effects >= max_emotional_effects:
                        break

                    if random.random() < effect_probability and self.enabled_effects.get(effect, True):
                        enhanced_text = self.add_effect_to_text(enhanced_text, effect)
                        added_effects += 1

        return enhanced_text

    def _get_emotion_appropriate_effects(self, emotion: str) -> list[str]:
        """
        Retourne les effets vocaux appropriés pour une émotion donnée.

        Args:
            emotion: L'émotion pour laquelle trouver des effets appropriés

        Returns:
            list: Liste des effets vocaux appropriés
        """
        emotion_effects_map = {
            "joie": ["laugh", "sparkle", "excited"],
            "tristesse": ["sad_sigh", "pause", "soft"],
            "colère": ["growl", "pause"],
            "peur": ["tremble", "whisper", "nervous"],
            "surprise": ["surprise", "pause"],
            "neutre": ["soft", "thinking"],
            "calme": ["soft_breath", "soft"],
            "curiosité": ["thinking", "hesitation"],
            "confusion": ["hesitation", "thinking", "pause"],
            "émerveillement": ["surprised", "sparkle"],
            "gratitude": ["soft_breath", "happy_sigh"],
            "mélancolie": ["sad_sigh", "pause", "soft"],
        }

        return emotion_effects_map.get(emotion, ["soft", "pause"])

    def clean_effects_from_text(self, text: str) -> str:
        """
        Nettoie un texte de tous les marqueurs d'effets vocaux.

        Args:
            text: Texte à nettoyer

        Returns:
            str: Texte nettoyé
        """
        cleaned_text = text

        # Supprimer tous les marqueurs d'effets vocaux connus
        for effect_symbol in VOCAL_EFFECTS.values():
            cleaned_text = cleaned_text.replace(effect_symbol, "")

        # Supprimer toutes les variations connues
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # Considérer l'utilisation d'itertools.product ou de compréhensions
        for variations in EFFECT_VARIATIONS.values():
            for variation in variations:
                cleaned_text = cleaned_text.replace(variation, "")

        # Supprimer les effets entre astérisques
        cleaned_text = re.sub(r"\*[^*]+\*", "", cleaned_text)

        # Nettoyer les espaces multiples
        cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

        return cleaned_text

    def apply_whisper_effect(self, text: str, intensity: float = 0.7) -> str:
        """
        Applique un effet de chuchotement au texte.

        Args:
            text: Texte à transformer en chuchotement
            intensity: Intensité de l'effet de chuchotement (0.0-1.0)

        Returns:
            str: Texte avec l'effet de chuchotement appliqué
        """
        if not text:
            return text

        # Intensité limitée entre 0 et 1
        intensity = max(0.0, min(1.0, intensity))

        # Marquer le texte avec l'effet de chuchotement
        whisper_variation = random.choice(EFFECT_VARIATIONS.get("whisper", ["*chuchote*"]))

        # Appliquer différemment selon l'intensité
        if intensity > 0.8:
            # Chuchotement intense - marquer au début et à la fin
            return f"{whisper_variation} {text} {whisper_variation}"
        elif intensity > 0.5:
            # Chuchotement modéré - marquer au début
            return f"{whisper_variation} {text}"
        else:
            # Chuchotement léger - ajouter des effets textuels subtils
            # Ajouter des pauses douces pour simuler un chuchotement
            sentences = re.split(r"([.!?])", text)
            if len(sentences) > 2:
                # Ajouter des pauses légères entre certaines phrases
                for i in range(2, len(sentences), 4):
                    if i < len(sentences):
                        sentences[i] = sentences[i] + " *pause douce* "
                enhanced_text = "".join(sentences)
            else:
                enhanced_text = text

            # Ajouter l'indicateur de chuchotement
            return f"{whisper_variation} {enhanced_text}"


def apply_whisper_effect(audio_segment):
    """
    Applique un effet de chuchotement à un segment audio.

    Cette fonction transforme un segment audio pour lui donner un effet de chuchotement
    en modifiant ses paramètres audio (volume, fréquence, etc.).

    Args:
        audio_segment: Le segment audio à transformer (objet AudioSegment ou similaire)

    Returns:
        Le segment audio modifié avec l'effet de chuchotement
    """
    # TODO: Implémenter l'effet de chuchotement
    # Cela pourrait inclure:
    # - Réduction du volume global
    # - Accentuation des hautes fréquences
    # - Filtrage pour enlever les basses fréquences
    # - Ajout de souffle léger

    # Pour l'instant, retourne simplement l'audio non modifié
    return audio_segment
