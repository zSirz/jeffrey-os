#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import json
import os
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


class EmpathicTriggerAnalyzer:
    """
    Détecte les déclencheurs empathiques dans les phrases de l'utilisateur.

    Cette classe est responsable de :
    - Analyser les phrases pour détecter des indices de détresse émotionnelle
    - Identifier les besoins d'empathie dans les interactions
    - Catégoriser les types de déclencheurs empathiques
    - Suggérer des réponses adaptées selon le contexte émotionnel

    Attributs:
        enabled (bool): Indique si le module est activé
        triggers_file (str): Chemin vers le fichier de configuration des déclencheurs
        triggers (Dict): Déclencheurs empathiques chargés
        trigger_history (List): Historique des déclencheurs détectés
    """

    def __init__(
        self,
        enabled: bool = True,
        triggers_file: str = "data/behavior/empathic_triggers.json",
        max_history: int = 50,
    ):
        """
        Initialise l'analyseur de déclencheurs empathiques.

        Args:
            enabled (bool): Active ou désactive le module
            triggers_file (str): Chemin vers le fichier de configuration
            max_history (int): Nombre maximum d'entrées dans l'historique
        """
        self.enabled = enabled
        self.triggers_file = triggers_file
        self.max_history = max_history
        self.triggers = self._load_triggers()
        self.trigger_history = []
        self.last_trigger_time = {}  # Pour suivre la dernière occurrence de chaque type
        logger.info(f"EmpathicTriggerAnalyzer initialized: enabled={enabled}")

    def _load_triggers(self) -> Dict:
        """
        Charge les déclencheurs empathiques depuis le fichier de configuration.

        Returns:
            Dict: Dictionnaire des déclencheurs empathiques
        """
        if not self.enabled:
        return {}

        try:
        if os.path.exists(self.triggers_file):
                                            with open(self.triggers_file, "r", encoding="utf-8") as f:
        return json.load(f)
                                                else:
                # Créer le dossier parent si nécessaire
                os.makedirs(os.path.dirname(self.triggers_file), exist_ok=True)

                # Déclencheurs par défaut
                default_triggers = {
                    "version": "1.0.0",
                    "categories": {
                        "solitude": {
                            "description": "Expressions liées à la solitude et l'isolement",
                            "patterns": [
                                r"(?i)je\s+(?:me\s+sens|suis)\s+seul",
                                r"(?i)personne\s+(?:ne\s+)?(?:m['']aime|ne\s+m['']aime)",
                                r"(?i)je\s+n['']ai\s+personne",
                                r"(?i)(?:il\s+n['']y\s+a\s+)?personne\s+(?:pour|à\s+qui)\s+parler",
                                r"(?i)je\s+(?:me\s+sens|suis)\s+isolé",
                            ],
                            "keywords": ["seul", "isolé", "abandonné", "solitude"],
                            "response_suggestions": [
                                "Je comprends que tu puisses te sentir seul(e). C'est une émotion que beaucoup de personnes ressentent. Veux-tu en parler davantage ?",
                                "La solitude peut être difficile à vivre. Je suis là pour échanger avec toi.",
                            ],
                            "intensity_levels": {
                                "low": ["parfois seul", "un peu seul"],
                                "medium": ["très seul", "vraiment seul"],
                                "high": [
                                    "totalement seul",
                                    "complètement abandonné",
                                    "personne ne m'aime",
                                ],
                            },
                        },
                        "tristesse": {
                            "description": "Expressions de tristesse ou de mélancolie",
                            "patterns": [
                                r"(?i)je\s+(?:me\s+sens|suis)\s+(?:triste|déprimé)",
                                r"(?i)j['']ai\s+(?:envie\s+de|le\s+goût\s+de)\s+pleurer",
                                r"(?i)je\s+n['']ai\s+plus\s+goût\s+à\s+rien",
                                r"(?i)je\s+me\s+sens\s+mal",
                                r"(?i)je\s+souffre",
                            ],
                            "keywords": [
                                "triste",
                                "déprimé",
                                "mélancolique",
                                "dépression",
                                "pleurer",
                            ],
                            "response_suggestions": [
                                "Je perçois de la tristesse dans ton message. Souhaites-tu m'en dire plus sur ce que tu ressens ?",
                                "Il est normal de se sentir triste parfois. Je suis là pour t'écouter si tu souhaites partager.",
                            ],
                            "intensity_levels": {
                                "low": ["un peu triste", "légèrement déprimé"],
                                "medium": ["vraiment triste", "assez déprimé"],
                                "high": ["profondément triste", "très déprimé", "désespéré"],
                            },
                        },
                        "anxiete": {
                            "description": "Expressions d'anxiété, stress ou inquiétude",
                            "patterns": [
                                r"(?i)je\s+(?:me\s+sens|suis)\s+(?:anxieux|stressé|inquiet)",
                                r"(?i)j['']ai\s+peur\s+(?:de|que)",
                                r"(?i)je\s+(?:stresse|m['']inquiète|angoisse)",
                                r"(?i)je\s+ne\s+sais\s+pas\s+quoi\s+faire",
                                r"(?i)(?:tout|ça)\s+m['']angoisse",
                            ],
                            "keywords": ["anxieux", "stressé", "peur", "inquiet", "angoisse"],
                            "response_suggestions": [
                                "Je comprends que tu ressentes de l'anxiété. Respirons ensemble un moment. Peux-tu me décrire plus précisément ce qui t'inquiète ?",
                                "L'anxiété peut être intense. Prends ton temps, je suis là pour t'écouter.",
                            ],
                            "intensity_levels": {
                                "low": ["un peu anxieux", "légèrement inquiet"],
                                "medium": ["très anxieux", "vraiment stressé"],
                                "high": ["extrêmement anxieux", "panique", "terrorisé"],
                            },
                        },
                        "colere": {
                            "description": "Expressions de colère ou frustration",
                            "patterns": [
                                r"(?i)je\s+(?:me\s+sens|suis)\s+(?:énervé|en\s+colère|furieux)",
                                r"(?i)(?:ça|cela)\s+m['']énerve",
                                r"(?i)je\s+(?:déteste|hais)",
                                r"(?i)j['']en\s+ai\s+marre",
                                r"(?i)je\s+n['']en\s+peux\s+plus",
                            ],
                            "keywords": ["colère", "énervé", "furieux", "frustré", "rage"],
                            "response_suggestions": [
                                "Je sens de la colère dans ton message. C'est une émotion naturelle. Veux-tu m'expliquer ce qui a provoqué cette réaction ?",
                                "La colère nous aide parfois à identifier ce qui nous dérange. Je suis là pour t'écouter.",
                            ],
                            "intensity_levels": {
                                "low": ["agacé", "contrarié"],
                                "medium": ["en colère", "énervé"],
                                "high": ["furieux", "enragé", "hors de moi"],
                            },
                        },
                        "desespoir": {
                            "description": "Expressions de désespoir ou pensées sombres",
                            "patterns": [
                                r"(?i)je\s+(?:n['']ai|ne\s+vois)\s+(?:plus|pas)\s+d['']espoir",
                                r"(?i)(?:je\s+veux|j['']ai\s+envie\s+de)\s+(?:mourir|disparaître)",
                                r"(?i)je\s+ne\s+vois\s+pas\s+(?:l['']intérêt|pourquoi\s+continuer)",
                                r"(?i)tout\s+est\s+(?:fini|terminé)",
                                r"(?i)rien\s+ne\s+va\s+jamais\s+s['']arranger",
                            ],
                            "keywords": [
                                "désespoir",
                                "sans espoir",
                                "mourir",
                                "abandonner",
                                "fini",
                            ],
                            "response_suggestions": [
                                "Je perçois une grande détresse dans ton message et je m'en inquiète. Sache que tu n'es pas seul(e) et qu'il existe des ressources pour t'aider.",
                                "Ce que tu traverses semble très difficile. Pourrais-tu envisager de parler à un professionnel qui pourrait t'accompagner ?",
                            ],
                            "intensity_levels": {
                                "low": ["découragé", "sans motivation"],
                                "medium": ["désespéré", "à bout"],
                                "high": ["suicidaire", "veux mourir", "en finir"],
                            },
                        },
                        "besoin_aide": {
                            "description": "Expressions explicites de besoin d'aide",
                            "patterns": [
                                r"(?i)(?:j['']ai\s+besoin|aide[rz]?[- ]moi)",
                                r"(?i)je\s+ne\s+sais\s+pas\s+quoi\s+faire",
                                r"(?i)(?:qu['']est[- ]ce\s+que|comment)\s+(?:je\s+dois|puis[- ]je)\s+faire",
                                r"(?i)je\s+suis\s+perdu",
                                r"(?i)je\s+ne\s+m['']en\s+sors\s+pas",
                            ],
                            "keywords": ["aide", "besoin", "conseil", "perdu", "guidance"],
                            "response_suggestions": [
                                "Je vois que tu cherches de l'aide. Je suis là pour t'accompagner. Comment puis-je t'aider précisément ?",
                                "C'est courageux de demander de l'aide. Explique-moi ta situation, et nous verrons ensemble les options possibles.",
                            ],
                            "intensity_levels": {
                                "low": ["besoin d'un conseil", "petite aide"],
                                "medium": ["besoin d'aide", "je suis perdu"],
                                "high": ["désespérément besoin d'aide", "urgence", "critique"],
                            },
                        },
                    },
                    "response_modifiers": {
                        "repeat_detection": {
                            "description": "Modifier la réponse si le même déclencheur est détecté plusieurs fois",
                            "thresholds": {
                                "same_category": {
                                    "timeframe_seconds": 600,
                                    "count": 2,
                                    "action": "escalate_empathy",
                                },
                                "all_categories": {
                                    "timeframe_seconds": 1200,
                                    "count": 4,
                                    "action": "suggest_resources",
                                },
                            },
                        },
                        "intensity_response": {
                            "description": "Adapter la réponse selon l'intensité détectée",
                            "actions": {
                                "low": "acknowledge",
                                "medium": "express_empathy",
                                "high": "offer_support_resources",
                            },
                        },
                    },
                    "external_resources": {
                        "mental_health": [
                            {
                                "name": "SOS Amitié",
                                "description": "Service d'écoute par téléphone",
                                "contact": "09 72 39 40 50",
                            },
                            {
                                "name": "Fil Santé Jeunes",
                                "description": "Service d'écoute pour les 12-25 ans",
                                "contact": "0 800 235 236",
                            },
                        ]
                    },
                }

                # Sauvegarder les déclencheurs par défaut
                                                    with open(self.triggers_file, "w", encoding="utf-8") as f:
                    json.dump(default_triggers, f, indent=2)
        return default_triggers
        except Exception as e:
            logger.error(f"Failed to load empathic triggers: {str(e)}")
            # Retourner une structure minimale en cas d'échec
        return {"categories": {}}

    def _record_trigger_history(self, text: str, triggers_detected: List[Dict]) -> None:
        """
        Enregistre une entrée dans l'historique des déclencheurs.

        Args:
            text (str): Texte analysé
            triggers_detected (List[Dict]): Déclencheurs détectés
        """
        if not self.enabled or not triggers_detected:
                                                                        return

        timestamp = datetime.now()

        # Mise à jour du dernier déclenchement par catégorie
        for trigger in triggers_detected:
            category = trigger.get("category", "unknown")
            self.last_trigger_time[category] = timestamp

        entry = {
            "timestamp": timestamp.isoformat(),
            "text_excerpt": text[:100] + "..." if len(text) > 100 else text,
            "triggers": triggers_detected,
        }

        self.trigger_history.append(entry)

        # Limiter la taille de l'historique
        if len(self.trigger_history) > self.max_history:
            self.trigger_history = self.trigger_history[-self.max_history :]

    def analyze(self, text: str, context: Dict = None) -> Tuple[bool, List[Dict], Optional[Dict]]:
        """
        Analyse un texte pour détecter des déclencheurs empathiques.

        Args:
            text (str): Texte à analyser
            context (Dict, optional): Contexte supplémentaire pour l'analyse

        Returns:
            Tuple[bool, List[Dict], Optional[Dict]]:
                - Premier élément: True si des déclencheurs sont détectés, False sinon
                - Deuxième élément: Liste des déclencheurs détectés
                - Troisième élément: Suggestions de réponse ou None
        """
        if not self.enabled or not text:
        return False, [], None

        triggers_detected = []
        context = context or {}
        now = datetime.now()

        # Récupérer les catégories de déclencheurs
        categories = self.triggers.get("categories", {})

        # Détecter les déclencheurs dans chaque catégorie
        for category_name, category_data in categories.items():
            # Récupérer les patterns de cette catégorie
            patterns = category_data.get("patterns", [])
            keywords = category_data.get("keywords", [])

            # Vérifier chaque pattern
        for pattern in patterns:
        try:
        if re.search(pattern, text):
                        # Déterminer l'intensité
                        intensity = self._determine_intensity(category_name, text)

                        trigger = {
                            "category": category_name,
                            "description": category_data.get("description", ""),
                            "matched_pattern": pattern,
                            "intensity": intensity,
                        }
                        triggers_detected.append(trigger)
                                                                                                                break  # Un seul match par catégorie suffit
        except Exception as e:
                    logger.error(f"Error in pattern matching: {str(e)} for pattern {pattern}")

            # Si aucun pattern n'a été trouvé, vérifier les mots-clés
        if not any(
                t.get("category") == category_name
        for t in triggers_detected  # TODO: Optimiser cette boucle imbriquée
            ):
        for keyword in keywords:
        if keyword.lower() in text.lower():
                        # Déterminer l'intensité
                        intensity = self._determine_intensity(category_name, text)

                        trigger = {
                            "category": category_name,
                            "description": category_data.get("description", ""),
                            "matched_keyword": keyword,
                            "intensity": intensity,
                        }
                        triggers_detected.append(trigger)
                                                                                                                                        break  # Un seul match par catégorie suffit

        # Vérification des répétitions pour modifier la réponse
        response_modifiers = self.triggers.get("response_modifiers", {})
        repeat_detection = response_modifiers.get("repeat_detection", {})

        # Vérifier les répétitions par catégorie
        for trigger in triggers_detected:
            category = trigger.get("category")
        if category in self.last_trigger_time:
                last_time = self.last_trigger_time[category]
                # Convertir last_time de string à datetime si nécessaire
        if isinstance(last_time, str):
                    last_time = datetime.fromisoformat(last_time)

                # Vérifier le seuil de répétition pour la même catégorie
                same_category_threshold = repeat_detection.get("thresholds", {}).get(
                    "same_category", {}
                )
                timeframe_seconds = same_category_threshold.get("timeframe_seconds", 600)

        if (now - last_time).total_seconds() < timeframe_seconds:
                    # Répétition détectée
                    trigger["is_repeated"] = True
                    trigger["action"] = same_category_threshold.get("action", "escalate_empathy")

        # Enregistrer l'historique
        self._record_trigger_history(text, triggers_detected)

        # Générer des suggestions de réponse si des déclencheurs sont détectés
        if triggers_detected:
            response_suggestions = self._generate_response_suggestions(triggers_detected)
        return True, triggers_detected, response_suggestions

        return False, [], None

    def _determine_intensity(self, category: str, text: str) -> str:
        """
        Détermine l'intensité d'un déclencheur empathique.

        Args:
            category (str): Catégorie du déclencheur
            text (str): Texte complet analysé

        Returns:
            str: Niveau d'intensité ("low", "medium", "high")
        """
        categories = self.triggers.get("categories", {})
        category_data = categories.get(category, {})
        intensity_levels = category_data.get("intensity_levels", {})

        # Vérifier les patterns d'intensité élevée
        for high_pattern in intensity_levels.get("high", []):
        if high_pattern.lower() in text.lower():
        return "high"

        # Vérifier les patterns d'intensité moyenne
        for medium_pattern in intensity_levels.get("medium", []):
        if medium_pattern.lower() in text.lower():
        return "medium"

        # Par défaut, intensité faible
        return "low"

    def _generate_response_suggestions(self, triggers: List[Dict]) -> Dict:
        """
        Génère des suggestions de réponse basées sur les déclencheurs détectés.

        Args:
            triggers (List[Dict]): Liste des déclencheurs détectés

        Returns:
            Dict: Suggestions de réponse
        """
        categories = self.triggers.get("categories", {})

        # Trier les déclencheurs par intensité
        sorted_triggers = sorted(
            triggers,
            key=lambda x: {"low": 1, "medium": 2, "high": 3}.get(x.get("intensity", "low"), 0),
            reverse=True,
        )

        # Le déclencheur le plus intense détermine la réponse principale
        primary_trigger = sorted_triggers[0] if sorted_triggers else {}
        primary_category = primary_trigger.get("category", "")
        primary_intensity = primary_trigger.get("intensity", "low")

        # Récupérer les suggestions de réponse pour cette catégorie
        category_data = categories.get(primary_category, {})
        response_suggestions = category_data.get("response_suggestions", [])

        # Vérifier s'il y a des répétitions qui nécessitent une escalade
        repeated_triggers = [t for t in triggers if t.get("is_repeated", False)]

        if repeated_triggers:
                                                                                                                                                                                                        pass

            # Si l'intensité est élevée et qu'il y a répétition, proposer des ressources
        if primary_intensity == "high":
                resources = self.triggers.get("external_resources", {}).get("mental_health", [])
        return {
                    "primary_category": primary_category,
                    "intensity": primary_intensity,
                    "type": "resource_referral",
                    "suggested_responses": response_suggestions[:1] if response_suggestions else [],
                    "resources": resources,
                }

        # Adapter la réponse selon l'intensité
        intensity_actions = (
            self.triggers.get("response_modifiers", {})
            .get("intensity_response", {})
            .get("actions", {})
        )
        action = intensity_actions.get(primary_intensity, "acknowledge")

        return {
            "primary_category": primary_category,
            "intensity": primary_intensity,
            "type": action,
            "is_repeated": bool(repeated_triggers),
            "suggested_responses": response_suggestions[:2] if response_suggestions else [],
        }

    def get_trigger_history(self) -> List[Dict]:
        """
        Récupère l'historique des déclencheurs détectés.

        Returns:
            List[Dict]: Liste des entrées de l'historique
        """
        if not self.enabled:
        return []

        return self.trigger_history

    def get_active_categories(self, timeframe_seconds: int = 3600) -> Set[str]:
        """
        Récupère les catégories de déclencheurs actives dans un intervalle de temps.

        Args:
            timeframe_seconds (int): Intervalle de temps en secondes

        Returns:
            Set[str]: Ensemble des catégories actives
        """
        if not self.enabled:
        return set()

        now = datetime.now()
        active_categories = set()

        for category, last_time in self.last_trigger_time.items():
            # Convertir last_time de string à datetime si nécessaire
        if isinstance(last_time, str):
                last_time = datetime.fromisoformat(last_time)

        if (now - last_time).total_seconds() < timeframe_seconds:
                active_categories.add(category)

        return active_categories
