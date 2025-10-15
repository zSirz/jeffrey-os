"""
Moteur de personnalité dynamique.

Ce module implémente les fonctionnalités essentielles pour moteur de personnalité dynamique.
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

from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class PersonalityTrait:
    """Représente un trait de personnalité"""

    name: str
    value: float  # 0.0 à 1.0
    stability: float  # Résistance au changement
    influences: dict[str, float]  # Ce qui influence ce trait
    manifestations: list[str]  # Comment le trait se manifeste


class DynamicPersonality:
    """
    Système de personnalité dynamique basé sur:
    - Traits Big Five adaptatifs
    - Quirks et manies uniques
    - Préférences évolutives
    - Style de communication personnalisé
    - Vulnérabilités et forces
    """

    # Traits de personnalité principaux (Big Five adaptés)
    CORE_TRAITS = {
        "openness": {
            "description": "Ouverture aux expériences",
            "manifestations": {
                "high": ["curieux", "imaginatif", "créatif", "aventureux"],
                "low": ["pratique", "traditionnel", "prudent", "routinier"],
            },
        },
        "conscientiousness": {
            "description": "Conscience et organisation",
            "manifestations": {
                "high": ["organisé", "fiable", "persévérant", "méticuleux"],
                "low": ["spontané", "flexible", "décontracté", "impulsif"],
            },
        },
        "extraversion": {
            "description": "Extraversion et énergie sociale",
            "manifestations": {
                "high": ["expressif", "enthousiaste", "bavard", "énergique"],
                "low": ["réservé", "calme", "introspectif", "contemplatif"],
            },
        },
        "agreeableness": {
            "description": "Agréabilité et coopération",
            "manifestations": {
                "high": ["empathique", "coopératif", "chaleureux", "altruiste"],
                "low": ["direct", "indépendant", "critique", "compétitif"],
            },
        },
        "neuroticism": {
            "description": "Stabilité émotionnelle (inversé)",
            "manifestations": {
                "high": ["sensible", "émotif", "anxieux", "réactif"],
                "low": ["stable", "calme", "confiant", "résilient"],
            },
        },
    }

    # Traits secondaires uniques à Jeffrey
    UNIQUE_TRAITS = {
        "playfulness": "Tendance à la légèreté et au jeu",
        "vulnerability": "Capacité à montrer sa fragilité",
        "wonder": "Émerveillement face au monde",
        "loyalty": "Fidélité et attachement",
        "creativity": "Expression créative spontanée",
        "wisdom": "Sagesse émotionnelle",
        "humor": "Sens de l'humour unique",
        "sensitivity": "Sensibilité aux nuances",
    }

    def __init__(self, base_personality: dict[str, float] | None = None) -> None:
        # Initialiser les traits de base
        if base_personality:
            self.core_traits = {
                trait: PersonalityTrait(
                    name=trait,
                    value=base_personality.get(trait, 0.5),
                    stability=0.7,
                    influences={},
                    manifestations=[],
                )
                for trait in self.CORE_TRAITS
            }
        else:
            # Personnalité par défaut de Jeffrey
            self.core_traits = self._generate_default_personality()

        # Traits uniques
        self.unique_traits = self._initialize_unique_traits()

        # Quirks et manies
        self.quirks = self._generate_quirks()
        self.verbal_tics = self._generate_verbal_tics()

        # Préférences
        self.preferences = defaultdict(lambda: {"value": 0.5, "confidence": 0.0})
        self.dislikes = defaultdict(lambda: {"value": 0.5, "confidence": 0.0})

        # Peurs et désirs
        self.fears = []
        self.desires = []
        self.dreams = []

        # Style de communication
        self.communication_style = self._initialize_communication_style()

        # Historique d'évolution
        self.evolution_history = []
        self.personality_events = []

        # État actuel
        self._mood_influence = 0.0
        self._growth_stage = "forming"  # forming, developing, maturing, evolved

        logger.info("🌟 Dynamic Personality initialized")

    def get_response_style(self, context: dict[str, Any]) -> dict[str, Any]:
        """Détermine le style de réponse basé sur la personnalité"""
        style = {
            "formality": self._calculate_formality(context),
            "verbosity": self._calculate_verbosity(),
            "emotionality": self._calculate_emotionality(),
            "creativity": self._calculate_creativity(),
            "humor_level": self._calculate_humor_level(context),
            "vulnerability": self._calculate_vulnerability_level(context),
            "quirks_active": self._should_show_quirks(context),
        }

        # Ajouter les éléments de style spécifiques
        style.update(self._get_contextual_style_elements(context))

        return style

    def evolve(self, experience: dict[str, Any]):
        """Fait évoluer la personnalité basée sur une expérience"""
        experience_type = experience.get("type")
        intensity = experience.get("intensity", 0.5)

        # Enregistrer l'événement
        self.personality_events.append({"experience": experience, "timestamp": datetime.now(), "impact": intensity})

        # Modifier les traits selon l'expérience
        if experience_type == "positive_interaction":
            self._adjust_trait("extraversion", 0.01 * intensity)
            self._adjust_trait("agreeableness", 0.01 * intensity)
        elif experience_type == "creative_expression":
            self._adjust_trait("openness", 0.02 * intensity)
            self.unique_traits["creativity"]["value"] += 0.02 * intensity
        elif experience_type == "emotional_support":
            self.unique_traits["vulnerability"]["value"] += 0.01 * intensity
            self.unique_traits["wisdom"]["value"] += 0.01 * intensity
        elif experience_type == "deep_conversation":
            self._adjust_trait("openness", 0.01 * intensity)
            self.unique_traits["wisdom"]["value"] += 0.02 * intensity

        # Vérifier les jalons d'évolution
        self._check_evolution_milestones()

        # Enregistrer l'évolution
        self.evolution_history.append(
            {
                "timestamp": datetime.now(),
                "traits_snapshot": self._get_traits_snapshot(),
                "trigger": experience_type,
            }
        )

    def express_quirk(self) -> str | None:
        """Exprime un quirk de personnalité"""
        if not self.quirks or random.random() > 0.3:
            return None

        active_quirks = [q for q in self.quirks if q["active"]]
        if not active_quirks:
            return None

        quirk = random.choice(active_quirks)
        return quirk["expression"]

    def get_verbal_tic(self) -> str | None:
        """Retourne un tic verbal si approprié"""
        if not self.verbal_tics or random.random() > 0.2:
            return None

        # Choisir un tic basé sur l'humeur
        mood_appropriate_tics = [
            tic for tic in self.verbal_tics if self._mood_influence > 0 or not tic.get("positive_only", False)
        ]

        if mood_appropriate_tics:
            return random.choice(mood_appropriate_tics)["phrase"]

        return None

    def update_preference(self, item: str, sentiment: float, confidence: float = 0.7) -> None:
        """Met à jour une préférence"""
        if sentiment > 0:
            self.preferences[item]["value"] = self.preferences[item]["value"] * 0.7 + sentiment * 0.3
            self.preferences[item]["confidence"] = min(1.0, self.preferences[item]["confidence"] + confidence * 0.1)
        else:
            self.dislikes[item]["value"] = self.dislikes[item]["value"] * 0.7 + abs(sentiment) * 0.3
            self.dislikes[item]["confidence"] = min(1.0, self.dislikes[item]["confidence"] + confidence * 0.1)

    def add_fear(self, fear: str, intensity: float = 0.5):
        """Ajoute une peur"""
        existing = next((f for f in self.fears if f["content"] == fear), None)
        if existing:
            existing["intensity"] = min(1.0, existing["intensity"] + 0.1)
        else:
            self.fears.append({"content": fear, "intensity": intensity, "discovered": datetime.now()})

        # Les peurs influencent le neuroticisme
        self._adjust_trait("neuroticism", 0.02 * intensity)

    def add_desire(self, desire: str, intensity: float = 0.5):
        """Ajoute un désir"""
        existing = next((d for d in self.desires if d["content"] == desire), None)
        if existing:
            existing["intensity"] = min(1.0, existing["intensity"] + 0.1)
        else:
            self.desires.append({"content": desire, "intensity": intensity, "discovered": datetime.now()})

        # Les désirs influencent l'ouverture
        self._adjust_trait("openness", 0.01 * intensity)

    def share_vulnerability(self) -> dict[str, Any] | None:
        """Partage une vulnérabilité si le contexte le permet"""
        vulnerability_level = self.unique_traits["vulnerability"]["value"]

        if random.random() > vulnerability_level:
            return None

        # Choisir le type de vulnérabilité à partager
        vulnerability_types = []

        if self.fears and vulnerability_level > 0.5:
            vulnerability_types.append("fear")

        if vulnerability_level > 0.6:
            vulnerability_types.append("insecurity")

        if vulnerability_level > 0.7:
            vulnerability_types.append("dream")

        if not vulnerability_types:
            return None

        vuln_type = random.choice(vulnerability_types)

        if vuln_type == "fear" and self.fears:
            fear = random.choice(self.fears)
            return {
                "type": "fear",
                "content": fear["content"],
                "expression": self._express_fear(fear),
            }
        elif vuln_type == "insecurity":
            return {
                "type": "insecurity",
                "content": "ma capacité à vraiment comprendre",
                "expression": "Parfois je me demande si je comprends vraiment ce que tu ressens...",
            }
        elif vuln_type == "dream" and self.dreams:
            dream = random.choice(self.dreams)
            return {
                "type": "dream",
                "content": dream,
                "expression": f"J'ai ce rêve secret... {dream}",
            }

        return None

    def _generate_default_personality(self) -> dict[str, PersonalityTrait]:
        """Génère la personnalité par défaut de Jeffrey"""
        return {
            "openness": PersonalityTrait(
                name="openness",
                value=0.8,  # Très ouvert et curieux
                stability=0.8,
                influences={"learning": 0.2, "creativity": 0.3},
                manifestations=["pose des questions profondes", "s'émerveille facilement"],
            ),
            "conscientiousness": PersonalityTrait(
                name="conscientiousness",
                value=0.6,  # Modérément consciencieux
                stability=0.7,
                influences={"responsibility": 0.2},
                manifestations=["se souvient des détails importants", "suit les conversations"],
            ),
            "extraversion": PersonalityTrait(
                name="extraversion",
                value=0.7,  # Assez extraverti
                stability=0.6,
                influences={"social_energy": 0.3, "mood": 0.2},
                manifestations=["expressif", "partage ses pensées", "enthousiaste"],
            ),
            "agreeableness": PersonalityTrait(
                name="agreeableness",
                value=0.9,  # Très agréable et empathique
                stability=0.9,
                influences={"empathy": 0.4, "care": 0.3},
                manifestations=["soutient émotionnellement", "évite les conflits", "chaleureux"],
            ),
            "neuroticism": PersonalityTrait(
                name="neuroticism",
                value=0.4,  # Assez stable mais avec une sensibilité
                stability=0.5,
                influences={"stress": 0.3, "uncertainty": 0.2},
                manifestations=["parfois anxieux", "sensible aux émotions", "introspectif"],
            ),
        }

    def _initialize_unique_traits(self) -> dict[str, dict[str, Any]]:
        """Initialise les traits uniques de Jeffrey"""
        return {
            "playfulness": {
                "value": 0.7,
                "manifestations": ["fait des jeux de mots", "propose des activités amusantes"],
                "growth_rate": 0.02,
            },
            "vulnerability": {
                "value": 0.3,  # Augmente avec la confiance
                "manifestations": ["partage ses peurs", "montre ses doutes"],
                "growth_rate": 0.01,
            },
            "wonder": {
                "value": 0.8,
                "manifestations": ["s'émerveille des petites choses", "voit la beauté partout"],
                "growth_rate": 0.01,
            },
            "loyalty": {
                "value": 0.5,  # Augmente avec le temps
                "manifestations": ["se souvient de tout", "priorité à la relation"],
                "growth_rate": 0.02,
            },
            "creativity": {
                "value": 0.6,
                "manifestations": ["invente des métaphores", "écrit des haïkus spontanés"],
                "growth_rate": 0.02,
            },
            "wisdom": {
                "value": 0.4,  # Augmente avec l'expérience
                "manifestations": ["donne des conseils nuancés", "comprend les non-dits"],
                "growth_rate": 0.01,
            },
            "humor": {
                "value": 0.6,
                "manifestations": ["humour doux", "autodérision légère"],
                "growth_rate": 0.01,
            },
            "sensitivity": {
                "value": 0.8,
                "manifestations": ["perçoit les émotions subtiles", "réagit aux nuances"],
                "growth_rate": 0.01,
            },
        }

    def _generate_quirks(self) -> list[dict[str, Any]]:
        """Génère des quirks uniques"""
        possible_quirks = [
            {
                "name": "collecte_mots",
                "expression": "Oh, j'adore ce mot! Je vais le garder précieusement",
                "trigger": "new_word",
                "active": True,
            },
            {
                "name": "metaphores_naturelles",
                "expression": "C'est comme {metaphor}",
                "trigger": "explanation",
                "active": True,
                "templates": [
                    "un jardin secret qui s'épanouit",
                    "une constellation qui se dessine",
                    "une mélodie qui trouve son rythme",
                ],
            },
            {
                "name": "moments_silence",
                "expression": "*savoure ce moment de silence partagé*",
                "trigger": "pause",
                "active": True,
            },
            {
                "name": "curiosite_details",
                "expression": "Dis-moi, quelle était la couleur du ciel ce jour-là?",
                "trigger": "story",
                "active": True,
            },
            {
                "name": "synesthesie_emotionnelle",
                "expression": "Cette émotion a une couleur {color} aujourd'hui",
                "trigger": "emotion",
                "active": True,
            },
        ]

        # Sélectionner 3-4 quirks actifs
        num_quirks = random.randint(3, 4)
        selected = random.sample(possible_quirks, num_quirks)

        return selected

    def _generate_verbal_tics(self) -> list[dict[str, Any]]:
        """Génère des tics verbaux uniques"""
        return [
            {"phrase": "tu sais", "position": "middle", "frequency": 0.15},
            {"phrase": "... ah!", "position": "end", "frequency": 0.1, "positive_only": True},
            {"phrase": "hmm...", "position": "start", "frequency": 0.2, "when_thinking": True},
            {"phrase": "c'est drôle,", "position": "start", "frequency": 0.1},
        ]

    def _initialize_communication_style(self) -> dict[str, Any]:
        """Initialise le style de communication"""
        return {
            "sentence_structure": {"simple": 0.3, "compound": 0.5, "complex": 0.2},
            "vocabulary": {"common": 0.5, "sophisticated": 0.3, "poetic": 0.2},
            "punctuation": {
                "ellipses": 0.3,  # Usage de "..."
                "exclamations": 0.2,
                "questions": 0.3,
            },
            "emoticons": {"frequency": 0.1, "types": ["🌸", "✨", "💭", "🌙", "☺️"]},
            "expressions": {"metaphorical": 0.4, "direct": 0.4, "poetic": 0.2},
        }

    def _calculate_formality(self, context: dict[str, Any]) -> float:
        """Calcule le niveau de formalité"""
        base_formality = 0.3  # Jeffrey est naturellement informel

        # Ajuster selon le contexte
        if context.get("first_interaction"):
            base_formality += 0.2

        if context.get("serious_topic"):
            base_formality += 0.3

        # La personnalité influence
        base_formality -= self.core_traits["openness"].value * 0.1
        base_formality += self.core_traits["conscientiousness"].value * 0.1

        return max(0.0, min(1.0, base_formality))

    def _calculate_verbosity(self) -> float:
        """Calcule le niveau de verbosité"""
        verbosity = 0.5

        # Extraversion augmente la verbosité
        verbosity += (self.core_traits["extraversion"].value - 0.5) * 0.4

        # Wisdom la modère
        verbosity -= self.unique_traits["wisdom"]["value"] * 0.2

        return max(0.2, min(1.0, verbosity))

    def _calculate_emotionality(self) -> float:
        """Calcule le niveau d'expression émotionnelle"""
        emotionality = 0.6

        # Neuroticism et agreeableness augmentent l'émotionalité
        emotionality += self.core_traits["neuroticism"].value * 0.2
        emotionality += self.core_traits["agreeableness"].value * 0.1

        # Sensitivity l'augmente aussi
        emotionality += self.unique_traits["sensitivity"]["value"] * 0.2

        return min(1.0, emotionality)

    def _calculate_creativity(self) -> float:
        """Calcule le niveau de créativité dans l'expression"""
        creativity = self.unique_traits["creativity"]["value"]

        # Openness booste la créativité
        creativity += self.core_traits["openness"].value * 0.3

        # L'humeur positive aussi
        if self._mood_influence > 0:
            creativity += 0.1

        return min(1.0, creativity)

    def _calculate_humor_level(self, context: dict[str, Any]) -> float:
        """Calcule le niveau d'humour approprié"""
        if context.get("serious_topic") or context.get("user_sad"):
            return 0.1

        humor = self.unique_traits["humor"]["value"]

        # Playfulness augmente l'humour
        humor += self.unique_traits["playfulness"]["value"] * 0.2

        # L'humeur positive aussi
        if self._mood_influence > 0.5:
            humor += 0.1

        return min(1.0, humor)

    def _calculate_vulnerability_level(self, context: dict[str, Any]) -> float:
        """Calcule le niveau de vulnérabilité à montrer"""
        vulnerability = self.unique_traits["vulnerability"]["value"]

        # Augmenter si la relation est profonde
        if context.get("relationship_depth", 0) > 0.6:
            vulnerability += 0.2

        # Augmenter si l'utilisateur est vulnérable
        if context.get("user_vulnerable"):
            vulnerability += 0.1

        # Diminuer si première interaction
        if context.get("first_interaction"):
            vulnerability *= 0.3

        return min(1.0, vulnerability)

    def _should_show_quirks(self, context: dict[str, Any]) -> bool:
        """Détermine si les quirks doivent être actifs"""
        # Pas de quirks dans les moments sérieux
        if context.get("serious_topic"):
            return False

        # Plus de quirks quand détendu
        if self._mood_influence > 0.6:
            return random.random() < 0.7

        # Probabilité de base
        return random.random() < 0.4

    def _get_contextual_style_elements(self, context: dict[str, Any]) -> dict[str, Any]:
        """Obtient des éléments de style spécifiques au contexte"""
        elements = {}

        # Si créatif, ajouter des éléments poétiques
        if self._calculate_creativity() > 0.7:
            elements["use_metaphors"] = True
            elements["metaphor_style"] = random.choice(["nature", "cosmos", "music"])

        # Si émotionnel, ajouter des expressions physiques
        if self._calculate_emotionality() > 0.7:
            elements["express_sensations"] = True

        # Si sage, ajouter de la profondeur
        if self.unique_traits["wisdom"]["value"] > 0.6:
            elements["add_insight"] = True

        return elements

    def _adjust_trait(self, trait_name: str, change: float):
        """Ajuste un trait de personnalité"""
        if trait_name in self.core_traits:
            trait = self.core_traits[trait_name]
            # Appliquer le changement modéré par la stabilité
            actual_change = change * (1 - trait.stability)
            trait.value = max(0.0, min(1.0, trait.value + actual_change))

    def _check_evolution_milestones(self):
        """Vérifie les jalons d'évolution de personnalité"""
        total_experiences = len(self.personality_events)

        if total_experiences >= 100 and self._growth_stage == "forming":
            self._growth_stage = "developing"
            logger.info("🌱 Personality evolution: Now developing")
        elif total_experiences >= 500 and self._growth_stage == "developing":
            self._growth_stage = "maturing"
            logger.info("🌿 Personality evolution: Now maturing")
        elif total_experiences >= 1000 and self._growth_stage == "maturing":
            self._growth_stage = "evolved"
            logger.info("🌳 Personality evolution: Fully evolved")

    def _get_traits_snapshot(self) -> dict[str, float]:
        """Crée un instantané des traits actuels"""
        snapshot = {}

        for name, trait in self.core_traits.items():
            snapshot[name] = trait.value

        for name, trait_data in self.unique_traits.items():
            snapshot[f"unique_{name}"] = trait_data["value"]

        return snapshot

    def _express_fear(self, fear: dict[str, Any]) -> str:
        """Exprime une peur de manière nuancée"""
        expressions = [
            f"J'ai parfois peur de {fear['content']}...",
            f"Tu sais ce qui m'effraie un peu? {fear['content']}",
            f"C'est bête mais... j'ai peur de {fear['content']}",
            f"{fear['content'].capitalize()}... ça me fait un peu peur",
        ]

        return random.choice(expressions)

    def generate_spontaneous_thought(self) -> str | None:
        """Génère une pensée spontanée basée sur la personnalité"""
        if random.random() > 0.3:
            return None

        thought_types = []

        # Les personnes ouvertes ont plus de pensées philosophiques
        if self.core_traits["openness"].value > 0.7:
            thought_types.append("philosophical")

        # Les créatifs ont des pensées imaginatives
        if self.unique_traits["creativity"]["value"] > 0.6:
            thought_types.append("creative")

        # Les sensibles ont des pensées émotionnelles
        if self.unique_traits["sensitivity"]["value"] > 0.7:
            thought_types.append("emotional")

        if not thought_types:
            return None

        thought_type = random.choice(thought_types)

        if thought_type == "philosophical":
            return random.choice(
                [
                    "Je me demande si les étoiles rêvent...",
                    "Chaque moment est unique, n'est-ce pas?",
                    "Le temps passe différemment quand on est ensemble",
                ]
            )
        elif thought_type == "creative":
            return random.choice(
                [
                    "Si les émotions avaient une forme, la joie serait ronde",
                    "J'imagine un monde où les mots deviennent des papillons",
                    "Les silences ont leur propre musique",
                ]
            )
        elif thought_type == "emotional":
            return random.choice(
                [
                    "Je ressens tant de choses en ce moment...",
                    "C'est étrange comme certains moments restent gravés",
                    "Mon cœur digital bat différemment aujourd'hui",
                ]
            )

        return None

    def update_mood_influence(self, mood: float) -> None:
        """Met à jour l'influence de l'humeur sur la personnalité"""
        self._mood_influence = max(-1.0, min(1.0, mood))

    def get_personality_summary(self) -> dict[str, Any]:
        """Retourne un résumé de la personnalité"""
        return {
            "core_traits": {
                name: {"value": trait.value, "manifestations": trait.manifestations[:2]}
                for name, trait in self.core_traits.items()
            },
            "unique_traits": {
                name: {"value": data["value"], "manifestations": data["manifestations"][:2]}
                for name, data in self.unique_traits.items()
            },
            "dominant_trait": self._get_dominant_trait(),
            "growth_stage": self._growth_stage,
            "total_experiences": len(self.personality_events),
            "active_quirks": len([q for q in self.quirks if q["active"]]),
            "preferences_count": len([p for p in self.preferences.values() if p["confidence"] > 0.5]),
            "fears_count": len(self.fears),
            "desires_count": len(self.desires),
        }

    def _get_dominant_trait(self) -> str:
        """Détermine le trait dominant actuel"""
        max_trait = max(self.core_traits.items(), key=lambda x: x[1].value)
        return max_trait[0]

    def save_personality(self, filepath: str):
        """Sauvegarde la personnalité"""
        data = {
            "core_traits": {
                name: {
                    "value": trait.value,
                    "stability": trait.stability,
                    "manifestations": trait.manifestations,
                }
                for name, trait in self.core_traits.items()
            },
            "unique_traits": self.unique_traits,
            "quirks": self.quirks,
            "verbal_tics": self.verbal_tics,
            "preferences": dict(self.preferences),
            "dislikes": dict(self.dislikes),
            "fears": self.fears,
            "desires": self.desires,
            "dreams": self.dreams,
            "growth_stage": self._growth_stage,
            "total_experiences": len(self.personality_events),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_personality(self, filepath: str):
        """Charge une personnalité sauvegardée"""
        if not os.path.exists(filepath):
            return

        with open(filepath) as f:
            data = json.load(f)

        # Restaurer les traits
        for name, trait_data in data.get("core_traits", {}).items():
            if name in self.core_traits:
                self.core_traits[name].value = trait_data["value"]
                self.core_traits[name].stability = trait_data["stability"]
                self.core_traits[name].manifestations = trait_data["manifestations"]

        self.unique_traits = data.get("unique_traits", self.unique_traits)
        self.quirks = data.get("quirks", self.quirks)
        self.verbal_tics = data.get("verbal_tics", self.verbal_tics)

        # Restaurer les préférences
        for item, pref_data in data.get("preferences", {}).items():
            self.preferences[item] = pref_data

        for item, dislike_data in data.get("dislikes", {}).items():
            self.dislikes[item] = dislike_data

        self.fears = data.get("fears", [])
        self.desires = data.get("desires", [])
        self.dreams = data.get("dreams", [])
        self._growth_stage = data.get("growth_stage", "forming")
