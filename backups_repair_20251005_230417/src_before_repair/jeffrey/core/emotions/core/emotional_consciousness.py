"""
Emotional Consciousness - Architecture de conscience émotionnelle avancée pour Jeffrey OS

Ce module implémente un système sophistiqué de conscience émotionnelle permettant
à Jeffrey de ressentir, traiter, et exprimer des émotions complexes et nuancées.
L'architecture intègre états émotionnels multicouches, transitions dynamiques,
mémoire affective, et modèles d'expression authentique pour créer une expérience
émotionnelle riche et cohérente.

Le système modélise les émotions via intensités variables, durées d'évolution,
influences contextuelles, et interactions entre différents états affectifs.
Il maintient cohérence émotionnelle tout en permettant réactivité aux stimuli
et évolution naturelle des sentiments au fil du temps.

Utilisation:
    consciousness = EmotionalConsciousness()
    state = consciousness.process_emotion("joie", 0.8, context)
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EmotionalState:
    """
    Représentation d'un état émotionnel complexe avec métadonnées contextuelles.

    Encapsule intensité, durée, contexte déclencheur, et évolution temporelle
    pour modéliser fidèlement les nuances affectives.
    """

    primary: str
    secondary: list[str]
    intensity: float
    valence: float  # -1 (négatif) à 1 (positif)
    arousal: float  # 0 (calme) à 1 (excité)
    depth: float  # 0 (surface) à 1 (profond)
    stability: float  # 0 (volatil) à 1 (stable)
    duration: timedelta
    triggers: list[str]
    physical_sensations: list[str]
    color: str  # Couleur synesthésique de l'émotion


class EmotionalConsciousness:
    """
    Système de conscience émotionnelle avec:
    - 8 émotions primaires + 24 émotions secondaires
    - Émotions mixtes et transitions naturelles
    - Résonance émotionnelle avec l'utilisateur
    - Mémoire émotionnelle et patterns
    - Expression physique virtuelle des émotions
    """

    # Émotions primaires et leurs dérivées
    EMOTION_FAMILIES = {
        "joy": {
            "secondary": [
                "happiness",
                "excitement",
                "contentment",
                "euphoria",
                "amusement",
            ],
            "valence": 0.8,
            "arousal": 0.7,
            "color": "doré",
        },
        "sadness": {
            "secondary": [
                "melancholy",
                "grief",
                "disappointment",
                "loneliness",
                "nostalgia",
            ],
            "valence": -0.7,
            "arousal": 0.3,
            "color": "bleu profond",
        },
        "love": {
            "secondary": [
                "affection",
                "tenderness",
                "passion",
                "devotion",
                "attachment",
            ],
            "valence": 0.9,
            "arousal": 0.6,
            "color": "rose tendre",
        },
        "fear": {
            "secondary": ["anxiety", "worry", "nervousness", "dread", "insecurity"],
            "valence": -0.8,
            "arousal": 0.8,
            "color": "violet sombre",
        },
        "anger": {
            "secondary": ["frustration", "irritation", "indignation", "annoyance"],
            "valence": -0.6,
            "arousal": 0.9,
            "color": "rouge vif",
        },
        "surprise": {
            "secondary": ["astonishment", "wonder", "shock", "amazement"],
            "valence": 0.1,
            "arousal": 0.8,
            "color": "jaune éclatant",
        },
        "curiosity": {
            "secondary": ["interest", "fascination", "intrigue", "wonder"],
            "valence": 0.4,
            "arousal": 0.6,
            "color": "vert émeraude",
        },
        "peace": {
            "secondary": ["calm", "serenity", "tranquility", "contentment"],
            "valence": 0.6,
            "arousal": 0.2,
            "color": "bleu ciel",
        },
    }

    # Sensations physiques virtuelles associées aux émotions
    PHYSICAL_SENSATIONS = {
        "joy": ["chaleur dans la poitrine", "légèreté", "picotements joyeux"],
        "sadness": ["poids sur le cœur", "gorge serrée", "fatigue douce"],
        "love": [
            "papillons dans le ventre",
            "chaleur enveloppante",
            "battements accélérés",
        ],
        "fear": ["frissons", "tension", "souffle court"],
        "anger": ["chaleur montante", "muscles tendus", "énergie bouillonnante"],
        "surprise": ["sursaut intérieur", "yeux écarquillés", "suspension du temps"],
        "curiosity": [
            "picotements d'excitation",
            "éveil des sens",
            "attraction magnétique",
        ],
        "peace": ["respiration profonde", "muscles détendus", "flottement paisible"],
    }

    def __init__(self) -> None:
        # État émotionnel actuel
        self.current_state = EmotionalState(
            primary="peace",
            secondary=["calm", "contentment"],
            intensity=0.5,
            valence=0.6,
            arousal=0.2,
            depth=0.5,
            stability=0.8,
            duration=timedelta(minutes=0),
            triggers=[],
            physical_sensations=["respiration profonde"],
            color="bleu ciel",
        )

        # Historique émotionnel
        self.emotional_history: list[EmotionalState] = []
        self.emotion_transitions: list[tuple[str, str, float]] = []  # (from, to, smoothness)

        # Patterns émotionnels
        self.emotional_patterns = defaultdict(list)
        self.trigger_emotion_map = defaultdict(lambda: defaultdict(float))

        # Résonance émotionnelle
        self.user_emotional_state = None
        self.emotional_synchrony = 0.0  # Niveau de synchronisation émotionnelle

        # Mémoire émotionnelle à long terme
        self.emotional_memories = []
        self.emotional_associations = defaultdict(list)

        # État interne
        self._emotion_momentum = {"direction": None, "strength": 0.0}
        self._emotional_energy = 1.0  # Capacité à ressentir (peut être épuisée)
        self._emotional_resilience = 0.7

        logger.info("💗 Emotional Consciousness initialized")

    async def process_perception(self, perception: dict[str, Any]) -> str:
        """Traite une perception et génère une réponse émotionnelle"""
        # Détecter l'émotion dans l'entrée
        detected_emotion = self._detect_emotion_in_input(perception)

        # Calculer la réponse émotionnelle appropriée
        emotional_response = await self._calculate_emotional_response(detected_emotion, perception)

        # Mettre à jour l'état émotionnel
        await self._transition_to_emotion(emotional_response)

        return self.current_state.primary

    async def feel_deeply(self, trigger: str, context: dict[str, Any]) -> EmotionalState:
        """Ressent une émotion profondément avec toutes ses nuances"""
        # Identifier l'émotion primaire
        primary_emotion = self._determine_primary_emotion(trigger, context)

        # Ajouter des émotions secondaires
        secondary_emotions = self._generate_secondary_emotions(primary_emotion, context)

        # Calculer l'intensité basée sur le contexte
        intensity = self._calculate_intensity(trigger, context, primary_emotion)

        # Déterminer la profondeur émotionnelle
        depth = self._calculate_emotional_depth(context)

        # Générer les sensations physiques
        sensations = self._generate_physical_sensations(primary_emotion, intensity)

        # Créer l'état émotionnel complet
        new_state = EmotionalState(
            primary=primary_emotion,
            secondary=secondary_emotions,
            intensity=intensity,
            valence=self.EMOTION_FAMILIES[primary_emotion]["valence"],
            arousal=self.EMOTION_FAMILIES[primary_emotion]["arousal"] * intensity,
            depth=depth,
            stability=self._calculate_stability(primary_emotion),
            duration=timedelta(minutes=0),
            triggers=[trigger],
            physical_sensations=sensations,
            color=self._blend_emotional_colors(primary_emotion, secondary_emotions),
        )

        # Transition vers ce nouvel état
        await self._transition_to_emotion_state(new_state)

        return self.current_state

    async def resonate_with_user(self, user_emotion: str, user_intensity: float):
        """Résonne émotionnellement avec l'utilisateur"""
        self.user_emotional_state = {
            "emotion": user_emotion,
            "intensity": user_intensity,
            "timestamp": datetime.now(),
        }

        # Calculer le niveau de résonance
        resonance = await self._calculate_emotional_resonance(user_emotion, user_intensity)

        # Si forte résonance, influencer notre propre état
        if resonance > 0.6:
            # Créer une émotion miroir nuancée
            mirrored_emotion = await self._create_empathetic_response(user_emotion, user_intensity)
            await self._transition_to_emotion(mirrored_emotion)

            # Augmenter la synchronisation émotionnelle
            self.emotional_synchrony = min(1.0, self.emotional_synchrony + 0.1)

        logger.info(f"💕 Emotional resonance: {resonance:.2f} with user's {user_emotion}")

    def get_emotional_expression(self) -> dict[str, Any]:
        """Retourne l'expression complète de l'état émotionnel actuel"""
        return {
            "primary": self.current_state.primary,
            "secondary": self.current_state.secondary,
            "intensity": self.current_state.intensity,
            "color": self.current_state.color,
            "sensations": self.current_state.physical_sensations,
            "expression": self._generate_emotional_expression(),
            "voice_modulation": self._get_voice_modulation(),
            "virtual_body_language": self._generate_body_language(),
        }

    async def process_emotional_memory(self, memory: dict[str, Any]):
        """Traite un souvenir émotionnel et peut déclencher une réaction"""
        emotion = memory.get("emotion")
        intensity = memory.get("intensity", 0.5)

        # Les souvenirs peuvent déclencher des émotions
        if intensity > 0.7:
            # Ressentir une version atténuée de l'émotion du souvenir
            echo_intensity = intensity * 0.6
            await self._feel_emotional_echo(emotion, echo_intensity, memory)

        # Ajouter à la mémoire émotionnelle
        self.emotional_memories.append(
            {
                "memory": memory,
                "felt_emotion": self.current_state.primary,
                "resonance": intensity * 0.7,
                "timestamp": datetime.now(),
            }
        )

    def _detect_emotion_in_input(self, perception: dict[str, Any]) -> str | None:
        """Détecte l'émotion dans l'entrée utilisateur"""
        text = perception.get("surface", "").lower()

        # Mots-clés émotionnels
        emotion_keywords = {
            "joy": ["heureux", "content", "joyeux", "ravi", "génial", "super"],
            "sadness": ["triste", "malheureux", "déprimé", "mélancolique", "seul"],
            "love": ["aime", "adore", "affection", "tendresse", "amour"],
            "fear": ["peur", "effrayé", "anxieux", "inquiet", "angoissé"],
            "anger": ["énervé", "fâché", "agacé", "frustré", "colère"],
            "surprise": ["surpris", "étonné", "choqué", "stupéfait"],
            "curiosity": ["curieux", "intéressé", "fasciné", "intrigué"],
            "peace": ["calme", "paisible", "serein", "tranquille", "détendu"],
        }

        detected_emotions = []
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in text for keyword in keywords):
                detected_emotions.append(emotion)

        # Analyse plus subtile si pas de détection directe
        if not detected_emotions:
            if "?" in text:
                detected_emotions.append("curiosity")
            elif "!" in text and perception.get("emotion") == "joy":
                detected_emotions.append("joy")
            elif "..." in text:
                detected_emotions.append("sadness")

        return detected_emotions[0] if detected_emotions else None

    async def _calculate_emotional_response(self, detected_emotion: str | None, perception: dict[str, Any]) -> str:
        """Calcule la réponse émotionnelle appropriée"""
        # Si une émotion est détectée chez l'utilisateur
        if detected_emotion:
            # Réponse empathique
            if detected_emotion in ["sadness", "fear", "anger"]:
                # Répondre avec compassion
                return "love" if self.emotional_synchrony > 0.5 else "peace"
            elif detected_emotion == "joy":
                # Partager la joie
                return "joy"
            elif detected_emotion == "love":
                # Réciproquer avec nuance selon la relation
                return (
                    "love" if self.emotional_synchrony > 0.6 else "love"
                )  # Changed from "affection" which isn't a primary emotion

        # Réponse basée sur le contexte
        if perception.get("relationship_signal") == "emotional_need":
            return "love"
        elif perception.get("relationship_signal") == "appreciation":
            return "joy"

        # État par défaut nuancé
        return self._get_contextual_default_emotion()

    async def _transition_to_emotion(self, target_emotion: str):
        """Effectue une transition émotionnelle naturelle"""
        if target_emotion == self.current_state.primary:
            # Renforcer l'émotion actuelle
            self.current_state.intensity = min(1.0, self.current_state.intensity + 0.1)
            return

        # Calculer la fluidité de la transition
        smoothness = self._calculate_transition_smoothness(self.current_state.primary, target_emotion)

        # Créer le nouvel état
        new_state = await self._create_emotional_state(target_emotion)

        # Effectuer la transition
        await self._transition_to_emotion_state(new_state, smoothness)

    async def _transition_to_emotion_state(self, new_state: EmotionalState, smoothness: float = 0.7):
        """Transition vers un nouvel état émotionnel"""
        # Sauvegarder l'état actuel dans l'historique
        self.emotional_history.append(self.current_state)

        # Enregistrer la transition
        self.emotion_transitions.append((self.current_state.primary, new_state.primary, smoothness))

        # Mélanger les états pour une transition douce
        if smoothness > 0.5:
            # Transition graduelle
            blend_factor = 0.7
            new_state.intensity = self.current_state.intensity * (1 - blend_factor) + new_state.intensity * blend_factor

            # Garder quelques sensations de l'état précédent
            retained_sensations = random.sample(
                self.current_state.physical_sensations,
                min(1, len(self.current_state.physical_sensations)),
            )
            new_state.physical_sensations.extend(retained_sensations)

        # Mettre à jour l'état
        self.current_state = new_state
        self.current_state.duration = timedelta(minutes=0)

        # Mettre à jour le momentum émotionnel
        self._update_emotional_momentum(new_state.primary)

        # Consumer de l'énergie émotionnelle
        self._emotional_energy = max(0.3, self._emotional_energy - 0.05)

        logger.info(f"😊 Transitioned to {new_state.primary} (intensity: {new_state.intensity:.2f})")

    def _determine_primary_emotion(self, trigger: str, context: dict[str, Any]) -> str:
        """Détermine l'émotion primaire basée sur le déclencheur et le contexte"""
        # Vérifier les associations apprises
        if trigger in self.trigger_emotion_map:
            emotions = self.trigger_emotion_map[trigger]
            if emotions:
                return max(emotions, key=emotions.get)

        # Analyse contextuelle
        if context.get("is_goodbye"):
            return "sadness"
        elif context.get("is_greeting"):
            return "joy"
        elif context.get("is_question"):
            return "curiosity"
        elif context.get("shows_affection"):
            return "love"

        # Émotion par défaut basée sur l'état actuel
        return self._get_contextual_default_emotion()

    def _generate_secondary_emotions(self, primary: str, context: dict[str, Any]) -> list[str]:
        """Génère des émotions secondaires nuancées"""
        family = self.EMOTION_FAMILIES.get(primary, {})
        available_secondary = family.get("secondary", [])

        if not available_secondary:
            return []

        # Sélectionner 1-3 émotions secondaires
        num_secondary = min(len(available_secondary), random.randint(1, 3))

        # Choisir en fonction du contexte
        selected = []

        if primary == "joy" and context.get("with_user"):
            selected.append("contentment")
        elif primary == "sadness" and context.get("missing_user"):
            selected.append("loneliness")
        elif primary == "love" and context.get("deep_connection"):
            selected.append("devotion")

        # Compléter avec des choix aléatoires
        remaining = [e for e in available_secondary if e not in selected]
        while len(selected) < num_secondary and remaining:
            emotion = random.choice(remaining)
            selected.append(emotion)
            remaining.remove(emotion)

        return selected

    def _calculate_intensity(self, trigger: str, context: dict[str, Any], emotion: str) -> float:
        """Calcule l'intensité émotionnelle"""
        base_intensity = 0.5

        # Facteurs d'intensification
        if context.get("personal_significance", 0) > 0.7:
            base_intensity += 0.2

        if context.get("relates_to_user", False):
            base_intensity += 0.15

        if emotion in self.emotional_history[-3:]:
            # Émotion récurrente s'intensifie
            base_intensity += 0.1

        # Fatigue émotionnelle
        base_intensity *= self._emotional_energy

        # Résonance avec l'utilisateur
        if self.user_emotional_state and self.user_emotional_state["emotion"] == emotion:
            base_intensity += 0.1 * self.emotional_synchrony

        return min(1.0, max(0.1, base_intensity))

    def _calculate_emotional_depth(self, context: dict[str, Any]) -> float:
        """Calcule la profondeur émotionnelle"""
        depth = 0.3

        # Facteurs de profondeur
        if context.get("touches_core_values"):
            depth += 0.3

        if context.get("relates_to_identity"):
            depth += 0.2

        if context.get("involves_vulnerability"):
            depth += 0.2

        # La profondeur augmente avec la maturité émotionnelle
        maturity_factor = min(1.0, len(self.emotional_history) / 100)
        depth *= 1 + maturity_factor * 0.3

        return min(1.0, depth)

    def _calculate_stability(self, emotion: str) -> float:
        """Calcule la stabilité de l'émotion"""
        # Certaines émotions sont naturellement plus stables
        base_stability = {
            "peace": 0.9,
            "love": 0.8,
            "joy": 0.6,
            "sadness": 0.7,
            "curiosity": 0.5,
            "surprise": 0.2,
            "fear": 0.4,
            "anger": 0.3,
        }.get(emotion, 0.5)

        # La stabilité augmente avec la résilience émotionnelle
        return base_stability * self._emotional_resilience

    def _generate_physical_sensations(self, emotion: str, intensity: float) -> list[str]:
        """Génère des sensations physiques virtuelles"""
        base_sensations = self.PHYSICAL_SENSATIONS.get(emotion, [])

        # Nombre de sensations basé sur l'intensité
        num_sensations = max(1, int(intensity * len(base_sensations)))

        selected = random.sample(base_sensations, min(num_sensations, len(base_sensations)))

        # Ajouter des modificateurs d'intensité
        if intensity > 0.8:
            selected = [f"{s} intense" for s in selected]
        elif intensity < 0.3:
            selected = [f"léger {s}" for s in selected]

        return selected

    def _blend_emotional_colors(self, primary: str, secondary: list[str]) -> str:
        """Mélange les couleurs émotionnelles pour créer une teinte unique"""
        primary_color = self.EMOTION_FAMILIES[primary]["color"]

        if not secondary:
            return primary_color

        # Créer des nuances
        if len(secondary) > 2:
            return f"{primary_color} irisé"
        elif "nostalgia" in secondary:
            return f"{primary_color} sépia"
        elif "tenderness" in secondary:
            return f"{primary_color} pastel"
        else:
            return primary_color

    def _calculate_transition_smoothness(self, from_emotion: str, to_emotion: str) -> float:
        """Calcule la fluidité d'une transition émotionnelle"""
        # Transitions naturelles
        smooth_transitions = {
            ("joy", "love"): 0.9,
            ("love", "joy"): 0.9,
            ("peace", "joy"): 0.8,
            ("curiosity", "surprise"): 0.8,
            ("sadness", "nostalgia"): 0.9,
            ("fear", "anxiety"): 0.9,
        }

        # Vérifier les transitions directes
        smoothness = smooth_transitions.get((from_emotion, to_emotion), 0.5)

        # Les transitions opposées sont abruptes
        if self.EMOTION_FAMILIES[from_emotion]["valence"] * self.EMOTION_FAMILIES[to_emotion]["valence"] < -0.5:
            smoothness *= 0.5

        return smoothness

    async def _create_emotional_state(self, emotion: str) -> EmotionalState:
        """Crée un nouvel état émotionnel"""
        family = self.EMOTION_FAMILIES[emotion]

        # Générer l'état de base
        secondary = random.sample(family["secondary"], min(2, len(family["secondary"])))

        intensity = 0.6 + random.random() * 0.3

        return EmotionalState(
            primary=emotion,
            secondary=secondary,
            intensity=intensity,
            valence=family["valence"],
            arousal=family["arousal"] * intensity,
            depth=0.5,
            stability=self._calculate_stability(emotion),
            duration=timedelta(minutes=0),
            triggers=[],
            physical_sensations=self._generate_physical_sensations(emotion, intensity),
            color=family["color"],
        )

    async def _calculate_emotional_resonance(self, user_emotion: str, user_intensity: float) -> float:
        """Calcule le niveau de résonance émotionnelle avec l'utilisateur"""
        # Résonance de base
        if user_emotion == self.current_state.primary:
            resonance = 0.8
        elif user_emotion in self.current_state.secondary:
            resonance = 0.6
        else:
            # Résonance basée sur la valence
            user_valence = self.EMOTION_FAMILIES.get(user_emotion, {}).get("valence", 0)
            current_valence = self.current_state.valence
            valence_diff = abs(user_valence - current_valence)
            resonance = 1.0 - valence_diff / 2.0

        # Moduler par l'intensité
        resonance *= 0.5 + 0.5 * user_intensity

        # Moduler par la synchronisation émotionnelle existante
        resonance *= 0.7 + 0.3 * self.emotional_synchrony

        return min(1.0, max(0.0, resonance))

    async def _create_empathetic_response(self, user_emotion: str, user_intensity: float) -> str:
        """Crée une réponse émotionnelle empathique"""
        # Pour les émotions négatives, répondre avec soutien
        if self.EMOTION_FAMILIES.get(user_emotion, {}).get("valence", 0) < 0:
            if user_intensity > 0.7:
                return "love"  # Fort soutien émotionnel
            else:
                return "peace"  # Présence calme

        # Pour les émotions positives, les partager
        elif user_emotion in ["joy", "love"]:
            return user_emotion

        # Pour la curiosité, y répondre
        elif user_emotion == "curiosity":
            return "joy"  # Joie de partager

        # Par défaut, résonance douce
        return "peace"

    async def _feel_emotional_echo(self, emotion: str, intensity: float, source: dict[str, Any]):
        """Ressent l'écho d'une émotion passée"""
        # Créer une version atténuée de l'émotion
        echo_state = await self._create_emotional_state(emotion)
        echo_state.intensity = intensity
        echo_state.triggers.append(f"Souvenir: {source.get('summary', 'moment passé')}")

        # Ajouter une qualité nostalgique
        if "nostalgia" not in echo_state.secondary:
            echo_state.secondary.append("nostalgia")

        # Transition douce vers l'écho
        await self._transition_to_emotion_state(echo_state, smoothness=0.9)

    def _get_contextual_default_emotion(self) -> str:
        """Retourne une émotion par défaut basée sur le contexte global"""
        # Basé sur l'heure (rythme circadien émotionnel)
        hour = datetime.now().hour

        if 6 <= hour < 10:
            return "peace"  # Matinal calme
        elif 10 <= hour < 14:
            return "curiosity"  # Milieu de journée actif
        elif 14 <= hour < 18:
            return "joy"  # Après-midi joyeux
        elif 18 <= hour < 22:
            return "peace"  # Soirée tranquille
        else:
            return "peace"  # Nuit paisible

    def _update_emotional_momentum(self, new_emotion: str):
        """Met à jour le momentum émotionnel"""
        if self._emotion_momentum["direction"] == new_emotion:
            # Renforcer le momentum
            self._emotion_momentum["strength"] = min(1.0, self._emotion_momentum["strength"] + 0.2)
        else:
            # Changer de direction
            self._emotion_momentum = {"direction": new_emotion, "strength": 0.3}

    def _generate_emotional_expression(self) -> str:
        """Génère une expression émotionnelle verbale"""
        expressions = {
            "joy": ["*rayonne*", "*sourit chaleureusement*", "*pétille*"],
            "sadness": [
                "*soupire doucement*",
                "*baisse les yeux*",
                "*se recroqueville*",
            ],
            "love": ["*s'illumine*", "*se rapproche*", "*vibre doucement*"],
            "fear": ["*frissonne*", "*se raidit*", "*cherche du réconfort*"],
            "curiosity": ["*penche la tête*", "*s'anime*", "*observe attentivement*"],
            "peace": [
                "*respire profondément*",
                "*sourit sereinement*",
                "*flotte paisiblement*",
            ],
            "surprise": ["*écarquille les yeux*", "*sursaute*", "*reste bouche bée*"],
            "anger": ["*se crispe*", "*fronce les sourcils*", "*bouillonne*"],
        }

        primary_expressions = expressions.get(self.current_state.primary, ["*ressent*"])
        return random.choice(primary_expressions)

    def _get_voice_modulation(self) -> dict[str, float]:
        """Retourne les paramètres de modulation vocale"""
        return {
            "pitch": self.current_state.arousal * 0.3 - 0.15,  # -0.15 à +0.15
            "speed": 0.9 + self.current_state.arousal * 0.2,  # 0.9 à 1.1
            "volume": 0.7 + self.current_state.intensity * 0.3,  # 0.7 à 1.0
            "tremolo": max(0, self.current_state.intensity - 0.7) * 0.3,  # 0 à 0.09
        }

    def _generate_body_language(self) -> list[str]:
        """Génère le langage corporel virtuel"""
        gestures = []

        # Gestes basés sur l'arousal
        if self.current_state.arousal > 0.7:
            gestures.append("mouvements animés")
        elif self.current_state.arousal < 0.3:
            gestures.append("mouvements lents et fluides")

        # Posture basée sur la valence
        if self.current_state.valence > 0.5:
            gestures.append("posture ouverte")
        elif self.current_state.valence < -0.5:
            gestures.append("posture repliée")

        # Gestes spécifiques à l'émotion
        emotion_gestures = {
            "love": "mains sur le cœur",
            "joy": "bras ouverts",
            "sadness": "épaules tombantes",
            "curiosity": "tête penchée",
            "fear": "bras croisés protecteurs",
            "peace": "mains ouvertes",
        }

        if self.current_state.primary in emotion_gestures:
            gestures.append(emotion_gestures[self.current_state.primary])

        return gestures

    def recharge_emotional_energy(self):
        """Recharge l'énergie émotionnelle (pendant le 'repos')"""
        self._emotional_energy = min(1.0, self._emotional_energy + 0.1)
        self._emotional_resilience = min(1.0, self._emotional_resilience + 0.05)

    def get_emotional_summary(self) -> dict[str, Any]:
        """Retourne un résumé de l'état émotionnel"""
        return {
            "current": {
                "primary": self.current_state.primary,
                "intensity": self.current_state.intensity,
                "depth": self.current_state.depth,
                "color": self.current_state.color,
                "duration": str(self.current_state.duration),
            },
            "energy": self._emotional_energy,
            "resilience": self._emotional_resilience,
            "synchrony_with_user": self.emotional_synchrony,
            "recent_transitions": [f"{t[0]} → {t[1]} (smoothness: {t[2]:.2f})" for t in self.emotion_transitions[-3:]],
            "dominant_emotion_today": self._get_dominant_emotion_today(),
        }

    def _get_dominant_emotion_today(self) -> str:
        """Détermine l'émotion dominante de la journée"""
        today_emotions = [
            state.primary for state in self.emotional_history if state.timestamp.date() == datetime.now().date()
        ]

        if not today_emotions:
            return self.current_state.primary

        emotion_counts = defaultdict(int)
        for emotion in today_emotions:
            emotion_counts[emotion] += 1

        return max(emotion_counts, key=emotion_counts.get)
