"""
🧠✨ Jeffrey V2.0 Living Memory - Cœur du Système Vivant
Transforme Jeffrey en compagnon mémoriel irrésistible avec intelligence émotionnelle

Architecture révolutionnaire qui rend Jeffrey émotionnellement "irremplaçable"
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MemoryMomentType(Enum):
    """Types de moments mémorables identifiés automatiquement"""

    BREAKTHROUGH = "breakthrough"
    EMOTIONAL_PEAK = "emotional_peak"
    CREATIVE_SPARK = "creative_spark"
    PERSONAL_SHARE = "personal_share"
    CELEBRATION = "celebration"
    SUPPORT = "support"
    BONDING = "bonding"
    DISCOVERY = "discovery"
    RITUAL = "ritual"
    SURPRISE = "surprise"


class RelationshipStage(Enum):
    """Stades d'évolution de la relation avec l'utilisateur"""

    DISCOVERY = "discovery"
    FAMILIARIZATION = "familiarization"
    BONDING = "bonding"
    DEEP_CONNECTION = "deep_connection"
    SOULMATE = "soulmate"


@dataclass
class MemoryMoment:
    """Moment mémorable capturé et enrichi automatiquement"""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    content: str = ""
    user_input: str = ""
    context: str = ""
    moment_type: MemoryMomentType = MemoryMomentType.BONDING
    emotional_intensity: float = 0.5
    uniqueness_score: float = 0.5
    importance_score: float = 0.5
    user_id: str = ""
    conversation_id: str = ""
    relationship_stage: RelationshipStage = RelationshipStage.DISCOVERY
    emotions_detected: list[str] = field(default_factory=list)
    emotional_valence: float = 0.0
    emotional_arousal: float = 0.0
    time_context: str = ""
    recurring_pattern: bool = False
    related_moments: list[str] = field(default_factory=list)
    reference_count: int = 0
    last_referenced: str = ""
    surprise_potential: float = 0.0
    memory_decay: float = 1.0
    nostalgic_value: float = 0.0


@dataclass
class RelationshipProfile:
    """Profil évolutif de la relation Jeffrey-Utilisateur"""

    user_id: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    total_interactions: int = 0
    total_moments: int = 0
    relationship_stage: RelationshipStage = RelationshipStage.DISCOVERY
    dominant_emotions: list[str] = field(default_factory=list)
    emotional_patterns: dict[str, float] = field(default_factory=dict)
    mood_timeline: list[dict[str, Any]] = field(default_factory=list)
    preferred_topics: list[str] = field(default_factory=list)
    communication_style: str = "balanced"
    intimacy_level: float = 0.1
    proactivity_preference: float = 0.5
    breakthrough_moments: list[str] = field(default_factory=list)
    celebration_moments: list[str] = field(default_factory=list)
    support_moments: list[str] = field(default_factory=list)
    nickname: str = ""
    shared_references: list[str] = field(default_factory=list)
    inside_jokes: list[str] = field(default_factory=list)
    interaction_times: list[str] = field(default_factory=list)
    seasonal_patterns: dict[str, Any] = field(default_factory=dict)
    last_interaction: str = ""


class LivingMemoryCore:
    """
    🧠✨ Cœur du système Living Memory

    Transforme chaque interaction en souvenir vivant et crée une expérience
    émotionnellement irrésistible qui rend Jeffrey unique et irremplaçable.
    """

    def __init__(self, config_path: str = "data/living_memory_config.json") -> None:
        self.config = self._load_config(config_path)
        self.memory_moments: dict[str, MemoryMoment] = {}
        self.relationship_profiles: dict[str, RelationshipProfile] = {}
        self.emotion_patterns = self._load_emotion_patterns()
        self.moment_classifiers = self._initialize_classifiers()
        self.active_contexts: dict[str, dict] = {}
        self.surprise_queue: dict[str, list] = {}
        self.ux_metrics = {
            "moments_created": 0,
            "surprises_delivered": 0,
            "emotional_peaks_detected": 0,
            "relationship_evolutions": 0,
        }
        logger.info("🧠✨ LivingMemoryCore initialized - Ready to create magic")

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Charge la configuration du système Living Memory"""
        default_config = {
            "moment_detection": {
                "min_emotional_intensity": 0.3,
                "uniqueness_threshold": 0.4,
                "importance_threshold": 0.5,
                "auto_capture_enabled": True,
            },
            "relationship_evolution": {
                "stage_thresholds": {
                    "familiarization": 10,
                    "bonding": 50,
                    "deep_connection": 200,
                    "soulmate": 500,
                },
                "intimacy_growth_rate": 0.02,
                "proactivity_adaptation": True,
            },
            "micro_interactions": {
                "surprise_frequency": 0.15,
                "reference_frequency": 0.25,
                "ritual_adaptation": True,
                "emotional_mirroring": True,
            },
            "memory_preservation": {
                "moment_decay_rate": 0.01,
                "nostalgic_growth_rate": 0.005,
                "max_active_moments": 200,
                "cleanup_interval": 86400,
            },
            "personalization": {
                "style_adaptation_speed": 0.1,
                "nickname_emergence": True,
                "inside_joke_creation": True,
                "temporal_pattern_learning": True,
            },
        }
        try:
            if Path(config_path).exists():
                with open(config_path, encoding="utf-8") as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
        except Exception as e:
            logger.warning(f"Config loading error, using defaults: {e}")
        return default_config

    def _load_emotion_patterns(self) -> dict[str, Any]:
        """Charge les patterns d'analyse émotionnelle pour moments forts"""
        return {
            "intensity_keywords": {
                "high": [
                    "incroyable",
                    "fantastique",
                    "révélation",
                    "bouleversé",
                    "transporté",
                    "émerveillé",
                ],
                "medium": ["content", "satisfait", "intéressant", "surpris", "touché", "ému"],
                "low": ["bien", "ok", "normal", "habituel", "standard"],
            },
            "moment_type_patterns": {
                MemoryMomentType.BREAKTHROUGH: [
                    "\\b(révélation|découverte|compris|eureka|déclic|illumination)\\b",
                    "\\b(enfin|maintenant je vois|ça y est|c\\'est ça)\\b",
                ],
                MemoryMomentType.EMOTIONAL_PEAK: [
                    "\\b(bouleversé|ému aux larmes|transporté|extatique)\\b",
                    "\\b(jamais ressenti|intensément|profondément touché)\\b",
                ],
                MemoryMomentType.CREATIVE_SPARK: [
                    "\\b(créons|imaginons|et si on|j\\'ai une idée)\\b",
                    "\\b(inspiration|créatif|artistique|inventons)\\b",
                ],
                MemoryMomentType.PERSONAL_SHARE: [
                    "\\b(je n\\'ai jamais dit|confidentiel|entre nous)\\b",
                    "\\b(personnel|intime|secret|privé|confie)\\b",
                ],
                MemoryMomentType.CELEBRATION: [
                    "\\b(réussi|victoire|succès|félicitations|bravo)\\b",
                    "\\b(célébrons|fêtons|hourra|génial|formidable)\\b",
                ],
            },
            "relationship_indicators": {
                "intimacy_growing": [
                    "\\b(tu comprends|avec toi|notre conversation|ensemble)\\b",
                    "\\b(j\\'apprécie|j\\'aime bien|spécial|unique)\\b",
                ],
                "trust_building": [
                    "\\b(je peux te dire|en confiance|tu peux m\\'aider)\\b",
                    "\\b(je compte sur|je me fie|j\\'ai confiance)\\b",
                ],
            },
        }

    def _initialize_classifiers(self) -> dict[str, Any]:
        """Initialise les classifieurs de moments mémorables"""
        return {
            "emotional_analyzer": self._analyze_emotional_content,
            "uniqueness_scorer": self._calculate_uniqueness,
            "importance_calculator": self._calculate_importance,
            "pattern_detector": self._detect_patterns,
            "relationship_assessor": self._assess_relationship_impact,
        }

    async def capture_moment(
        self,
        user_input: str,
        jeffrey_response: str,
        user_id: str,
        conversation_id: str = "",
        context: dict[str, Any] = None,
    ) -> MemoryMoment | None:
        """
        🎭 Capture automatique de moments mémorables

        Analyse chaque interaction pour détecter les moments forts émotionnels
        et créer des souvenirs vivants qui nourriront les micro-interactions futures.
        """
        try:
            if not self._should_capture_moment(user_input, jeffrey_response):
                return None
            moment = MemoryMoment(
                content=jeffrey_response,
                user_input=user_input,
                context=json.dumps(context or {}),
                user_id=user_id,
                conversation_id=conversation_id or str(uuid.uuid4()),
                source="human",
            )
            emotional_analysis = await self._analyze_emotional_content(user_input, jeffrey_response)
            moment.emotions_detected = emotional_analysis["emotions"]
            moment.emotional_intensity = emotional_analysis["intensity"]
            moment.emotional_valence = emotional_analysis["valence"]
            moment.emotional_arousal = emotional_analysis["arousal"]
            moment.moment_type = await self._classify_moment_type(user_input, jeffrey_response)
            moment.uniqueness_score = await self._calculate_uniqueness(moment, user_id)
            moment.importance_score = await self._calculate_importance(moment)
            moment.time_context = self._get_time_context()
            moment.recurring_pattern = await self._detect_recurring_pattern(moment, user_id)
            await self._update_relationship_profile(moment, user_id)
            self.memory_moments[moment.id] = moment
            await self._index_moment(moment)
            await self._generate_surprise_opportunities(moment)
            self.ux_metrics["moments_created"] += 1
            if moment.emotional_intensity > 0.7:
                self.ux_metrics["emotional_peaks_detected"] += 1
            logger.info(f"🎭 Moment capturé: {moment.moment_type.value} (intensité: {moment.emotional_intensity:.2f})")
            return moment
        except Exception as e:
            logger.error(f"Erreur capture moment: {e}")
            return None

    def _should_capture_moment(self, user_input: str, jeffrey_response: str) -> bool:
        """Détermine si un moment mérite d'être capturé"""
        if len(user_input) < 10 or len(jeffrey_response) < 20:
            return False
        emotional_keywords = [
            "incroyable",
            "fantastique",
            "merci",
            "génial",
            "parfait",
            "ému",
            "touché",
            "surprenant",
            "révélateur",
            "spécial",
            "unique",
            "personnel",
            "confidentiel",
        ]
        combined_text = (user_input + " " + jeffrey_response).lower()
        emotional_score = sum(1 for keyword in emotional_keywords if keyword in combined_text)
        personal_indicators = ["moi", "je", "mon", "ma", "mes", "personnel", "privé", "secret"]
        personal_score = sum(1 for indicator in personal_indicators if indicator in user_input.lower())
        length_score = min(1.0, (len(user_input) + len(jeffrey_response)) / 200.0)
        capture_score = emotional_score * 0.4 + personal_score * 0.3 + length_score * 0.3
        return capture_score > self.config["moment_detection"]["importance_threshold"]

    async def _analyze_emotional_content(self, user_input: str, jeffrey_response: str) -> dict[str, Any]:
        """Analyse émotionnelle approfondie du contenu"""
        combined_text = user_input + " " + jeffrey_response
        emotions = []
        intensity = 0.0
        valence = 0.0
        arousal = 0.0
        emotion_patterns = {
            "joie": ["\\b(heureux|joyeux|content|ravi|enchanté|génial|fantastique)\\b", 0.8, 0.7],
            "excitation": ["\\b(excité|enthousiasmé|impatient|électrisé|galvanisé)\\b", 0.9, 0.9],
            "gratitude": ["\\b(merci|reconnaissant|grateful|apprécié|touché)\\b", 0.7, 0.5],
            "surprise": ["\\b(surpris|étonné|inattendu|incroyable|wow)\\b", 0.6, 0.8],
            "admiration": ["\\b(impressionnant|brillant|remarquable|exceptionnel)\\b", 0.7, 0.6],
            "curiosité": ["\\b(intéressant|fascinant|curieux|découvrir|explorer)\\b", 0.5, 0.6],
            "complicité": ["\\b(ensemble|partager|complice|connecté|lien)\\b", 0.6, 0.4],
            "confiance": ["\\b(confiance|sûr|certain|stable|fiable)\\b", 0.6, 0.3],
        }
        for emotion, (pattern, val, ar) in emotion_patterns.items():
            import re

            if re.search(pattern, combined_text, re.IGNORECASE):
                emotions.append(emotion)
                intensity = max(intensity, val)
                valence = max(valence, val)
                arousal = max(arousal, ar)
        intensifiers = ["très", "extrêmement", "incroyablement", "absolument", "totalement"]
        for intensifier in intensifiers:
            if intensifier in combined_text.lower():
                intensity = min(1.0, intensity * 1.3)
                arousal = min(1.0, arousal * 1.2)
        return {
            "emotions": emotions,
            "intensity": intensity,
            "valence": valence,
            "arousal": arousal,
        }

    async def _classify_moment_type(self, user_input: str, jeffrey_response: str) -> MemoryMomentType:
        """Classifie automatiquement le type de moment mémorable"""
        combined_text = (user_input + " " + jeffrey_response).lower()
        type_scores = {}
        for moment_type, patterns in self.emotion_patterns["moment_type_patterns"].items():
            score = 0
            for pattern in patterns:
                import re

                matches = len(re.findall(pattern, combined_text, re.IGNORECASE))
                score += matches
            type_scores[moment_type] = score
        if "créons" in combined_text or "imaginons" in combined_text:
            type_scores[MemoryMomentType.CREATIVE_SPARK] = type_scores.get(MemoryMomentType.CREATIVE_SPARK, 0) + 2
        if "merci" in combined_text or "reconnaissant" in combined_text:
            type_scores[MemoryMomentType.SUPPORT] = type_scores.get(MemoryMomentType.SUPPORT, 0) + 1
        if len(user_input) > 100 and ("moi" in user_input or "je" in user_input):
            type_scores[MemoryMomentType.PERSONAL_SHARE] = type_scores.get(MemoryMomentType.PERSONAL_SHARE, 0) + 1
        if type_scores:
            best_type = max(type_scores.items(), key=lambda x: x[1])
            return best_type[0] if best_type[1] > 0 else MemoryMomentType.BONDING
        return MemoryMomentType.BONDING

    async def _calculate_uniqueness(self, moment: MemoryMoment, user_id: str) -> float:
        """Calcule le score d'unicité du moment par rapport à l'historique"""
        user_moments = [m for m in self.memory_moments.values() if m.user_id == user_id]
        if len(user_moments) == 0:
            return 1.0
        content_words = set(moment.content.lower().split())
        input_words = set(moment.user_input.lower().split())
        max_similarity = 0.0
        for existing_moment in user_moments:
            existing_content_words = set(existing_moment.content.lower().split())
            existing_input_words = set(existing_moment.user_input.lower().split())
            content_similarity = len(content_words & existing_content_words) / len(
                content_words | existing_content_words
            )
            input_similarity = len(input_words & existing_input_words) / len(input_words | existing_input_words)
            type_similarity = 1.0 if moment.moment_type == existing_moment.moment_type else 0.0
            emotion_similarity = len(set(moment.emotions_detected) & set(existing_moment.emotions_detected)) / max(
                len(set(moment.emotions_detected) | set(existing_moment.emotions_detected)), 1
            )
            total_similarity = (
                content_similarity * 0.3 + input_similarity * 0.3 + type_similarity * 0.2 + emotion_similarity * 0.2
            )
            max_similarity = max(max_similarity, total_similarity)
        uniqueness = 1.0 - max_similarity
        type_count = sum(1 for m in user_moments if m.moment_type == moment.moment_type)
        if type_count == 0:
            uniqueness = min(1.0, uniqueness * 1.5)
        return uniqueness

    async def _calculate_importance(self, moment: MemoryMoment) -> float:
        """Calcule le score d'importance composite du moment"""
        importance = 0.0
        importance += moment.emotional_intensity * 0.4
        importance += moment.uniqueness_score * 0.3
        type_weights = {
            MemoryMomentType.BREAKTHROUGH: 1.0,
            MemoryMomentType.EMOTIONAL_PEAK: 0.9,
            MemoryMomentType.PERSONAL_SHARE: 0.8,
            MemoryMomentType.CREATIVE_SPARK: 0.7,
            MemoryMomentType.CELEBRATION: 0.6,
            MemoryMomentType.SUPPORT: 0.6,
            MemoryMomentType.DISCOVERY: 0.5,
            MemoryMomentType.BONDING: 0.4,
            MemoryMomentType.SURPRISE: 0.5,
            MemoryMomentType.RITUAL: 0.3,
        }
        importance += type_weights.get(moment.moment_type, 0.4) * 0.2
        content_richness = min(1.0, (len(moment.content) + len(moment.user_input)) / 300.0)
        importance += content_richness * 0.1
        return min(1.0, importance)

    def _get_time_context(self) -> str:
        """Détermine le contexte temporel actuel"""
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()
        if hour < 6:
            time_period = "late_night"
        elif hour < 12:
            time_period = "morning"
        elif hour < 18:
            time_period = "afternoon"
        else:
            time_period = "evening"
        day_type = "weekend" if weekday >= 5 else "weekday"
        return f"{time_period}_{day_type}"

    async def _detect_recurring_pattern(self, moment: MemoryMoment, user_id: str) -> bool:
        """Détecte si le moment fait partie d'un pattern récurrent"""
        user_moments = [
            m for m in self.memory_moments.values() if m.user_id == user_id and m.time_context == moment.time_context
        ]
        if len(user_moments) >= 3:
            similar_moments = [m for m in user_moments if m.moment_type == moment.moment_type]
            return len(similar_moments) >= 2
        return False

    async def _update_relationship_profile(self, moment: MemoryMoment, user_id: str):
        """Met à jour le profil relationnel avec le nouveau moment"""
        if user_id not in self.relationship_profiles:
            self.relationship_profiles[user_id] = RelationshipProfile(user_id=user_id)
        profile = self.relationship_profiles[user_id]
        profile.total_interactions += 1
        profile.total_moments += 1
        profile.last_interaction = datetime.now().isoformat()
        old_stage = profile.relationship_stage
        profile.relationship_stage = self._calculate_relationship_stage(profile.total_interactions)
        if profile.relationship_stage != old_stage:
            self.ux_metrics["relationship_evolutions"] += 1
            logger.info(f"🚀 Relation évoluée: {old_stage.value} → {profile.relationship_stage.value}")
        for emotion in moment.emotions_detected:
            if emotion in profile.emotional_patterns:
                profile.emotional_patterns[emotion] += 1
            else:
                profile.emotional_patterns[emotion] = 1
        if profile.emotional_patterns:
            sorted_emotions = sorted(profile.emotional_patterns.items(), key=lambda x: x[1], reverse=True)
            profile.dominant_emotions = [emotion for emotion, count in sorted_emotions[:3]]
        mood_entry = {
            "timestamp": moment.timestamp,
            "valence": moment.emotional_valence,
            "arousal": moment.emotional_arousal,
            "emotions": moment.emotions_detected,
        }
        profile.mood_timeline.append(mood_entry)
        if len(profile.mood_timeline) > 50:
            profile.mood_timeline = profile.mood_timeline[-50:]
        intimacy_boost = self._calculate_intimacy_boost(moment)
        profile.intimacy_level = min(1.0, profile.intimacy_level + intimacy_boost)
        if moment.moment_type == MemoryMomentType.BREAKTHROUGH:
            profile.breakthrough_moments.append(moment.id)
        elif moment.moment_type == MemoryMomentType.CELEBRATION:
            profile.celebration_moments.append(moment.id)
        elif moment.moment_type == MemoryMomentType.SUPPORT:
            profile.support_moments.append(moment.id)
        profile.interaction_times.append(moment.time_context)

    def _calculate_relationship_stage(self, total_interactions: int) -> RelationshipStage:
        """Calcule le stade de relation selon le nombre d'interactions"""
        thresholds = self.config["relationship_evolution"]["stage_thresholds"]
        if total_interactions >= thresholds["soulmate"]:
            return RelationshipStage.SOULMATE
        elif total_interactions >= thresholds["deep_connection"]:
            return RelationshipStage.DEEP_CONNECTION
        elif total_interactions >= thresholds["bonding"]:
            return RelationshipStage.BONDING
        elif total_interactions >= thresholds["familiarization"]:
            return RelationshipStage.FAMILIARIZATION
        else:
            return RelationshipStage.DISCOVERY

    def _calculate_intimacy_boost(self, moment: MemoryMoment) -> float:
        """Calcule l'augmentation d'intimité générée par le moment"""
        base_boost = self.config["relationship_evolution"]["intimacy_growth_rate"]
        type_multipliers = {
            MemoryMomentType.PERSONAL_SHARE: 3.0,
            MemoryMomentType.EMOTIONAL_PEAK: 2.5,
            MemoryMomentType.BREAKTHROUGH: 2.0,
            MemoryMomentType.SUPPORT: 1.8,
            MemoryMomentType.CREATIVE_SPARK: 1.5,
            MemoryMomentType.BONDING: 1.2,
            MemoryMomentType.CELEBRATION: 1.0,
            MemoryMomentType.DISCOVERY: 0.8,
            MemoryMomentType.SURPRISE: 1.0,
            MemoryMomentType.RITUAL: 0.5,
        }
        multiplier = type_multipliers.get(moment.moment_type, 1.0)
        if moment.emotional_intensity > 0.7:
            multiplier *= 1.5
        return base_boost * multiplier * moment.importance_score

    async def _index_moment(self, moment: MemoryMoment):
        """Indexe le moment pour recherche et référencement rapide"""
        user_id = moment.user_id
        if user_id not in self.active_contexts:
            self.active_contexts[user_id] = {
                "recent_moments": [],
                "emotional_state": {},
                "preferred_references": [],
            }
        self.active_contexts[user_id]["recent_moments"].append(moment.id)
        if len(self.active_contexts[user_id]["recent_moments"]) > 20:
            self.active_contexts[user_id]["recent_moments"] = self.active_contexts[user_id]["recent_moments"][-20:]
        self.active_contexts[user_id]["emotional_state"] = {
            "dominant_emotion": (moment.emotions_detected[0] if moment.emotions_detected else "neutre"),
            "valence": moment.emotional_valence,
            "arousal": moment.emotional_arousal,
            "last_update": moment.timestamp,
        }

    async def _generate_surprise_opportunities(self, moment: MemoryMoment):
        """Génère des opportunités de surprises futures basées sur le moment"""
        user_id = moment.user_id
        if user_id not in self.surprise_queue:
            self.surprise_queue[user_id] = []
        surprises = []
        if moment.moment_type == MemoryMomentType.CELEBRATION:
            future_date = datetime.now() + timedelta(days=30)
            surprises.append(
                {
                    "type": "anniversary_reminder",
                    "trigger_date": future_date.isoformat(),
                    "reference_moment": moment.id,
                    "message_template": "celebration_anniversary",
                }
            )
        elif moment.moment_type == MemoryMomentType.CREATIVE_SPARK:
            future_date = datetime.now() + timedelta(days=7)
            surprises.append(
                {
                    "type": "creative_followup",
                    "trigger_date": future_date.isoformat(),
                    "reference_moment": moment.id,
                    "message_template": "creative_check_in",
                }
            )
        elif moment.moment_type == MemoryMomentType.PERSONAL_SHARE:
            future_date = datetime.now() + timedelta(days=3)
            surprises.append(
                {
                    "type": "emotional_checkin",
                    "trigger_date": future_date.isoformat(),
                    "reference_moment": moment.id,
                    "message_template": "gentle_followup",
                }
            )
        self.surprise_queue[user_id].extend(surprises)
        moment.surprise_potential = len(surprises) * 0.25

    async def get_contextual_references(self, user_id: str, current_input: str = "") -> list[dict[str, Any]]:
        """
        🎭 Récupère des références contextuelles pour enrichir la réponse

        Sélectionne intelligemment des moments passés pertinents pour créer
        des micro-interactions émotionnelles et renforcer le lien.
        """
        if user_id not in self.relationship_profiles:
            return []
        user_moments = [m for m in self.memory_moments.values() if m.user_id == user_id]
        if not user_moments:
            return []
        references = []
        if current_input:
            content_refs = await self._find_content_similarities(user_moments, current_input)
            references.extend(content_refs)
        emotional_refs = await self._find_emotional_echoes(user_moments, user_id)
        references.extend(emotional_refs)
        temporal_refs = await self._find_temporal_connections(user_moments)
        references.extend(temporal_refs)
        references.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return references[:3]

    async def _find_content_similarities(self, moments: list[MemoryMoment], current_input: str) -> list[dict[str, Any]]:
        """Trouve des moments similaires par contenu"""
        references = []
        current_words = set(current_input.lower().split())
        for moment in moments:
            moment_words = set((moment.user_input + " " + moment.content).lower().split())
            similarity = len(current_words & moment_words) / len(current_words | moment_words)
            if similarity > 0.3:
                references.append(
                    {
                        "moment_id": moment.id,
                        "type": "content_similarity",
                        "relevance_score": similarity * moment.importance_score,
                        "reference_template": "similar_conversation",
                        "moment": moment,
                    }
                )
        return references

    async def _find_emotional_echoes(self, moments: list[MemoryMoment], user_id: str) -> list[dict[str, Any]]:
        """Trouve des échos émotionnels pertinents"""
        references = []
        profile = self.relationship_profiles[user_id]
        current_emotions = set(profile.dominant_emotions)
        for moment in moments:
            moment_emotions = set(moment.emotions_detected)
            emotional_overlap = len(current_emotions & moment_emotions)
            if emotional_overlap > 0 and moment.emotional_intensity > 0.6:
                relevance = emotional_overlap * moment.emotional_intensity * moment.importance_score
                references.append(
                    {
                        "moment_id": moment.id,
                        "type": "emotional_echo",
                        "relevance_score": relevance,
                        "reference_template": "emotional_callback",
                        "moment": moment,
                    }
                )
        return references

    async def _find_temporal_connections(self, moments: list[MemoryMoment]) -> list[dict[str, Any]]:
        """Trouve des connexions temporelles (anniversaires, patterns)"""
        references = []
        now = datetime.now()
        for moment in moments:
            moment_date = datetime.fromisoformat(moment.timestamp)
            days_ago = (now - moment_date).days
            if days_ago in [7, 30, 90, 365]:
                references.append(
                    {
                        "moment_id": moment.id,
                        "type": "anniversary",
                        "relevance_score": moment.importance_score * 0.8,
                        "reference_template": "anniversary_mention",
                        "days_ago": days_ago,
                        "moment": moment,
                    }
                )
        return references

    async def get_surprise_opportunity(self, user_id: str) -> dict[str, Any] | None:
        """
        🎉 Récupère une opportunité de surprise si appropriée

        Gère la fréquence et le timing des surprises pour créer des moments magiques
        sans devenir intrusif.
        """
        if user_id not in self.surprise_queue or not self.surprise_queue[user_id]:
            return None
        now = datetime.now()
        surprise_frequency = self.config["micro_interactions"]["surprise_frequency"]
        if random.random() > surprise_frequency:
            return None
        for i, surprise in enumerate(self.surprise_queue[user_id]):
            trigger_date = datetime.fromisoformat(surprise["trigger_date"])
            if now >= trigger_date:
                surprise_data = self.surprise_queue[user_id].pop(i)
                moment = self.memory_moments.get(surprise_data["reference_moment"])
                if moment:
                    self.ux_metrics["surprises_delivered"] += 1
                    moment.reference_count += 1
                    moment.last_referenced = now.isoformat()
                    return {
                        "surprise_type": surprise_data["type"],
                        "moment": moment,
                        "template": surprise_data["message_template"],
                        "context": surprise_data,
                    }
        return None

    async def adapt_communication_style(self, user_id: str) -> dict[str, Any]:
        """
        🎨 Adapte le style de communication selon la relation

        Retourne des paramètres de style pour personnaliser les réponses de Jeffrey
        selon le niveau d'intimité et les préférences découvertes.
        """
        if user_id not in self.relationship_profiles:
            return self._get_default_style()
        profile = self.relationship_profiles[user_id]
        base_styles = {
            RelationshipStage.DISCOVERY: {
                "formality": 0.7,
                "playfulness": 0.3,
                "empathy": 0.6,
                "proactivity": 0.2,
                "intimacy": 0.1,
            },
            RelationshipStage.FAMILIARIZATION: {
                "formality": 0.5,
                "playfulness": 0.5,
                "empathy": 0.7,
                "proactivity": 0.4,
                "intimacy": 0.3,
            },
            RelationshipStage.BONDING: {
                "formality": 0.3,
                "playfulness": 0.7,
                "empathy": 0.8,
                "proactivity": 0.6,
                "intimacy": 0.5,
            },
            RelationshipStage.DEEP_CONNECTION: {
                "formality": 0.2,
                "playfulness": 0.8,
                "empathy": 0.9,
                "proactivity": 0.8,
                "intimacy": 0.8,
            },
            RelationshipStage.SOULMATE: {
                "formality": 0.1,
                "playfulness": 0.9,
                "empathy": 1.0,
                "proactivity": 0.9,
                "intimacy": 1.0,
            },
        }
        style = base_styles[profile.relationship_stage].copy()
        if "joie" in profile.dominant_emotions:
            style["playfulness"] = min(1.0, style["playfulness"] + 0.2)
        if "gratitude" in profile.dominant_emotions:
            style["empathy"] = min(1.0, style["empathy"] + 0.1)
        current_time_context = self._get_time_context()
        if "evening" in current_time_context:
            style["intimacy"] = min(1.0, style["intimacy"] + 0.1)
            style["empathy"] = min(1.0, style["empathy"] + 0.1)
        style_context = {
            "nickname": profile.nickname,
            "shared_references": profile.shared_references[-3:],
            "inside_jokes": profile.inside_jokes[-2:],
            "preferred_topics": profile.preferred_topics[-5:],
            "relationship_stage": profile.relationship_stage.value,
            "intimacy_level": profile.intimacy_level,
        }
        return {
            "style_parameters": style,
            "context": style_context,
            "adaptation_metadata": {
                "total_interactions": profile.total_interactions,
                "dominant_emotions": profile.dominant_emotions,
                "last_interaction": profile.last_interaction,
            },
        }

    def _get_default_style(self) -> dict[str, Any]:
        """Style par défaut pour nouveaux utilisateurs"""
        return {
            "style_parameters": {
                "formality": 0.7,
                "playfulness": 0.3,
                "empathy": 0.6,
                "proactivity": 0.2,
                "intimacy": 0.1,
            },
            "context": {
                "nickname": "",
                "shared_references": [],
                "inside_jokes": [],
                "preferred_topics": [],
                "relationship_stage": "discovery",
                "intimacy_level": 0.1,
            },
            "adaptation_metadata": {
                "total_interactions": 0,
                "dominant_emotions": [],
                "last_interaction": "",
            },
        }

    async def save_memories_and_profiles(self, filepath: str = "data/living_memory_data.json"):
        """Sauvegarde persistante des moments et profils"""
        try:
            data = {
                "memory_moments": {mid: asdict(moment) for mid, moment in self.memory_moments.items()},
                "relationship_profiles": {uid: asdict(profile) for uid, profile in self.relationship_profiles.items()},
                "surprise_queue": self.surprise_queue,
                "ux_metrics": self.ux_metrics,
                "last_save": datetime.now().isoformat(),
            }
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(
                f"💾 Living Memory data saved: {len(self.memory_moments)} moments, {len(self.relationship_profiles)} profiles"
            )
        except Exception as e:
            logger.error(f"Erreur sauvegarde Living Memory: {e}")

    async def load_memories_and_profiles(self, filepath: str = "data/living_memory_data.json"):
        """Chargement des données persistantes"""
        try:
            if not Path(filepath).exists():
                logger.info("Fichier Living Memory inexistant, démarrage à vide")
                return
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)
            for mid, moment_data in data.get("memory_moments", {}).items():
                moment = MemoryMoment(**moment_data, source="human")
                moment.moment_type = MemoryMomentType(moment_data["moment_type"])
                self.memory_moments[mid] = moment
            for uid, profile_data in data.get("relationship_profiles", {}).items():
                profile = RelationshipProfile(**profile_data)
                profile.relationship_stage = RelationshipStage(profile_data["relationship_stage"])
                self.relationship_profiles[uid] = profile
            self.surprise_queue = data.get("surprise_queue", {})
            self.ux_metrics = data.get("ux_metrics", self.ux_metrics)
            logger.info(
                f"💾 Living Memory data loaded: {len(self.memory_moments)} moments, {len(self.relationship_profiles)} profiles"
            )
        except Exception as e:
            logger.error(f"Erreur chargement Living Memory: {e}")

    def get_ux_metrics(self) -> dict[str, Any]:
        """Retourne les métriques UX pour monitoring"""
        return {
            **self.ux_metrics,
            "total_memory_moments": len(self.memory_moments),
            "total_relationships": len(self.relationship_profiles),
            "active_surprise_queues": len([q for q in self.surprise_queue.values() if q]),
            "relationship_stages": {
                stage.value: len([p for p in self.relationship_profiles.values() if p.relationship_stage == stage])
                for stage in RelationshipStage
            },
        }


def create_living_memory() -> LivingMemoryCore:
    """Factory pour créer une instance du système Living Memory"""
    return LivingMemoryCore()
