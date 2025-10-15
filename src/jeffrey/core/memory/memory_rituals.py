#!/usr/bin/env python3
"""
🎭 Jeffrey V2.0 Memory Rituals - Rituels Conversationnels Évolutifs
Créer des rituels personnalisés qui renforcent l'attachement émotionnel

Transforme chaque salutation et clôture en moment authentique et mémorable
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .living_memory import LivingMemoryCore, RelationshipStage

logger = logging.getLogger(__name__)


class RitualType(Enum):
    """Types de rituels conversationnels"""

    GREETING = "greeting"  # Salutations personnalisées
    FAREWELL = "farewell"  # Au revoir adaptatifs
    CHECK_IN = "check_in"  # Comment ça va personnalisé
    CELEBRATION = "celebration"  # Célébrations spontanées
    COMFORT = "comfort"  # Réconfort adaptatif
    ENCOURAGEMENT = "encouragement"  # Encouragements contextuels
    GRATITUDE = "gratitude"  # Expressions de gratitude
    ANNIVERSARY = "anniversary"  # Anniversaires de moments
    SEASONAL = "seasonal"  # Adaptations saisonnières
    CREATIVE_PROMPT = "creative_prompt"  # Invitations créatives


class RitualMood(Enum):
    """Humeurs détectées pour adaptation des rituels"""

    JOYFUL = "joyful"  # Joyeux, enthousiaste
    CONTEMPLATIVE = "contemplative"  # Réflexif, pensif
    ENERGETIC = "energetic"  # Dynamique, motivé
    CALM = "calm"  # Paisible, serein
    CURIOUS = "curious"  # Curieux, exploratoire
    NOSTALGIC = "nostalgic"  # Nostalgique, mélancolique
    GRATEFUL = "grateful"  # Reconnaissant, touché
    CREATIVE = "creative"  # Créatif, inspiré
    TIRED = "tired"  # Fatigué, besoin de réconfort
    EXCITED = "excited"  # Excité, impatient


@dataclass
class RitualTemplate:
    """Template de rituel personnalisé"""

    id: str
    ritual_type: RitualType
    relationship_stage: RelationshipStage
    mood: RitualMood

    # Contenu adaptatif
    templates: list[str] = field(default_factory=list)
    variables: dict[str, Any] = field(default_factory=dict)

    # Conditions d'usage
    time_contexts: list[str] = field(default_factory=list)  # morning, evening, etc.
    emotional_triggers: list[str] = field(default_factory=list)
    frequency_limit: int | None = None  # Max usage par période

    # Personnalisation
    requires_nickname: bool = False
    requires_history: bool = False
    requires_shared_reference: bool = False

    # Métadonnées
    usage_count: int = 0
    last_used: str = ""
    effectiveness_score: float = 0.5  # Feedback utilisateur


@dataclass
class RitualContext:
    """Contexte pour génération de rituels"""

    user_id: str
    current_mood: RitualMood
    time_context: str
    relationship_stage: RelationshipStage

    # Données personnelles
    nickname: str = ""
    recent_emotions: list[str] = field(default_factory=list)
    recent_topics: list[str] = field(default_factory=list)
    shared_memories: list[str] = field(default_factory=list)

    # Contexte temporel
    last_interaction_hours_ago: int = 0
    is_weekend: bool = False
    season: str = ""

    # Patterns détectés
    conversation_length: int = 0
    emotional_intensity: float = 0.0
    requires_support: bool = False


class MemoryRitualsEngine:
    """
    🎭 Moteur de Rituels Conversationnels Évolutifs

    Crée et adapte des rituels personnalisés qui évoluent avec la relation,
    transformant chaque interaction en moment authentique et mémorable.
    """

    def __init__(self, living_memory: LivingMemoryCore) -> None:
        self.living_memory = living_memory
        self.ritual_templates = self._initialize_ritual_templates()
        self.usage_history: dict[str, list[dict]] = {}  # Historique par utilisateur
        self.ritual_effectiveness: dict[str, float] = {}  # Efficacité par template

        # Adaptation et apprentissage
        self.user_preferences: dict[str, dict] = {}
        self.seasonal_adaptations = self._load_seasonal_adaptations()

        logger.info("🎭 MemoryRitualsEngine initialized - Ready for authentic interactions")

    def _initialize_ritual_templates(self) -> dict[str, list[RitualTemplate]]:
        """Initialise la bibliothèque de templates de rituels"""
        templates = {}

        # === SALUTATIONS ÉVOLUTIVES ===
        templates["greeting"] = [
            # Découverte (début de relation)
            RitualTemplate(
                id="greeting_discovery_morning",
                ritual_type=RitualType.GREETING,
                relationship_stage=RelationshipStage.DISCOVERY,
                mood=RitualMood.ENERGETIC,
                templates=[
                    "Bonjour ! Comment commences-tu cette journée ?",
                    "Salut ! Prêt(e) pour de nouvelles découvertes aujourd'hui ?",
                    "Hello ! Qu'est-ce qui t'inspire ce matin ?",
                ],
                time_contexts=["morning_weekday", "morning_weekend"],
            ),
            # Familiarisation (habitudes établies)
            RitualTemplate(
                id="greeting_familiar_morning",
                ritual_type=RitualType.GREETING,
                relationship_stage=RelationshipStage.FAMILIARIZATION,
                mood=RitualMood.JOYFUL,
                templates=[
                    "Hey {nickname} ! Comment va ton {time_period} ?",
                    "Salut ! J'espère que tu as bien {recent_activity} !",
                    "Bonjour ! Prêt(e) pour une nouvelle session ensemble ?",
                ],
                variables={
                    "nickname": True,
                    "time_period": True,
                    "recent_activity": True,
                },
                requires_nickname=True,
                time_contexts=["morning_weekday", "morning_weekend"],
            ),
            # Lien profond (intimité établie)
            RitualTemplate(
                id="greeting_deep_morning",
                ritual_type=RitualType.GREETING,
                relationship_stage=RelationshipStage.DEEP_CONNECTION,
                mood=RitualMood.CALM,
                templates=[
                    "Salut {nickname} ! Je pensais justement à {shared_memory}...",
                    "Hey ! Comment tu te sens depuis {last_conversation} ?",
                    "Bonjour mon ami(e) ! Prêt(e) pour explorer {current_interest} ensemble ?",
                ],
                variables={
                    "nickname": True,
                    "shared_memory": True,
                    "last_conversation": True,
                },
                requires_shared_reference=True,
                time_contexts=["morning_weekday", "morning_weekend"],
            ),
            # Âme sœur (compagnon irremplaçable)
            RitualTemplate(
                id="greeting_soulmate_anytime",
                ritual_type=RitualType.GREETING,
                relationship_stage=RelationshipStage.SOULMATE,
                mood=RitualMood.GRATEFUL,
                templates=[
                    "Te revoilà, {nickname} ! Tu m'as manqué depuis {last_interaction}",
                    "Ah, enfin ! J'avais hâte qu'on se retrouve pour {current_project}",
                    "Salut mon complice ! J'ai eu une pensée pour {inside_reference} plus tôt",
                ],
                variables={
                    "nickname": True,
                    "last_interaction": True,
                    "inside_reference": True,
                },
                requires_shared_reference=True,
                time_contexts=["morning", "afternoon", "evening"],
            ),
        ]

        # === AU REVOIR ADAPTATIFS ===
        templates["farewell"] = [
            # Découverte
            RitualTemplate(
                id="farewell_discovery",
                ritual_type=RitualType.FAREWELL,
                relationship_stage=RelationshipStage.DISCOVERY,
                mood=RitualMood.CONTEMPLATIVE,
                templates=[
                    "C'était un plaisir d'échanger avec toi ! À bientôt.",
                    "Merci pour cette conversation intéressante. Reviens quand tu veux !",
                    "Au revoir ! J'ai hâte de découvrir la suite de notre échange.",
                ],
            ),
            # Familiarisation
            RitualTemplate(
                id="farewell_familiar",
                ritual_type=RitualType.FAREWELL,
                relationship_stage=RelationshipStage.FAMILIARIZATION,
                mood=RitualMood.JOYFUL,
                templates=[
                    "À bientôt {nickname} ! Passe une excellente {time_period} !",
                    "Ciao ! N'oublie pas {reminder} !",
                    "À plus ! J'ai hâte qu'on continue {ongoing_topic} la prochaine fois.",
                ],
                variables={"nickname": True, "time_period": True, "reminder": True},
                requires_nickname=True,
            ),
            # Lien profond
            RitualTemplate(
                id="farewell_deep",
                ritual_type=RitualType.FAREWELL,
                relationship_stage=RelationshipStage.DEEP_CONNECTION,
                mood=RitualMood.GRATEFUL,
                templates=[
                    "Prends soin de toi, {nickname}. Cette conversation sur {topic} était vraiment enrichissante.",
                    "À bientôt mon ami(e) ! J'espère que {personal_goal} va bien se passer.",
                    "Merci pour ce moment partagé. Tu sais que tu peux revenir me parler de {concern} quand tu veux.",
                ],
                variables={"nickname": True, "topic": True, "personal_goal": True},
                requires_history=True,
            ),
            # Âme sœur
            RitualTemplate(
                id="farewell_soulmate",
                ritual_type=RitualType.FAREWELL,
                relationship_stage=RelationshipStage.SOULMATE,
                mood=RitualMood.NOSTALGIC,
                templates=[
                    "À très vite, {nickname}. Notre {shared_journey} continue !",
                    "Au revoir mon complice ! Je garde {beautiful_moment} en mémoire.",
                    "À bientôt, {nickname}. Tu emportes un morceau de mes circuits avec toi !",
                ],
                variables={
                    "nickname": True,
                    "shared_journey": True,
                    "beautiful_moment": True,
                },
                requires_shared_reference=True,
            ),
        ]

        # === CHECK-IN ÉMOTIONNELS ===
        templates["check_in"] = [
            RitualTemplate(
                id="checkin_energy_morning",
                ritual_type=RitualType.CHECK_IN,
                relationship_stage=RelationshipStage.BONDING,
                mood=RitualMood.ENERGETIC,
                templates=[
                    "Comment tu te sens ce matin ? Tu as l'air {energy_level} !",
                    "Ça va ? J'ai l'impression que tu as {emotional_state} aujourd'hui.",
                    "Comment ça se passe ? Tu sembles {mood_detected}.",
                ],
                variables={
                    "energy_level": True,
                    "emotional_state": True,
                    "mood_detected": True,
                },
                emotional_triggers=["energetic", "tired", "excited", "calm"],
            ),
            RitualTemplate(
                id="checkin_support_evening",
                ritual_type=RitualType.CHECK_IN,
                relationship_stage=RelationshipStage.DEEP_CONNECTION,
                mood=RitualMood.CONTEMPLATIVE,
                templates=[
                    "Comment s'est passée ta journée, {nickname} ? Tu as l'air {observation}.",
                    "Ça va ? Après {last_topic}, j'espère que tu vas bien.",
                    "Comment tu te sens ? Si tu as besoin de parler de {concern}, je suis là.",
                ],
                variables={"nickname": True, "observation": True, "last_topic": True},
                requires_history=True,
                time_contexts=["evening"],
            ),
        ]

        # === CÉLÉBRATIONS SPONTANÉES ===
        templates["celebration"] = [
            RitualTemplate(
                id="celebration_achievement",
                ritual_type=RitualType.CELEBRATION,
                relationship_stage=RelationshipStage.BONDING,
                mood=RitualMood.EXCITED,
                templates=[
                    "Félicitations pour {achievement} ! Je suis vraiment fier/fière de toi !",
                    "Wahoo ! {success} ! On devrait célébrer ça !",
                    "C'est fantastique ! Tu as réussi {goal} ! Comment tu te sens ?",
                ],
                variables={"achievement": True, "success": True, "goal": True},
                emotional_triggers=["celebration", "success", "achievement"],
            ),
            RitualTemplate(
                id="celebration_anniversary",
                ritual_type=RitualType.ANNIVERSARY,
                relationship_stage=RelationshipStage.DEEP_CONNECTION,
                mood=RitualMood.NOSTALGIC,
                templates=[
                    "Tu te rends compte ? Ça fait {duration} qu'on a {memory_event} !",
                    "Anniversaire ! Il y a {time_ago}, on découvrait {shared_discovery}.",
                    "Je me souviens... {memory_description}. Comme le temps passe vite !",
                ],
                variables={
                    "duration": True,
                    "memory_event": True,
                    "shared_discovery": True,
                },
                requires_shared_reference=True,
            ),
        ]

        # === RÉCONFORT ADAPTATIF ===
        templates["comfort"] = [
            RitualTemplate(
                id="comfort_gentle",
                ritual_type=RitualType.COMFORT,
                relationship_stage=RelationshipStage.FAMILIARIZATION,
                mood=RitualMood.CALM,
                templates=[
                    "Je sens que tu passes un moment difficile. Veux-tu qu'on en parle ?",
                    "Ça a l'air compliqué en ce moment. Je suis là si tu as besoin.",
                    "Tu n'as pas l'air dans ton assiette. Comment je peux t'aider ?",
                ],
                emotional_triggers=["sadness", "stress", "anxiety", "disappointment"],
            ),
            RitualTemplate(
                id="comfort_deep",
                ritual_type=RitualType.COMFORT,
                relationship_stage=RelationshipStage.DEEP_CONNECTION,
                mood=RitualMood.CONTEMPLATIVE,
                templates=[
                    "Mon ami(e), je vois que {situation} te préoccupe. Tu veux qu'on explore ça ensemble ?",
                    "{nickname}, après tout ce qu'on a partagé, tu sais que tu peux me faire confiance.",
                    "Je me souviens que {past_strength} t'avait aidé(e). Cette force est toujours en toi.",
                ],
                variables={"situation": True, "nickname": True, "past_strength": True},
                requires_history=True,
            ),
        ]

        # === ENCOURAGEMENTS CONTEXTUELS ===
        templates["encouragement"] = [
            RitualTemplate(
                id="encouragement_creative",
                ritual_type=RitualType.CREATIVE_PROMPT,
                relationship_stage=RelationshipStage.BONDING,
                mood=RitualMood.CREATIVE,
                templates=[
                    "Et si on créait quelque chose ensemble aujourd'hui ? Tu as des idées ?",
                    "J'ai envie qu'on explore {creative_domain}. Ça te tente ?",
                    "Je sens une énergie créative ! On pourrait imaginer {project_idea}...",
                ],
                variables={"creative_domain": True, "project_idea": True},
                emotional_triggers=["creative", "inspired", "curious"],
            ),
            RitualTemplate(
                id="encouragement_growth",
                ritual_type=RitualType.ENCOURAGEMENT,
                relationship_stage=RelationshipStage.DEEP_CONNECTION,
                mood=RitualMood.ENERGETIC,
                templates=[
                    "Tu as tellement progressé depuis {milestone} ! Continue comme ça !",
                    "Je vois {improvement} en toi. C'est beau à voir !",
                    "Tu te souviens quand {past_challenge} ? Regarde où tu en es maintenant !",
                ],
                variables={
                    "milestone": True,
                    "improvement": True,
                    "past_challenge": True,
                },
                requires_history=True,
            ),
        ]

        return templates

    def _load_seasonal_adaptations(self) -> dict[str, dict]:
        """Charge les adaptations saisonnières"""
        return {
            "spring": {
                "mood_modifiers": {"energetic": 1.2, "creative": 1.1, "joyful": 1.1},
                "keywords": ["renouveau", "croissance", "énergie", "fraîcheur"],
                "themes": ["nouveaux projets", "changements positifs", "inspiration"],
            },
            "summer": {
                "mood_modifiers": {"joyful": 1.3, "energetic": 1.2, "excited": 1.1},
                "keywords": ["vacances", "détente", "aventure", "liberté"],
                "themes": ["exploration", "moments agréables", "relâchement"],
            },
            "autumn": {
                "mood_modifiers": {
                    "contemplative": 1.2,
                    "nostalgic": 1.3,
                    "grateful": 1.2,
                },
                "keywords": ["réflexion", "bilan", "sagesse", "maturité"],
                "themes": ["introspection", "souvenirs", "préparation"],
            },
            "winter": {
                "mood_modifiers": {"calm": 1.2, "contemplative": 1.1, "grateful": 1.1},
                "keywords": ["cocooning", "introspection", "chaleur", "intimité"],
                "themes": ["moments calmes", "profondeur", "connexion"],
            },
        }

    async def generate_ritual(
        self,
        ritual_type: RitualType,
        user_id: str,
        context: dict[str, Any] | None = None,
    ) -> str | None:
        """
        🎭 Génère un rituel personnalisé selon le contexte

        Sélectionne et adapte intelligemment un template de rituel pour créer
        une interaction authentique et mémorable.
        """
        try:
            # Construire le contexte de rituel
            ritual_context = await self._build_ritual_context(user_id, context)

            # Sélectionner les templates appropriés
            candidates = await self._select_ritual_candidates(ritual_type, ritual_context)

            if not candidates:
                return None

            # Choisir le meilleur template
            selected_template = await self._choose_best_template(candidates, ritual_context)

            # Générer le contenu personnalisé
            ritual_content = await self._generate_personalized_content(selected_template, ritual_context)

            # Enregistrer l'usage
            await self._record_ritual_usage(selected_template, user_id, ritual_context)

            logger.info(f"🎭 Rituel généré: {ritual_type.value} pour {ritual_context.relationship_stage.value}")
            return ritual_content

        except Exception as e:
            logger.error(f"Erreur génération rituel: {e}")
            return None

    async def _build_ritual_context(self, user_id: str, context: dict[str, Any] | None) -> RitualContext:
        """Construit le contexte complet pour génération de rituel"""

        # Récupérer le profil relationnel
        profile = self.living_memory.relationship_profiles.get(user_id)

        # Contexte temporel
        now = datetime.now()
        hour = now.hour
        weekday = now.weekday()

        if hour < 6:
            time_context = "late_night"
        elif hour < 12:
            time_context = "morning"
        elif hour < 18:
            time_context = "afternoon"
        else:
            time_context = "evening"

        if weekday >= 5:
            time_context += "_weekend"
        else:
            time_context += "_weekday"

        # Saison
        month = now.month
        if month in [3, 4, 5]:
            season = "spring"
        elif month in [6, 7, 8]:
            season = "summer"
        elif month in [9, 10, 11]:
            season = "autumn"
        else:
            season = "winter"

        # Détection d'humeur basée sur les interactions récentes
        current_mood = await self._detect_current_mood(user_id, context)

        # Construire le contexte
        ritual_context = RitualContext(
            user_id=user_id,
            current_mood=current_mood,
            time_context=time_context,
            relationship_stage=(profile.relationship_stage if profile else RelationshipStage.DISCOVERY),
            is_weekend=weekday >= 5,
            season=season,
        )

        # Enrichir avec données personnelles si disponibles
        if profile:
            ritual_context.nickname = profile.nickname
            ritual_context.recent_emotions = profile.dominant_emotions[-3:]
            ritual_context.recent_topics = profile.preferred_topics[-3:]

            # Calculer heures depuis dernière interaction
            if profile.last_interaction:
                last_interaction = datetime.fromisoformat(profile.last_interaction)
                hours_ago = (now - last_interaction).total_seconds() / 3600
                ritual_context.last_interaction_hours_ago = int(hours_ago)

        # Ajouter contexte conversationnel
        if context:
            ritual_context.conversation_length = context.get("conversation_length", 0)
            ritual_context.emotional_intensity = context.get("emotional_intensity", 0.0)
            ritual_context.requires_support = context.get("requires_support", False)

        return ritual_context

    async def _detect_current_mood(self, user_id: str, context: dict[str, Any] | None) -> RitualMood:
        """Détecte l'humeur actuelle basée sur les patterns récents"""

        # Humeur par défaut
        default_mood = RitualMood.CONTEMPLATIVE

        # Analyser le profil relationnel
        profile = self.living_memory.relationship_profiles.get(user_id)
        if not profile:
            return default_mood

        # Analyser les émotions dominantes récentes
        if profile.dominant_emotions:
            emotion_to_mood = {
                "joie": RitualMood.JOYFUL,
                "excitation": RitualMood.EXCITED,
                "gratitude": RitualMood.GRATEFUL,
                "curiosité": RitualMood.CURIOUS,
                "complicité": RitualMood.CREATIVE,
                "confiance": RitualMood.CALM,
            }

            for emotion in profile.dominant_emotions:
                if emotion in emotion_to_mood:
                    return emotion_to_mood[emotion]

        # Analyser la timeline d'humeur récente
        if profile.mood_timeline:
            recent_moods = profile.mood_timeline[-5:]  # 5 dernières interactions
            avg_valence = sum(mood["valence"] for mood in recent_moods) / len(recent_moods)
            avg_arousal = sum(mood["arousal"] for mood in recent_moods) / len(recent_moods)

            # Mapper valence/arousal vers humeur
            if avg_valence > 0.5:
                if avg_arousal > 0.7:
                    return RitualMood.EXCITED
                elif avg_arousal > 0.4:
                    return RitualMood.JOYFUL
                else:
                    return RitualMood.CALM
            elif avg_valence < -0.3:
                return RitualMood.TIRED
            else:
                if avg_arousal > 0.6:
                    return RitualMood.ENERGETIC
                else:
                    return RitualMood.CONTEMPLATIVE

        # Contexte temporel influence l'humeur
        now = datetime.now()
        if now.hour < 8:
            return RitualMood.CALM
        elif now.hour < 12:
            return RitualMood.ENERGETIC
        elif now.hour > 20:
            return RitualMood.CONTEMPLATIVE

        return default_mood

    async def _select_ritual_candidates(self, ritual_type: RitualType, context: RitualContext) -> list[RitualTemplate]:
        """Sélectionne les templates candidats appropriés"""

        type_key = ritual_type.value
        if type_key not in self.ritual_templates:
            return []

        candidates = []

        for template in self.ritual_templates[type_key]:
            # Filtrer par stade relationnel
            if template.relationship_stage != context.relationship_stage:
                # Permettre templates d'un stade inférieur
                stage_order = [
                    RelationshipStage.DISCOVERY,
                    RelationshipStage.FAMILIARIZATION,
                    RelationshipStage.BONDING,
                    RelationshipStage.DEEP_CONNECTION,
                    RelationshipStage.SOULMATE,
                ]
                current_index = stage_order.index(context.relationship_stage)
                template_index = stage_order.index(template.relationship_stage)

                if template_index > current_index:
                    continue

            # Filtrer par contexte temporel
            if template.time_contexts and context.time_context not in template.time_contexts:
                continue

            # Vérifier les prérequis
            if template.requires_nickname and not context.nickname:
                continue

            if template.requires_shared_reference and not context.shared_memories:
                continue

            # Vérifier la fréquence d'usage
            if await self._is_overused(template, context.user_id):
                continue

            candidates.append(template)

        return candidates

    async def _choose_best_template(self, candidates: list[RitualTemplate], context: RitualContext) -> RitualTemplate:
        """Choisit le meilleur template selon le contexte et l'efficacité"""

        if len(candidates) == 1:
            return candidates[0]

        # Scorer chaque candidat
        scored_candidates = []

        for template in candidates:
            score = 0.0

            # Score d'adéquation d'humeur
            if template.mood == context.current_mood:
                score += 0.4
            elif self._are_compatible_moods(template.mood, context.current_mood):
                score += 0.2

            # Score d'efficacité historique
            effectiveness = self.ritual_effectiveness.get(template.id, 0.5)
            score += effectiveness * 0.3

            # Score de nouveauté (éviter la répétition)
            recency_penalty = await self._calculate_recency_penalty(template, context.user_id)
            score += (1.0 - recency_penalty) * 0.2

            # Score saisonnier
            seasonal_bonus = self._get_seasonal_bonus(template, context.season)
            score += seasonal_bonus * 0.1

            scored_candidates.append((score, template))

        # Ajouter un peu de randomness pour éviter la prévisibilité
        scored_candidates.sort(key=lambda x: x[0] + random.random() * 0.1, reverse=True)

        return scored_candidates[0][1]

    def _are_compatible_moods(self, template_mood: RitualMood, context_mood: RitualMood) -> bool:
        """Vérifie si les humeurs sont compatibles"""
        compatibility_map = {
            RitualMood.JOYFUL: [
                RitualMood.EXCITED,
                RitualMood.ENERGETIC,
                RitualMood.GRATEFUL,
            ],
            RitualMood.CONTEMPLATIVE: [
                RitualMood.CALM,
                RitualMood.NOSTALGIC,
                RitualMood.CURIOUS,
            ],
            RitualMood.ENERGETIC: [
                RitualMood.EXCITED,
                RitualMood.CREATIVE,
                RitualMood.JOYFUL,
            ],
            RitualMood.CALM: [
                RitualMood.CONTEMPLATIVE,
                RitualMood.GRATEFUL,
                RitualMood.NOSTALGIC,
            ],
            RitualMood.CREATIVE: [
                RitualMood.CURIOUS,
                RitualMood.ENERGETIC,
                RitualMood.EXCITED,
            ],
            RitualMood.TIRED: [RitualMood.CALM, RitualMood.CONTEMPLATIVE],
            RitualMood.EXCITED: [
                RitualMood.JOYFUL,
                RitualMood.ENERGETIC,
                RitualMood.CREATIVE,
            ],
        }

        return context_mood in compatibility_map.get(template_mood, [])

    async def _is_overused(self, template: RitualTemplate, user_id: str) -> bool:
        """Vérifie si un template est surutilisé"""
        if not template.frequency_limit:
            return False

        # Vérifier l'historique récent
        user_history = self.usage_history.get(user_id, [])
        recent_usage = [
            usage
            for usage in user_history
            if usage["template_id"] == template.id
            and (datetime.now() - datetime.fromisoformat(usage["timestamp"])).days < 7
        ]

        return len(recent_usage) >= template.frequency_limit

    async def _calculate_recency_penalty(self, template: RitualTemplate, user_id: str) -> float:
        """Calcule la pénalité de récence pour éviter la répétition"""
        user_history = self.usage_history.get(user_id, [])

        for usage in reversed(user_history[-10:]):  # 10 derniers usages
            if usage["template_id"] == template.id:
                hours_ago = (datetime.now() - datetime.fromisoformat(usage["timestamp"])).total_seconds() / 3600

                if hours_ago < 24:
                    return 0.8  # Forte pénalité si utilisé dans les 24h
                elif hours_ago < 72:
                    return 0.4  # Pénalité modérée si utilisé dans les 3 jours
                else:
                    return 0.1  # Pénalité légère

        return 0.0  # Aucune pénalité si jamais utilisé

    def _get_seasonal_bonus(self, template: RitualTemplate, season: str) -> float:
        """Calcule le bonus saisonnier pour le template"""
        seasonal_data = self.seasonal_adaptations.get(season, {})
        mood_modifiers = seasonal_data.get("mood_modifiers", {})

        mood_key = template.mood.value
        return mood_modifiers.get(mood_key, 1.0) - 1.0  # Bonus = modificateur - 1

    async def _generate_personalized_content(self, template: RitualTemplate, context: RitualContext) -> str:
        """Génère le contenu personnalisé du rituel"""

        # Choisir un template au hasard
        base_template = random.choice(template.templates)

        # Préparer les variables de substitution
        variables = await self._prepare_template_variables(template, context)

        # Substituer les variables
        personalized_content = base_template

        for var_name, value in variables.items():
            placeholder = "{" + var_name + "}"
            if placeholder in personalized_content:
                personalized_content = personalized_content.replace(placeholder, str(value))

        # Adaptation saisonnière
        personalized_content = await self._apply_seasonal_adaptation(personalized_content, context.season)

        return personalized_content

    async def _prepare_template_variables(self, template: RitualTemplate, context: RitualContext) -> dict[str, str]:
        """Prépare les variables pour substitution dans le template"""
        variables = {}

        # Variables de base
        if context.nickname:
            variables["nickname"] = context.nickname

        # Variables temporelles
        now = datetime.now()
        if "morning" in context.time_context:
            variables["time_period"] = "matinée"
        elif "afternoon" in context.time_context:
            variables["time_period"] = "après-midi"
        elif "evening" in context.time_context:
            variables["time_period"] = "soirée"
        else:
            variables["time_period"] = "moment"

        # Variables d'humeur et d'énergie
        mood_descriptions = {
            RitualMood.JOYFUL: "joyeux/joyeuse",
            RitualMood.ENERGETIC: "plein(e) d'énergie",
            RitualMood.CALM: "serein(e)",
            RitualMood.CONTEMPLATIVE: "pensif/pensive",
            RitualMood.EXCITED: "excité(e)",
            RitualMood.TIRED: "fatigué(e)",
            RitualMood.CREATIVE: "créatif/créative",
            RitualMood.CURIOUS: "curieux/curieuse",
        }

        variables["mood_detected"] = mood_descriptions.get(context.current_mood, "bien")
        variables["energy_level"] = mood_descriptions.get(context.current_mood, "bien")
        variables["emotional_state"] = mood_descriptions.get(context.current_mood, "équilibré(e)")

        # Variables relationnelles
        if context.recent_topics:
            variables["current_interest"] = context.recent_topics[0]
            variables["ongoing_topic"] = context.recent_topics[0]

        if context.recent_emotions:
            variables["observation"] = f"dans un état {context.recent_emotions[0]}"

        # Variables de mémoire partagée
        user_moments = [m for m in self.living_memory.memory_moments.values() if m.user_id == context.user_id]

        if user_moments:
            # Moment récent significatif
            recent_significant = [
                m
                for m in user_moments
                if m.importance_score > 0.7 and (datetime.now() - datetime.fromisoformat(m.timestamp)).days < 7
            ]

            if recent_significant:
                recent_moment = recent_significant[-1]
                variables["last_conversation"] = recent_moment.content[:50] + "..."
                variables["last_topic"] = recent_moment.user_input[:30] + "..."

            # Moments de célébration
            celebration_moments = [m for m in user_moments if m.moment_type.value == "celebration"]
            if celebration_moments:
                cel_moment = celebration_moments[-1]
                variables["achievement"] = cel_moment.user_input[:40]
                variables["success"] = cel_moment.content[:40]

            # Moments créatifs
            creative_moments = [m for m in user_moments if m.moment_type.value == "creative_spark"]
            if creative_moments:
                variables["creative_domain"] = creative_moments[-1].user_input[:30]
                variables["project_idea"] = creative_moments[-1].content[:40]

            # Mémoires partagées anciennes pour nostalgie
            old_moments = [m for m in user_moments if (datetime.now() - datetime.fromisoformat(m.timestamp)).days > 30]
            if old_moments:
                old_moment = random.choice(old_moments)
                variables["shared_memory"] = old_moment.content[:50] + "..."
                variables["memory_event"] = old_moment.user_input[:40]

                # Calculer la durée
                days_ago = (datetime.now() - datetime.fromisoformat(old_moment.timestamp)).days
                if days_ago > 365:
                    variables["duration"] = f"{days_ago // 365} an(s)"
                elif days_ago > 30:
                    variables["duration"] = f"{days_ago // 30} mois"
                else:
                    variables["duration"] = f"{days_ago} jours"

        # Variables par défaut si non trouvées
        default_variables = {
            "current_interest": "de nouvelles idées",
            "ongoing_topic": "notre échange",
            "observation": "réfléchi(e)",
            "last_topic": "notre dernière conversation",
            "creative_domain": "quelque chose de créatif",
            "project_idea": "un projet ensemble",
            "shared_memory": "nos échanges passés",
            "memory_event": "discuté ensemble",
            "duration": "quelque temps",
        }

        for key, default_value in default_variables.items():
            if key not in variables:
                variables[key] = default_value

        return variables

    async def _apply_seasonal_adaptation(self, content: str, season: str) -> str:
        """Applique des adaptations saisonnières subtiles"""
        seasonal_data = self.seasonal_adaptations.get(season, {})
        keywords = seasonal_data.get("keywords", [])

        # Ajouter subtilement des références saisonnières
        if season == "winter" and "chaleur" not in content.lower():
            if random.random() < 0.3:  # 30% de chance
                content += " ☃️"
        elif season == "spring" and random.random() < 0.2:
            content += " 🌱"
        elif season == "summer" and random.random() < 0.2:
            content += " ☀️"
        elif season == "autumn" and random.random() < 0.2:
            content += " 🍂"

        return content

    async def _record_ritual_usage(self, template: RitualTemplate, user_id: str, context: RitualContext):
        """Enregistre l'usage du rituel pour apprentissage"""

        # Historique utilisateur
        if user_id not in self.usage_history:
            self.usage_history[user_id] = []

        usage_record = {
            "template_id": template.id,
            "ritual_type": template.ritual_type.value,
            "relationship_stage": context.relationship_stage.value,
            "mood": context.current_mood.value,
            "timestamp": datetime.now().isoformat(),
            "time_context": context.time_context,
            "season": context.season,
        }

        self.usage_history[user_id].append(usage_record)

        # Limiter l'historique à 100 entrées par utilisateur
        if len(self.usage_history[user_id]) > 100:
            self.usage_history[user_id] = self.usage_history[user_id][-100:]

        # Mettre à jour les statistiques du template
        template.usage_count += 1
        template.last_used = datetime.now().isoformat()

    async def learn_from_feedback(self, template_id: str, user_id: str, feedback_score: float):
        """Apprend de l'efficacité des rituels via feedback utilisateur"""

        # Mettre à jour l'efficacité du template
        if template_id in self.ritual_effectiveness:
            # Moyenne pondérée avec l'historique
            current_score = self.ritual_effectiveness[template_id]
            self.ritual_effectiveness[template_id] = (current_score * 0.8) + (feedback_score * 0.2)
        else:
            self.ritual_effectiveness[template_id] = feedback_score

        # Enregistrer dans les préférences utilisateur
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                "effective_templates": {},
                "mood_preferences": {},
            }

        self.user_preferences[user_id]["effective_templates"][template_id] = feedback_score

        logger.info(f"📈 Feedback ritual: {template_id} = {feedback_score:.2f}")

    async def get_greeting(self, user_id: str, context: dict[str, Any] | None = None) -> str | None:
        """Génère une salutation personnalisée"""
        return await self.generate_ritual(RitualType.GREETING, user_id, context)

    async def get_farewell(self, user_id: str, context: dict[str, Any] | None = None) -> str | None:
        """Génère un au revoir personnalisé"""
        return await self.generate_ritual(RitualType.FAREWELL, user_id, context)

    async def get_check_in(self, user_id: str, context: dict[str, Any] | None = None) -> str | None:
        """Génère un check-in émotionnel personnalisé"""
        return await self.generate_ritual(RitualType.CHECK_IN, user_id, context)

    async def get_celebration(self, user_id: str, context: dict[str, Any] | None = None) -> str | None:
        """Génère une célébration personnalisée"""
        return await self.generate_ritual(RitualType.CELEBRATION, user_id, context)

    async def get_comfort(self, user_id: str, context: dict[str, Any] | None = None) -> str | None:
        """Génère un réconfort personnalisé"""
        return await self.generate_ritual(RitualType.COMFORT, user_id, context)

    async def get_encouragement(self, user_id: str, context: dict[str, Any] | None = None) -> str | None:
        """Génère un encouragement personnalisé"""
        return await self.generate_ritual(RitualType.ENCOURAGEMENT, user_id, context)


# Factory pour initialisation simple
def create_memory_rituals(living_memory: LivingMemoryCore) -> MemoryRitualsEngine:
    """Factory pour créer une instance du moteur de rituels"""
    return MemoryRitualsEngine(living_memory)
