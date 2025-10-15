#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Module SilenceSenseAnalyzer pour la détection et l'interprétation des silences significatifs.
Permet d'analyser les silences dans la conversation et d'en dériver des intentions potentielles
pour adapter le comportement de Jeffrey.
"""

import logging
import datetime
from typing import Dict, Any, List, Optional, NamedTuple
from enum import Enum, auto

logger = logging.getLogger(__name__)


# Classes d'énumération pour les types de silence et intentions
class SilenceType(Enum):
    """Types de silence identifiés."""

    PAUSE_COURTE = auto()  # 1-3 secondes
    PAUSE_MOYENNE = auto()  # 3-7 secondes
    PAUSE_LONGUE = auto()  # 7-15 secondes
    ABSENCE_REPONSE = auto()  # >15 secondes
    HESITATION = auto()  # Pauses courtes répétées

class SilenceIntention(Enum):
    """Intentions potentielles dérivées des silences."""

    REFLEXION = auto()  # Utilisateur réfléchit
    ATTENTE = auto()  # Utilisateur attend une action de Jeffrey
    FATIGUE = auto()  # Utilisateur fatigué
    DISTRACTION = auto()  # Utilisateur distrait/occupé
    MALAISE = auto()  # Utilisateur mal à l'aise
    BESOIN_ESPACE = auto()  # Utilisateur a besoin d'espace
    CONFUSION = auto()  # Utilisateur confus
    FIN_INTERACTION = auto()  # Utilisateur souhaite terminer
    INDECISION = auto()  # Utilisateur indécis
    NORMAL = auto()  # Silence naturel dans la conversation

class SilenceAnalysis(NamedTuple):
    """Résultat d'analyse d'un silence."""

    silence_type: SilenceType
    primary_intention: SilenceIntention
    secondary_intention: Optional[SilenceIntention]
    confidence: float  # 0-1
    adaptation_suggested: Optional[str]


# Configuration des seuils pour les types de silence (en secondes)
SILENCE_THRESHOLDS = {
    "pause_courte": (1, 3),
    "pause_moyenne": (3, 7),
    "pause_longue": (7, 15),
    "absence_reponse": 15,
}

# Modèles de silences et leur interprétation selon le contexte
SILENCE_PATTERNS = {
    # Format: (contexte, type de silence) -> (intention principale, intention
    # secondaire, confiance)
    ("conversation_normale", SilenceType.PAUSE_COURTE): (SilenceIntention.NORMAL, None, 0.8),
    ("conversation_normale", SilenceType.PAUSE_MOYENNE): (
        SilenceIntention.REFLEXION,
        SilenceIntention.NORMAL,
        0.7,
    ),
    ("conversation_normale", SilenceType.PAUSE_LONGUE): (
        SilenceIntention.DISTRACTION,
        SilenceIntention.REFLEXION,
        0.6,
    ),
    ("conversation_normale", SilenceType.ABSENCE_REPONSE): (
        SilenceIntention.DISTRACTION,
        SilenceIntention.FIN_INTERACTION,
        0.5,
    ),
    ("question_posee", SilenceType.PAUSE_COURTE): (SilenceIntention.REFLEXION, None, 0.8),
    ("question_posee", SilenceType.PAUSE_MOYENNE): (
        SilenceIntention.REFLEXION,
        SilenceIntention.INDECISION,
        0.7,
    ),
    ("question_posee", SilenceType.PAUSE_LONGUE): (
        SilenceIntention.INDECISION,
        SilenceIntention.MALAISE,
        0.6,
    ),
    ("question_posee", SilenceType.ABSENCE_REPONSE): (
        SilenceIntention.MALAISE,
        SilenceIntention.BESOIN_ESPACE,
        0.5,
    ),
    ("sujet_sensible", SilenceType.PAUSE_COURTE): (
        SilenceIntention.REFLEXION,
        SilenceIntention.MALAISE,
        0.7,
    ),
    ("sujet_sensible", SilenceType.PAUSE_MOYENNE): (
        SilenceIntention.MALAISE,
        SilenceIntention.REFLEXION,
        0.8,
    ),
    ("sujet_sensible", SilenceType.PAUSE_LONGUE): (
        SilenceIntention.MALAISE,
        SilenceIntention.BESOIN_ESPACE,
        0.9,
    ),
    ("sujet_sensible", SilenceType.ABSENCE_REPONSE): (
        SilenceIntention.BESOIN_ESPACE,
        SilenceIntention.FIN_INTERACTION,
        0.9,
    ),
    ("fin_session", SilenceType.PAUSE_COURTE): (SilenceIntention.NORMAL, None, 0.6),
    ("fin_session", SilenceType.PAUSE_MOYENNE): (SilenceIntention.FIN_INTERACTION, None, 0.7),
    ("fin_session", SilenceType.PAUSE_LONGUE): (SilenceIntention.FIN_INTERACTION, None, 0.9),
    ("fin_session", SilenceType.ABSENCE_REPONSE): (SilenceIntention.FIN_INTERACTION, None, 0.95),
    ("interaction_longue", SilenceType.PAUSE_COURTE): (SilenceIntention.NORMAL, None, 0.6),
    ("interaction_longue", SilenceType.PAUSE_MOYENNE): (
        SilenceIntention.FATIGUE,
        SilenceIntention.NORMAL,
        0.6,
    ),
    ("interaction_longue", SilenceType.PAUSE_LONGUE): (
        SilenceIntention.FATIGUE,
        SilenceIntention.DISTRACTION,
        0.7,
    ),
    ("interaction_longue", SilenceType.ABSENCE_REPONSE): (
        SilenceIntention.FATIGUE,
        SilenceIntention.FIN_INTERACTION,
        0.8,
    ),
    ("hesitation_detectee", SilenceType.HESITATION): (
        SilenceIntention.INDECISION,
        SilenceIntention.CONFUSION,
        0.8,
    ),
}

# Suggestions d'adaptation pour chaque intention
ADAPTATION_SUGGESTIONS = {
    SilenceIntention.REFLEXION: "Laisser le temps de réflexion, ralentir le rythme de conversation",
    SilenceIntention.ATTENTE: "Proposer une action concrète ou clarifier les prochaines étapes",
    SilenceIntention.FATIGUE: "Simplifier les réponses, suggérer une pause ou proposer de continuer plus tard",
    SilenceIntention.DISTRACTION: "Vérifier la présence, résumer la conversation, réengager avec une question directe",
    SilenceIntention.MALAISE: "Adopter un ton plus léger, offrir de changer de sujet, ne pas insister",
    SilenceIntention.BESOIN_ESPACE: "Respecter le silence, signaler la disponibilité sans pression",
    SilenceIntention.CONFUSION: "Clarifier le dernier point, reformuler simplement, vérifier la compréhension",
    SilenceIntention.FIN_INTERACTION: "Proposer de conclure la conversation, résumer les points importants",
    SilenceIntention.INDECISION: "Proposer des options claires, simplifier les choix, rassurer",
    SilenceIntention.NORMAL: "Continuer naturellement la conversation",
}

class SilenceSenseAnalyzer:
    """
    Analyse les silences dans la conversation pour en dériver des intentions
    et adapter le comportement de Jeffrey.
    """

    def __init__(self, sensitivity: float = 0.7):
        """
        Initialise l'analyseur de silence.

        Args:
            sensitivity: Sensibilité de détection des silences significatifs (0-1)
        """
        self.sensitivity = max(0.0, min(1.0, sensitivity))
        self.silence_history = []
        self.last_interaction_time = datetime.datetime.now()
        self.context_history = []
        logger.info(f"SilenceSenseAnalyzer initialisé avec sensibilité {self.sensitivity}")

    def _classify_silence(self, duration: float) -> SilenceType:
        """
        Classifie un silence selon sa durée.

        Args:
            duration: Durée du silence en secondes

        Returns:
            Type de silence classifié
        """
        if duration < SILENCE_THRESHOLDS["pause_courte"][1]:
        return SilenceType.PAUSE_COURTE
                                        elif duration < SILENCE_THRESHOLDS["pause_moyenne"][1]:
        return SilenceType.PAUSE_MOYENNE
                                            elif duration < SILENCE_THRESHOLDS["pause_longue"][1]:
        return SilenceType.PAUSE_LONGUE
                                                else:
        return SilenceType.ABSENCE_REPONSE

    def _detect_hesitation_pattern(self, recent_silences: List[float]) -> bool:
        """
        Détecte un motif d'hésitation dans les silences récents.

        Args:
            recent_silences: Liste des durées de silences récents

        Returns:
            True si un motif d'hésitation est détecté
        """
        if len(recent_silences) < 3:
        return False

        # Définition de l'hésitation: plusieurs pauses courtes consécutives
        short_pauses = [
            s
        for s in recent_silences[-3:]
        if SILENCE_THRESHOLDS["pause_courte"][0] <= s <= SILENCE_THRESHOLDS["pause_courte"][1]
        ]

        return len(short_pauses) >= 2

    def _determine_context(self, last_interaction: Dict[str, Any]) -> str:
        """
        Détermine le contexte conversationnel actuel.

        Args:
            last_interaction: Détails de la dernière interaction

        Returns:
            Contexte identifié
        """
        # Par défaut, considère une conversation normale
        context = "conversation_normale"

        # Si l'interaction contient une question
        if last_interaction.get("contains_question", False):
            context = "question_posee"

        # Si le sujet est considéré sensible
        if last_interaction.get("sensitive_topic", False):
            context = "sujet_sensible"

        # Si l'interaction dure depuis longtemps
        if last_interaction.get("interaction_duration", 0) > 20:  # minutes
            context = "interaction_longue"

        # Si on détecte des signes de conclusion
        if last_interaction.get("closing_signals", 0) > 0.5:
            context = "fin_session"

        # Si on détecte un pattern d'hésitation
        if last_interaction.get("recent_silences") and self._detect_hesitation_pattern(
            last_interaction["recent_silences"]
        ):
            context = "hesitation_detectee"

        return context

    def analyze_silence(
        self, duration: float, last_interaction: Optional[Dict[str, Any]] = None
    ) -> SilenceAnalysis:
        """
        Analyse un silence et renvoie l'intention probable et les adaptations suggérées.

        Args:
            duration: Durée du silence en secondes
            last_interaction: Détails de la dernière interaction, peut inclure:
                - contains_question: si la dernière interaction était une question
                - sensitive_topic: si le sujet est considéré sensible
                - interaction_duration: durée totale de l'interaction en minutes
                - closing_signals: indice de signaux de fin (0-1)
                - recent_silences: liste des durées des silences récents

        Returns:
            Analyse du silence avec type, intentions et suggestion d'adaptation
        """
        if not last_interaction:
            last_interaction = {}

        # Mise à jour de l'historique
        self.silence_history.append(duration)
        if len(self.silence_history) > 10:
            self.silence_history = self.silence_history[-10:]

        # Si le dictionnaire ne contient pas déjà recent_silences
        if "recent_silences" not in last_interaction:
            last_interaction["recent_silences"] = self.silence_history

        # Détermination du type de silence
        silence_type = self._classify_silence(duration)

        # Si on détecte un motif d'hésitation, remplace le type
        if silence_type in [
            SilenceType.PAUSE_COURTE,
            SilenceType.PAUSE_MOYENNE,
        ] and self._detect_hesitation_pattern(last_interaction.get("recent_silences", [])):
            silence_type = SilenceType.HESITATION

        # Détermination du contexte
        context = self._determine_context(last_interaction)
        self.context_history.append(context)
        if len(self.context_history) > 5:
            self.context_history = self.context_history[-5:]

        # Recherche dans les modèles de silence
        pattern_key = (context, silence_type)

        # Si modèle exact trouvé
        if pattern_key in SILENCE_PATTERNS:
            primary_intention, secondary_intention, confidence = SILENCE_PATTERNS[pattern_key]

            # Ajustement de la confiance selon la sensibilité configurée
            adjusted_confidence = confidence * self.sensitivity

        # Sinon, cherche un modèle générique pour ce type de silence
                                                                                                                                                                else:
            # Essaie avec le contexte par défaut "conversation_normale"
            generic_key = ("conversation_normale", silence_type)
        if generic_key in SILENCE_PATTERNS:
                primary_intention, secondary_intention, confidence = SILENCE_PATTERNS[generic_key]
                # Confiance réduite car utilisation de modèle générique
                adjusted_confidence = confidence * self.sensitivity * 0.8
                                                                                                                                                                        else:
                # Cas de secours si aucun modèle ne correspond
                primary_intention = SilenceIntention.NORMAL
                secondary_intention = None
                adjusted_confidence = 0.5 * self.sensitivity

        # Récupère la suggestion d'adaptation pour l'intention principale
        adaptation_suggested = ADAPTATION_SUGGESTIONS.get(primary_intention)

        # Crée et renvoie l'analyse complète
        return SilenceAnalysis(
            silence_type=silence_type,
            primary_intention=primary_intention,
            secondary_intention=secondary_intention,
            confidence=adjusted_confidence,
            adaptation_suggested=adaptation_suggested,
        )

    def get_context_aware_response(
        self, analysis: SilenceAnalysis, current_state: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Génère une réponse adaptée en fonction de l'analyse du silence et de l'état actuel.

        Args:
            analysis: Analyse du silence
            current_state: État actuel de Jeffrey, incluant son état émotionnel, etc.

        Returns:
            Dictionnaire contenant les adaptations recommandées pour la réponse
        """
        response_adaptations = {
            "should_respond": True,  # Par défaut, Jeffrey devrait répondre
            "tone_adjustment": None,  # Ajustement du ton
            "pace_adjustment": 0,  # Ajustement du rythme (-1: ralentir, 0: normal, 1: accélérer)
            "verbosity_adjustment": 0,  # Ajustement de la verbosité (-1: moins, 0: normal, 1: plus)
            "content_suggestion": None,  # Suggestion de contenu
            "question_suggestion": None,  # Suggestion de question à poser
            "emotional_tone": None,  # Tonalité émotionnelle suggérée
        }

        # Adaptations selon l'intention principale
        if analysis.primary_intention == SilenceIntention.REFLEXION:
            response_adaptations["pace_adjustment"] = -1
            response_adaptations["tone_adjustment"] = "thoughtful"
            response_adaptations["emotional_tone"] = "calme"

                                                                                                                                                                                                elif analysis.primary_intention == SilenceIntention.ATTENTE:
            response_adaptations["content_suggestion"] = "proposition_action"
            response_adaptations["verbosity_adjustment"] = -1

                                                                                                                                                                                                    elif analysis.primary_intention == SilenceIntention.FATIGUE:
            response_adaptations["pace_adjustment"] = -1
            response_adaptations["verbosity_adjustment"] = -1
            response_adaptations["content_suggestion"] = "proposition_pause"
            response_adaptations["emotional_tone"] = "douceur"

                                                                                                                                                                                                        elif analysis.primary_intention == SilenceIntention.DISTRACTION:
            response_adaptations["question_suggestion"] = "verification_presence"
            response_adaptations["content_suggestion"] = "resume_conversation"

                                                                                                                                                                                                            elif analysis.primary_intention == SilenceIntention.MALAISE:
            response_adaptations["tone_adjustment"] = "gentle"
            response_adaptations["content_suggestion"] = "changement_sujet"
            response_adaptations["emotional_tone"] = "confiance"

                                                                                                                                                                                                                elif analysis.primary_intention == SilenceIntention.BESOIN_ESPACE:
            response_adaptations["should_respond"] = analysis.confidence < 0.7
            response_adaptations["pace_adjustment"] = -1
            response_adaptations["verbosity_adjustment"] = -1
            response_adaptations["tone_adjustment"] = "respectful"

                                                                                                                                                                                                                    elif analysis.primary_intention == SilenceIntention.CONFUSION:
            response_adaptations["verbosity_adjustment"] = -1
            response_adaptations["content_suggestion"] = "clarification"
            response_adaptations["tone_adjustment"] = "clear"
            response_adaptations["emotional_tone"] = "calme"

                                                                                                                                                                                                                        elif analysis.primary_intention == SilenceIntention.FIN_INTERACTION:
            response_adaptations["content_suggestion"] = "conclusion"
            response_adaptations["verbosity_adjustment"] = -1

                                                                                                                                                                                                                            elif analysis.primary_intention == SilenceIntention.INDECISION:
            response_adaptations["content_suggestion"] = "options_claires"
            response_adaptations["tone_adjustment"] = "reassuring"
            response_adaptations["emotional_tone"] = "confiance"

        # Prise en compte de l'état émotionnel actuel de Jeffrey si disponible
        if current_state and "emotional_state" in current_state:
            # Complexité supplémentaire pour adapter en fonction de l'état actuel
            # Logique plus avancée qui pourrait être implémentée ici
                                                                                                                                                                                                                                    pass

        return response_adaptations

    def update_last_interaction_time(self, timestamp=None):
        """
        Met à jour le timestamp de la dernière interaction.

        Args:
            timestamp: Horodatage à utiliser (ou moment actuel si None)
        """
        self.last_interaction_time = timestamp or datetime.datetime.now()

    def get_silence_duration(self) -> float:
        """
        Calcule la durée du silence depuis la dernière interaction.

        Returns:
            Durée du silence en secondes
        """
        now = datetime.datetime.now()
        return (now - self.last_interaction_time).total_seconds()

    def should_prompt_user(self) -> bool:
        """
        Détermine si Jeffrey devrait relancer l'utilisateur après un long silence.

        Returns:
            True si Jeffrey devrait relancer la conversation
        """
        silence_duration = self.get_silence_duration()

        # Si la durée est supérieure au seuil d'absence de réponse
        if silence_duration > SILENCE_THRESHOLDS["absence_reponse"]:
            # Fréquence de relance diminuant avec le temps
            # Évite de relancer trop souvent après de longs silences
            minutes_silent = silence_duration / 60

            # Première relance après 15-20 secondes
        if 0.25 <= minutes_silent < 1:
        return True

            # Puis après ~5 minutes
                                                                                                                                                                                                                                                                elif 4.9 <= minutes_silent < 5.1:
        return True

            # Puis après ~15 minutes
                                                                                                                                                                                                                                                                    elif 14.9 <= minutes_silent < 15.1:
        return True

            # Dernière relance après ~30 minutes
                                                                                                                                                                                                                                                                        elif 29.9 <= minutes_silent < 30.1:
        return True

        return False

    def get_reconnection_prompt(self, user_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Génère un prompt pour relancer l'utilisateur après un long silence.

        Args:
            user_context: Contexte utilisateur pour personnaliser le prompt

        Returns:
            Prompt de reconnexion
        """
        silence_duration = self.get_silence_duration()
        minutes_silent = int(silence_duration / 60)

        # Prompts pour différentes durées de silence
        if minutes_silent < 1:
            prompts = [
                "Est-ce que tu réfléchis à quelque chose ?",
                "Tu sembles pensif. Je peux aider ?",
                "Tu es toujours là ?",
            ]
                                                                                                                                                                                                                                                                                            elif minutes_silent < 5:
            prompts = [
                "On dirait que tu as pris une petite pause. On continue ?",
                "Tu es revenu ? J'attendais tranquillement.",
                "Est-ce que tu souhaites qu'on reprenne notre conversation ?",
            ]
                                                                                                                                                                                                                                                                                                elif minutes_silent < 15:
            prompts = [
                "Ça fait un moment qu'on n'a pas échangé. Tu veux reprendre ?",
                "Je suis toujours là si tu as besoin de moi.",
                "Tu es occupé ? Je peux attendre ou on peut continuer plus tard.",
            ]
                                                                                                                                                                                                                                                                                                    else:
            prompts = [
                "Bonjour ! On s'était quittés il y a un moment. Tu veux reprendre où on en était ?",
                "Je vois que tu es de retour. Comment puis-je t'aider aujourd'hui ?",
                "Ravi de te revoir ! On continue notre précédente conversation ou on commence quelque chose de nouveau ?",
            ]

        # Si contexte fourni, personnalise davantage
        if user_context:
            # Logique supplémentaire pour personnaliser selon le contexte
                                                                                                                                                                                                                                                                                                            pass

        # Choisit un prompt aléatoire dans la catégorie appropriée
        import random

        return random.choice(prompts)
