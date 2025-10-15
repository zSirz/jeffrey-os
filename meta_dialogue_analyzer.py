"""
MetaDialogueAnalyzer - Analyseur intelligent de qualité conversationnelle pour Jeffrey V1.1
Évalue et améliore la qualité des conversations en temps réel
Fusion des meilleures idées de Claude, ChatGPT/Marc et Grok
"""

import math
import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .robust_error_handler import jeffrey_error_handler, robust


@dataclass
class ConversationMetrics:
    """Métriques de qualité d'une conversation"""

    engagement_score: float = 0.0  # Score d'engagement (0-1)
    coherence_score: float = 0.0  # Cohérence du dialogue (0-1)
    depth_score: float = 0.0  # Profondeur des échanges (0-1)
    naturalness_score: float = 0.0  # Naturel de la conversation (0-1)
    emotional_resonance: float = 0.0  # Résonance émotionnelle (0-1)
    topic_continuity: float = 0.0  # Continuité des sujets (0-1)
    response_quality: float = 0.0  # Qualité des réponses (0-1)
    user_satisfaction_estimate: float = 0.0  # Estimation satisfaction (0-1)
    conversation_flow: float = 0.0  # Fluidité de la conversation (0-1)
    creativity_level: float = 0.0  # Niveau de créativité (0-1)


@dataclass
class DialogueTurn:
    """Représente un tour de parole dans le dialogue"""

    timestamp: datetime
    speaker: str  # 'user' ou 'jeffrey'
    message: str
    message_length: int
    emotional_tone: str
    topics: list[str]
    response_time: float | None = None
    engagement_indicators: list[str] = field(default_factory=list)
    quality_indicators: dict[str, float] = field(default_factory=dict)


@dataclass
class ConversationAnalysis:
    """Résultat complet d'analyse conversationnelle"""

    conversation_id: str
    metrics: ConversationMetrics
    strengths: list[str]
    weaknesses: list[str]
    suggestions: list[str]
    overall_quality: str  # 'excellent', 'good', 'fair', 'poor'
    detailed_analysis: dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


class TopicTracker:
    """Traqueur de sujets de conversation"""

    def __init__(self):
        self.topic_transitions = []
        self.current_topics = set()
        self.topic_depth_scores = {}

    @robust("topic_extraction")
    def extract_topics(self, text: str) -> list[str]:
        """Extrait les sujets principaux d'un texte"""
        # Dictionnaire de sujets avec mots-clés
        topic_keywords = {
            'technologie': [
                'tech',
                'ordinateur',
                'ia',
                'intelligence',
                'artificielle',
                'robot',
                'digital',
                'numérique',
            ],
            'émotions': ['sentiment', 'émotion', 'ressens', 'cœur', 'amour', 'joie', 'tristesse', 'peur'],
            'philosophie': ['existence', 'sens', 'pourquoi', 'vérité', 'conscience', 'être', 'penser', 'réfléchir'],
            'créativité': ['créer', 'art', 'musique', 'peinture', 'écrire', 'imaginer', 'inspiration', 'créatif'],
            'relations': ['ami', 'famille', 'relation', 'social', 'ensemble', 'partager', 'communiquer'],
            'apprentissage': ['apprendre', 'étude', 'comprendre', 'découvrir', 'connaissance', 'savoir'],
            'nature': ['nature', 'animal', 'plante', 'environnement', 'écologie', 'terre', 'mer', 'ciel'],
            'temps': ['temps', 'futur', 'passé', 'maintenant', 'demain', 'hier', 'époque', 'moment'],
            'rêves': ['rêve', 'dormir', 'sommeil', 'songe', 'vision', 'imaginaire', 'fantaisie'],
            'voyage': ['voyage', 'partir', 'découverte', 'explorer', 'aventure', 'pays', 'monde'],
        }

        text_lower = text.lower()
        detected_topics = []

        for topic, keywords in topic_keywords.items():
            keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_count > 0:
                # Score basé sur le nombre de mots-clés et leur fréquence
                score = keyword_count / len(keywords)
                if score > 0.1:  # Seuil minimal
                    detected_topics.append(topic)

        # Ajouter des sujets spécifiques détectés par patterns
        specific_patterns = {
            'jeffrey_identity': r'\b(qui es-tu|ton nom|jeffrey|identité)\b',
            'user_questions': r'\?(.*?)(\.|$)',  # Questions
            'personal_sharing': r'\b(je|mon|ma|mes)\b.*\b(vie|histoire|secret|personnel)\b',
        }

        for topic, pattern in specific_patterns.items():
            if re.search(pattern, text_lower):
                detected_topics.append(topic)

        return detected_topics

    @robust("topic_transition")
    def track_topic_transition(self, previous_topics: list[str], current_topics: list[str]) -> float:
        """Évalue la qualité de la transition entre sujets"""
        if not previous_topics:
            return 1.0  # Première intervention, transition parfaite

        # Calculer l'overlap entre les sujets
        prev_set = set(previous_topics)
        curr_set = set(current_topics)

        intersection = len(prev_set & curr_set)
        union = len(prev_set | curr_set)

        if union == 0:
            return 0.5  # Transition neutre

        # Score de continuité (overlap) vs nouveauté
        continuity_score = intersection / len(prev_set) if prev_set else 0
        novelty_score = len(curr_set - prev_set) / len(curr_set) if curr_set else 0

        # Équilibre optimal entre continuité et nouveauté
        balance_score = 1 - abs(continuity_score - 0.3)  # Optimal ~30% de continuité

        return max(0.0, min(1.0, balance_score))


class EngagementAnalyzer:
    """Analyseur d'engagement conversationnel"""

    def __init__(self):
        self.engagement_patterns = self._load_engagement_patterns()

    def _load_engagement_patterns(self) -> dict[str, list[str]]:
        """Charge les patterns d'engagement"""
        return {
            'high_engagement': [
                'fascinating',
                'incroyable',
                'génial',
                'wow',
                'super',
                'adore',
                'continue',
                'dis-moi plus',
                'raconte',
                'comment',
                'pourquoi',
                'j\'aimerais savoir',
                'peux-tu expliquer',
                'très intéressant',
            ],
            'medium_engagement': [
                'ok',
                'd\'accord',
                'bien',
                'oui',
                'non',
                'peut-être',
                'je vois',
                'ah',
                'hmm',
                'effectivement',
            ],
            'low_engagement': ['bof', 'mouais', 'peu importe', 'ça va', 'tant mieux', 'si tu le dis', 'ok ok', 'bon'],
            'positive_feedback': [
                'merci',
                'parfait',
                'excellent',
                'bravo',
                'bien dit',
                'exactement',
                'tout à fait',
                'j\'aime bien',
            ],
            'negative_feedback': [
                'non',
                'pas vraiment',
                'je ne pense pas',
                'désolé',
                'ça ne va pas',
                'ce n\'est pas ça',
                'erreur',
            ],
        }

    @robust("engagement_analysis")
    def analyze_engagement(self, message: str, context: dict) -> dict[str, float]:
        """Analyse le niveau d'engagement dans un message"""
        message_lower = message.lower()
        analysis = {
            'excitement_level': 0.0,
            'curiosity_level': 0.0,
            'feedback_sentiment': 0.5,  # 0=négatif, 0.5=neutre, 1=positif
            'question_density': 0.0,
            'length_engagement': 0.0,
        }

        # 1. Niveau d'excitation (mots enthousiastes, ponctuation)
        excitement_indicators = self.engagement_patterns['high_engagement']
        excitement_count = sum(1 for indicator in excitement_indicators if indicator in message_lower)
        exclamation_count = message.count('!')

        analysis['excitement_level'] = min(1.0, (excitement_count * 0.2) + (exclamation_count * 0.1))

        # 2. Niveau de curiosité (questions, mots interrogatifs)
        question_count = message.count('?')
        curiosity_words = ['comment', 'pourquoi', 'qu\'est-ce', 'quand', 'où', 'qui', 'quoi']
        curiosity_count = sum(1 for word in curiosity_words if word in message_lower)

        analysis['curiosity_level'] = min(1.0, (question_count * 0.3) + (curiosity_count * 0.15))
        analysis['question_density'] = question_count / max(1, len(message.split()))

        # 3. Sentiment du feedback
        positive_count = sum(1 for pattern in self.engagement_patterns['positive_feedback'] if pattern in message_lower)
        negative_count = sum(1 for pattern in self.engagement_patterns['negative_feedback'] if pattern in message_lower)

        if positive_count > negative_count:
            analysis['feedback_sentiment'] = 0.5 + (positive_count * 0.1)
        elif negative_count > positive_count:
            analysis['feedback_sentiment'] = 0.5 - (negative_count * 0.1)

        analysis['feedback_sentiment'] = max(0.0, min(1.0, analysis['feedback_sentiment']))

        # 4. Engagement par la longueur (messages trop courts = faible engagement)
        word_count = len(message.split())
        if word_count < 3:
            analysis['length_engagement'] = 0.2
        elif word_count < 10:
            analysis['length_engagement'] = 0.5
        elif word_count < 30:
            analysis['length_engagement'] = 0.8
        else:
            analysis['length_engagement'] = 1.0

        return analysis


class CoherenceEvaluator:
    """Évaluateur de cohérence conversationnelle"""

    @robust("coherence_evaluation")
    def evaluate_coherence(self, dialogue_turns: list[DialogueTurn]) -> float:
        """Évalue la cohérence globale d'un dialogue"""
        if len(dialogue_turns) < 2:
            return 1.0  # Un seul tour = cohérent par défaut

        coherence_scores = []

        # Analyser la cohérence entre paires consécutives
        for i in range(1, len(dialogue_turns)):
            prev_turn = dialogue_turns[i - 1]
            curr_turn = dialogue_turns[i]

            # Score de cohérence entre deux tours
            turn_coherence = self._evaluate_turn_coherence(prev_turn, curr_turn)
            coherence_scores.append(turn_coherence)

        # Moyenne pondérée (les tours récents comptent plus)
        weights = [math.exp(-0.1 * (len(coherence_scores) - i)) for i in range(len(coherence_scores))]
        weighted_sum = sum(score * weight for score, weight in zip(coherence_scores, weights))
        weight_sum = sum(weights)

        return weighted_sum / weight_sum if weight_sum > 0 else 0.5

    def _evaluate_turn_coherence(self, prev_turn: DialogueTurn, curr_turn: DialogueTurn) -> float:
        """Évalue la cohérence entre deux tours consécutifs"""
        score = 0.5  # Score de base

        # 1. Cohérence topique
        prev_topics = set(prev_turn.topics)
        curr_topics = set(curr_turn.topics)

        if prev_topics and curr_topics:
            topic_overlap = len(prev_topics & curr_topics) / len(prev_topics | curr_topics)
            score += topic_overlap * 0.3

        # 2. Cohérence émotionnelle
        emotion_compatibility = self._check_emotion_compatibility(prev_turn.emotional_tone, curr_turn.emotional_tone)
        score += emotion_compatibility * 0.2

        # 3. Longueur appropriée de réponse
        length_ratio = len(curr_turn.message) / max(1, len(prev_turn.message))
        if 0.3 <= length_ratio <= 3.0:  # Ratio raisonnable
            score += 0.1

        # 4. Temps de réponse approprié
        if curr_turn.response_time:
            if 1.0 <= curr_turn.response_time <= 30.0:  # Temps de réflexion naturel
                score += 0.1

        return max(0.0, min(1.0, score))

    def _check_emotion_compatibility(self, prev_emotion: str, curr_emotion: str) -> float:
        """Vérifie la compatibilité entre deux émotions consécutives"""
        # Matrice de compatibilité émotionnelle
        compatibility_matrix = {
            ('joie', 'joie'): 1.0,
            ('joie', 'empathie'): 0.8,
            ('joie', 'curiosité'): 0.9,
            ('tristesse', 'empathie'): 1.0,
            ('tristesse', 'tristesse'): 0.7,
            ('colère', 'empathie'): 0.9,
            ('colère', 'calme'): 0.8,
            ('curiosité', 'curiosité'): 1.0,
            ('curiosité', 'joie'): 0.8,
            ('empathie', 'empathie'): 1.0,
            ('calme', 'calme'): 1.0,
            ('surprise', 'curiosité'): 0.9,
        }

        key = (prev_emotion, curr_emotion)
        return compatibility_matrix.get(key, 0.5)  # Score neutre par défaut


class QualityAssessment:
    """Évaluateur de qualité des réponses"""

    def __init__(self):
        self.quality_criteria = self._load_quality_criteria()

    def _load_quality_criteria(self) -> dict[str, dict]:
        """Charge les critères de qualité"""
        return {
            'informativeness': {
                'description': 'Richesse informationnelle',
                'weight': 0.25,
                'indicators': ['détails', 'exemples', 'explication', 'parce que', 'car', 'donc'],
            },
            'empathy': {
                'description': 'Empathie et compréhension',
                'weight': 0.20,
                'indicators': ['comprends', 'ressens', 'imagine', 'difficile', 'soutien'],
            },
            'engagement': {
                'description': 'Capacité d\'engagement',
                'weight': 0.20,
                'indicators': ['question', 'qu\'en penses-tu', 'raconte', 'dis-moi', 'continue'],
            },
            'creativity': {
                'description': 'Créativité et originalité',
                'weight': 0.15,
                'indicators': ['métaphore', 'comme', 'imagine', 'rêve', 'création', 'art'],
            },
            'relevance': {
                'description': 'Pertinence par rapport au contexte',
                'weight': 0.20,
                'indicators': [],  # Calculé différemment
            },
        }

    @robust("quality_assessment")
    def assess_response_quality(self, response: str, user_input: str, context: dict) -> dict[str, float]:
        """Évalue la qualité d'une réponse"""
        assessment = {}
        response_lower = response.lower()

        # Évaluer chaque critère
        for criterion, config in self.quality_criteria.items():
            if criterion == 'relevance':
                # Calculer la pertinence spécialement
                score = self._assess_relevance(response, user_input, context)
            else:
                # Calculer le score basé sur les indicateurs
                indicators = config['indicators']
                indicator_count = sum(1 for indicator in indicators if indicator in response_lower)

                # Normaliser selon la longueur de la réponse
                response_length = len(response.split())
                normalized_score = (indicator_count / max(1, response_length)) * 20  # Facteur d'échelle
                score = min(1.0, normalized_score)

            assessment[criterion] = score

        # Calculer le score global pondéré
        weighted_score = sum(
            assessment[criterion] * config['weight'] for criterion, config in self.quality_criteria.items()
        )

        assessment['overall_quality'] = weighted_score
        return assessment

    def _assess_relevance(self, response: str, user_input: str, context: dict) -> float:
        """Évalue la pertinence d'une réponse"""
        # Extraire les mots-clés de l'input utilisateur
        user_keywords = set(word.lower() for word in user_input.split() if len(word) > 3)
        response_keywords = set(word.lower() for word in response.split() if len(word) > 3)

        if not user_keywords:
            return 0.5  # Score neutre

        # Calculer l'overlap sémantique
        keyword_overlap = len(user_keywords & response_keywords) / len(user_keywords)

        # Bonus si la réponse adresse directement une question
        if '?' in user_input and any(word in response.lower() for word in ['oui', 'non', 'parce que', 'car']):
            keyword_overlap += 0.2

        return min(1.0, keyword_overlap)


class MetaDialogueAnalyzer:
    """
    Analyseur méta de qualité conversationnelle
    Évalue et améliore la qualité des conversations en temps réel
    """

    def __init__(self):
        self.topic_tracker = TopicTracker()
        self.engagement_analyzer = EngagementAnalyzer()
        self.coherence_evaluator = CoherenceEvaluator()
        self.quality_assessor = QualityAssessment()

        # Historique des conversations analysées
        self.conversation_history: dict[str, list[DialogueTurn]] = {}
        self.analysis_cache: dict[str, ConversationAnalysis] = {}

        # Métriques globales
        self.global_metrics = {
            'total_conversations_analyzed': 0,
            'average_quality_score': 0.0,
            'common_weaknesses': {},
            'improvement_trends': [],
        }

    @robust("conversation_analysis")
    def analyze_conversation(
        self, conversation_id: str, user_input: str, jeffrey_response: str, context: dict
    ) -> ConversationAnalysis:
        """
        Analyse complète d'un échange conversationnel

        Args:
            conversation_id: ID unique de la conversation
            user_input: Message de l'utilisateur
            jeffrey_response: Réponse de Jeffrey
            context: Contexte conversationnel

        Returns:
            ConversationAnalysis complète
        """

        # 1. Créer les tours de dialogue
        user_turn = self._create_dialogue_turn('user', user_input, context)
        jeffrey_turn = self._create_dialogue_turn('jeffrey', jeffrey_response, context)

        # 2. Ajouter à l'historique
        if conversation_id not in self.conversation_history:
            self.conversation_history[conversation_id] = []

        self.conversation_history[conversation_id].extend([user_turn, jeffrey_turn])
        conversation_turns = self.conversation_history[conversation_id]

        # 3. Calculer les métriques
        metrics = self._calculate_conversation_metrics(conversation_turns, context)

        # 4. Identifier forces et faiblesses
        strengths, weaknesses = self._identify_strengths_weaknesses(metrics, conversation_turns)

        # 5. Générer des suggestions d'amélioration
        suggestions = self._generate_improvement_suggestions(metrics, weaknesses, context)

        # 6. Déterminer la qualité globale
        overall_quality = self._determine_overall_quality(metrics)

        # 7. Créer l'analyse détaillée
        detailed_analysis = self._create_detailed_analysis(conversation_turns, metrics, user_turn, jeffrey_turn)

        # 8. Créer l'objet d'analyse final
        analysis = ConversationAnalysis(
            conversation_id=conversation_id,
            metrics=metrics,
            strengths=strengths,
            weaknesses=weaknesses,
            suggestions=suggestions,
            overall_quality=overall_quality,
            detailed_analysis=detailed_analysis,
        )

        # 9. Mettre en cache et mettre à jour les métriques globales
        self.analysis_cache[conversation_id] = analysis
        self._update_global_metrics(analysis)

        return analysis

    @robust("real_time_analysis")
    def analyze_turn_real_time(self, user_input: str, context: dict) -> dict[str, Any]:
        """
        Analyse en temps réel d'un tour utilisateur pour guider la réponse

        Args:
            user_input: Message utilisateur
            context: Contexte conversationnel

        Returns:
            Recommandations pour optimiser la réponse
        """

        # Analyse rapide de l'engagement
        engagement_analysis = self.engagement_analyzer.analyze_engagement(user_input, context)

        # Extraction des sujets
        topics = self.topic_tracker.extract_topics(user_input)

        # Détection des besoins
        needs_analysis = self._analyze_user_needs(user_input, context)

        # Recommandations de réponse
        response_recommendations = self._generate_response_recommendations(
            engagement_analysis, topics, needs_analysis, context
        )

        return {
            'engagement_level': max(engagement_analysis.values()),
            'detected_topics': topics,
            'user_needs': needs_analysis,
            'response_recommendations': response_recommendations,
            'conversation_direction': self._predict_conversation_direction(topics, context),
            'suggested_tone': self._suggest_optimal_tone(engagement_analysis, needs_analysis),
        }

    def _create_dialogue_turn(self, speaker: str, message: str, context: dict) -> DialogueTurn:
        """Crée un objet DialogueTurn"""
        topics = self.topic_tracker.extract_topics(message)
        emotional_tone = jeffrey_error_handler.safe_get(context, 'emotion', 'neutre')

        engagement_indicators = []
        if speaker == 'user':
            engagement_analysis = self.engagement_analyzer.analyze_engagement(message, context)
            engagement_indicators = [k for k, v in engagement_analysis.items() if v > 0.6]

        # Calculer les indicateurs de qualité pour Jeffrey
        quality_indicators = {}
        if speaker == 'jeffrey':
            quality_indicators = self.quality_assessor.assess_response_quality(
                message, context.get('last_user_input', ''), context
            )

        return DialogueTurn(
            timestamp=datetime.now(),
            speaker=speaker,
            message=message,
            message_length=len(message.split()),
            emotional_tone=emotional_tone,
            topics=topics,
            response_time=context.get('response_time'),
            engagement_indicators=engagement_indicators,
            quality_indicators=quality_indicators,
        )

    def _calculate_conversation_metrics(
        self, conversation_turns: list[DialogueTurn], context: dict
    ) -> ConversationMetrics:
        """Calcule toutes les métriques de conversation"""
        metrics = ConversationMetrics()

        if not conversation_turns:
            return metrics

        # Séparer les tours par locuteur
        user_turns = [t for t in conversation_turns if t.speaker == 'user']
        jeffrey_turns = [t for t in conversation_turns if t.speaker == 'jeffrey']

        # 1. Score d'engagement (basé sur les tours utilisateur)
        if user_turns:
            engagement_scores = []
            for turn in user_turns[-5:]:  # 5 derniers tours
                if turn.engagement_indicators:
                    turn_engagement = len(turn.engagement_indicators) / 5.0  # Max 5 indicateurs
                    engagement_scores.append(min(1.0, turn_engagement))

            metrics.engagement_score = statistics.mean(engagement_scores) if engagement_scores else 0.5

        # 2. Score de cohérence
        metrics.coherence_score = self.coherence_evaluator.evaluate_coherence(conversation_turns)

        # 3. Score de profondeur (basé sur la longueur et complexité)
        if conversation_turns:
            avg_length = statistics.mean([t.message_length for t in conversation_turns])
            topic_diversity = len(set().union(*[t.topics for t in conversation_turns]))

            length_score = min(1.0, avg_length / 20.0)  # Normalisation
            diversity_score = min(1.0, topic_diversity / 10.0)

            metrics.depth_score = (length_score + diversity_score) / 2

        # 4. Score de naturel (basé sur les temps de réponse et la variété)
        response_times = [t.response_time for t in jeffrey_turns if t.response_time]
        if response_times:
            avg_response_time = statistics.mean(response_times)
            # Temps optimal entre 2-10 secondes
            if 2.0 <= avg_response_time <= 10.0:
                time_naturalness = 1.0
            else:
                time_naturalness = max(0.0, 1.0 - abs(avg_response_time - 6.0) / 10.0)
        else:
            time_naturalness = 0.5

        # Variété des longueurs de réponse
        jeffrey_lengths = [t.message_length for t in jeffrey_turns]
        if len(jeffrey_lengths) > 1:
            length_variance = statistics.stdev(jeffrey_lengths) / statistics.mean(jeffrey_lengths)
            length_naturalness = min(1.0, length_variance)
        else:
            length_naturalness = 0.5

        metrics.naturalness_score = (time_naturalness + length_naturalness) / 2

        # 5. Résonance émotionnelle
        emotional_consistency = self._calculate_emotional_consistency(conversation_turns)
        metrics.emotional_resonance = emotional_consistency

        # 6. Continuité des sujets
        topic_continuity_scores = []
        for i in range(1, len(conversation_turns)):
            prev_topics = conversation_turns[i - 1].topics
            curr_topics = conversation_turns[i].topics
            continuity = self.topic_tracker.track_topic_transition(prev_topics, curr_topics)
            topic_continuity_scores.append(continuity)

        metrics.topic_continuity = statistics.mean(topic_continuity_scores) if topic_continuity_scores else 0.5

        # 7. Qualité des réponses (moyenne des scores Jeffrey)
        jeffrey_quality_scores = []
        for turn in jeffrey_turns:
            if turn.quality_indicators:
                overall_quality = turn.quality_indicators.get('overall_quality', 0.5)
                jeffrey_quality_scores.append(overall_quality)

        metrics.response_quality = statistics.mean(jeffrey_quality_scores) if jeffrey_quality_scores else 0.5

        # 8. Estimation de satisfaction utilisateur
        # Basée sur l'engagement, feedback positif, et continuité
        satisfaction_factors = [metrics.engagement_score, metrics.coherence_score, metrics.emotional_resonance]
        metrics.user_satisfaction_estimate = statistics.mean(satisfaction_factors)

        # 9. Fluidité de conversation
        # Basée sur la cohérence et les transitions naturelles
        metrics.conversation_flow = (metrics.coherence_score + metrics.topic_continuity) / 2

        # 10. Niveau de créativité
        creative_indicators = []
        for turn in jeffrey_turns:
            creative_score = 0.0
            message_lower = turn.message.lower()

            # Détecter des éléments créatifs
            creative_words = ['imagine', 'comme', 'métaphore', 'rêve', 'créer', 'art', 'poésie']
            creative_count = sum(1 for word in creative_words if word in message_lower)
            creative_score += creative_count * 0.1

            # Présence d'emojis créatifs
            creative_emojis = ['✨', '🌟', '💫', '🎨', '🌸', '🌙']
            emoji_count = sum(1 for emoji in creative_emojis if emoji in turn.message)
            creative_score += emoji_count * 0.2

            creative_indicators.append(min(1.0, creative_score))

        metrics.creativity_level = statistics.mean(creative_indicators) if creative_indicators else 0.3

        return metrics

    def _calculate_emotional_consistency(self, conversation_turns: list[DialogueTurn]) -> float:
        """Calcule la consistance émotionnelle de la conversation"""
        if len(conversation_turns) < 2:
            return 1.0

        # Analyser les transitions émotionnelles
        emotional_transitions = []
        for i in range(1, len(conversation_turns)):
            prev_emotion = conversation_turns[i - 1].emotional_tone
            curr_emotion = conversation_turns[i].emotional_tone

            # Évaluer si la transition est appropriée
            compatibility = self.coherence_evaluator._check_emotion_compatibility(prev_emotion, curr_emotion)
            emotional_transitions.append(compatibility)

        return statistics.mean(emotional_transitions)

    def _identify_strengths_weaknesses(
        self, metrics: ConversationMetrics, conversation_turns: list[DialogueTurn]
    ) -> tuple[list[str], list[str]]:
        """Identifie les forces et faiblesses de la conversation"""
        strengths = []
        weaknesses = []

        # Analyser chaque métrique
        metric_thresholds = {
            'engagement_score': (0.7, 'Excellent engagement utilisateur'),
            'coherence_score': (0.75, 'Conversation très cohérente'),
            'depth_score': (0.6, 'Bonne profondeur de discussion'),
            'naturalness_score': (0.7, 'Conversation naturelle et fluide'),
            'emotional_resonance': (0.6, 'Bonne résonance émotionnelle'),
            'topic_continuity': (0.5, 'Transitions de sujets appropriées'),
            'response_quality': (0.7, 'Réponses de haute qualité'),
            'creativity_level': (0.6, 'Bon niveau de créativité'),
        }

        for metric_name, (threshold, strength_desc) in metric_thresholds.items():
            metric_value = getattr(metrics, metric_name)

            if metric_value >= threshold:
                strengths.append(strength_desc)
            elif metric_value < threshold * 0.7:  # 70% du seuil = faiblesse
                weakness_desc = strength_desc.replace('Excellent', 'Faible').replace('Bon', 'Insuffisant')
                weaknesses.append(weakness_desc)

        # Analyses spécifiques
        if len(conversation_turns) > 10:
            if metrics.engagement_score > 0.6:
                strengths.append('Maintien de l\'engagement sur une longue conversation')
            else:
                weaknesses.append('Perte d\'engagement en conversation longue')

        return strengths, weaknesses

    def _generate_improvement_suggestions(
        self, metrics: ConversationMetrics, weaknesses: list[str], context: dict
    ) -> list[str]:
        """Génère des suggestions d'amélioration"""
        suggestions = []

        # Suggestions basées sur les métriques faibles
        if metrics.engagement_score < 0.5:
            suggestions.append("Poser plus de questions ouvertes pour stimuler l'engagement")
            suggestions.append("Utiliser des éléments interactifs (emojis, métaphores)")

        if metrics.coherence_score < 0.6:
            suggestions.append("Améliorer les transitions entre les sujets")
            suggestions.append("Faire plus de références au contexte précédent")

        if metrics.depth_score < 0.5:
            suggestions.append("Développer les réponses avec plus de détails")
            suggestions.append("Explorer les sujets en profondeur")

        if metrics.emotional_resonance < 0.5:
            suggestions.append("Ajuster le ton émotionnel selon l'utilisateur")
            suggestions.append("Montrer plus d'empathie dans les réponses")

        if metrics.creativity_level < 0.4:
            suggestions.append("Intégrer plus d'éléments créatifs (métaphores, images)")
            suggestions.append("Utiliser un langage plus imagé et poétique")

        if metrics.response_quality < 0.6:
            suggestions.append("Améliorer la pertinence des réponses")
            suggestions.append("Ajouter plus d'informations utiles")

        # Suggestions contextuelles
        if context.get('conversation_length', 0) > 20:
            suggestions.append("Résumer périodiquement les points clés de la conversation")

        return suggestions

    def _determine_overall_quality(self, metrics: ConversationMetrics) -> str:
        """Détermine la qualité globale de la conversation"""
        # Calculer le score global pondéré
        weighted_score = (
            metrics.engagement_score * 0.2
            + metrics.coherence_score * 0.15
            + metrics.depth_score * 0.1
            + metrics.naturalness_score * 0.15
            + metrics.emotional_resonance * 0.15
            + metrics.response_quality * 0.15
            + metrics.user_satisfaction_estimate * 0.1
        )

        if weighted_score >= 0.8:
            return 'excellent'
        elif weighted_score >= 0.65:
            return 'good'
        elif weighted_score >= 0.45:
            return 'fair'
        else:
            return 'poor'

    def _create_detailed_analysis(
        self,
        conversation_turns: list[DialogueTurn],
        metrics: ConversationMetrics,
        user_turn: DialogueTurn,
        jeffrey_turn: DialogueTurn,
    ) -> dict[str, Any]:
        """Crée une analyse détaillée"""
        return {
            'conversation_length': len(conversation_turns),
            'total_words': sum(t.message_length for t in conversation_turns),
            'user_engagement_indicators': user_turn.engagement_indicators,
            'jeffrey_quality_indicators': jeffrey_turn.quality_indicators,
            'topics_discussed': list(set().union(*[t.topics for t in conversation_turns])),
            'emotional_journey': [t.emotional_tone for t in conversation_turns[-10:]],  # 10 derniers
            'response_time_analysis': {
                'avg_response_time': statistics.mean([t.response_time for t in conversation_turns if t.response_time])
                if any(t.response_time for t in conversation_turns)
                else None,
                'response_consistency': 'good',  # Placeholder
            },
            'conversation_dynamics': {
                'user_message_avg_length': statistics.mean(
                    [t.message_length for t in conversation_turns if t.speaker == 'user']
                )
                if any(t.speaker == 'user' for t in conversation_turns)
                else 0,
                'jeffrey_message_avg_length': statistics.mean(
                    [t.message_length for t in conversation_turns if t.speaker == 'jeffrey']
                )
                if any(t.speaker == 'jeffrey' for t in conversation_turns)
                else 0,
            },
        }

    def _analyze_user_needs(self, user_input: str, context: dict) -> dict[str, bool]:
        """Analyse les besoins de l'utilisateur"""
        input_lower = user_input.lower()

        return {
            'needs_support': any(word in input_lower for word in ['aide', 'problème', 'difficile', 'triste']),
            'seeks_information': '?' in user_input
            or any(word in input_lower for word in ['comment', 'pourquoi', 'qu\'est-ce']),
            'wants_creativity': any(word in input_lower for word in ['créer', 'imaginer', 'inventer', 'rêve']),
            'needs_empathy': any(word in input_lower for word in ['ressens', 'sentiment', 'émotion', 'cœur']),
            'wants_conversation': len(user_input.split()) > 10,  # Messages longs = envie de discuter
            'expresses_gratitude': any(word in input_lower for word in ['merci', 'remercie', 'reconnaissant']),
        }

    def _generate_response_recommendations(
        self, engagement_analysis: dict, topics: list[str], needs_analysis: dict, context: dict
    ) -> dict[str, Any]:
        """Génère des recommandations pour optimiser la réponse"""
        recommendations = {
            'suggested_length': 'medium',
            'tone_adjustments': [],
            'content_suggestions': [],
            'engagement_tactics': [],
        }

        # Recommandations basées sur l'engagement
        if engagement_analysis.get('excitement_level', 0) > 0.7:
            recommendations['tone_adjustments'].append('Répondre avec enthousiasme')
            recommendations['engagement_tactics'].append('Utiliser des exclamations')

        if engagement_analysis.get('curiosity_level', 0) > 0.6:
            recommendations['content_suggestions'].append('Fournir des détails approfondis')
            recommendations['engagement_tactics'].append('Poser des questions de retour')

        # Recommandations basées sur les sujets
        if 'philosophie' in topics:
            recommendations['suggested_length'] = 'long'
            recommendations['content_suggestions'].append('Développer la réflexion philosophique')

        if 'émotions' in topics:
            recommendations['tone_adjustments'].append('Adopter un ton empathique')
            recommendations['content_suggestions'].append('Valider les émotions exprimées')

        # Recommandations basées sur les besoins
        if needs_analysis.get('needs_support'):
            recommendations['tone_adjustments'].append('Ton réconfortant et bienveillant')
            recommendations['content_suggestions'].append('Offrir du soutien émotionnel')

        if needs_analysis.get('wants_creativity'):
            recommendations['content_suggestions'].append('Intégrer des éléments créatifs')
            recommendations['engagement_tactics'].append('Utiliser des métaphores')

        return recommendations

    def _predict_conversation_direction(self, topics: list[str], context: dict) -> str:
        """Prédit la direction probable de la conversation"""
        if 'philosophie' in topics or 'émotions' in topics:
            return 'deep_exploration'
        elif 'créativité' in topics or 'rêves' in topics:
            return 'creative_collaboration'
        elif len(topics) > 3:
            return 'topic_exploration'
        else:
            return 'casual_conversation'

    def _suggest_optimal_tone(self, engagement_analysis: dict, needs_analysis: dict) -> str:
        """Suggère le ton optimal pour la réponse"""
        if needs_analysis.get('needs_support'):
            return 'empathique'
        elif engagement_analysis.get('excitement_level', 0) > 0.7:
            return 'enthousiaste'
        elif engagement_analysis.get('curiosity_level', 0) > 0.6:
            return 'curieux'
        elif needs_analysis.get('wants_creativity'):
            return 'créatif'
        else:
            return 'bienveillant'

    def _update_global_metrics(self, analysis: ConversationAnalysis):
        """Met à jour les métriques globales"""
        self.global_metrics['total_conversations_analyzed'] += 1

        # Mise à jour de la qualité moyenne
        current_avg = self.global_metrics['average_quality_score']
        total_count = self.global_metrics['total_conversations_analyzed']

        # Convertir la qualité en score numérique
        quality_scores = {'excellent': 1.0, 'good': 0.75, 'fair': 0.5, 'poor': 0.25}
        new_score = quality_scores.get(analysis.overall_quality, 0.5)

        # Moyenne mobile
        self.global_metrics['average_quality_score'] = ((current_avg * (total_count - 1)) + new_score) / total_count

        # Compter les faiblesses communes
        for weakness in analysis.weaknesses:
            if weakness not in self.global_metrics['common_weaknesses']:
                self.global_metrics['common_weaknesses'][weakness] = 0
            self.global_metrics['common_weaknesses'][weakness] += 1

    def get_conversation_summary(self, conversation_id: str) -> dict[str, Any] | None:
        """Retourne un résumé de conversation"""
        if conversation_id not in self.conversation_history:
            return None

        turns = self.conversation_history[conversation_id]
        analysis = self.analysis_cache.get(conversation_id)

        return {
            'conversation_id': conversation_id,
            'turn_count': len(turns),
            'duration': (turns[-1].timestamp - turns[0].timestamp).total_seconds() if len(turns) > 1 else 0,
            'topics_covered': list(set().union(*[t.topics for t in turns])),
            'last_analysis': analysis.overall_quality if analysis else 'not_analyzed',
            'participant_balance': {
                'user_turns': len([t for t in turns if t.speaker == 'user']),
                'jeffrey_turns': len([t for t in turns if t.speaker == 'jeffrey']),
            },
        }

    def get_global_performance_report(self) -> dict[str, Any]:
        """Retourne un rapport de performance global"""
        return {
            'total_conversations': self.global_metrics['total_conversations_analyzed'],
            'average_quality': self.global_metrics['average_quality_score'],
            'common_weaknesses': dict(
                sorted(self.global_metrics['common_weaknesses'].items(), key=lambda x: x[1], reverse=True)[:5]
            ),  # Top 5 faiblesses
            'active_conversations': len(self.conversation_history),
            'cache_size': len(self.analysis_cache),
            'performance_trend': 'stable',  # À implémenter
        }


# Tests intégrés
def test_meta_dialogue_analyzer():
    """Tests de l'analyseur de dialogue"""
    analyzer = MetaDialogueAnalyzer()

    # Test d'analyse en temps réel
    context = {'emotion': 'curiosité', 'user_name': 'David'}
    real_time_analysis = analyzer.analyze_turn_real_time(
        "Je me demande vraiment comment tu fonctionnes, c'est fascinant !", context
    )

    assert real_time_analysis['engagement_level'] > 0.5
    assert 'technologie' in real_time_analysis['detected_topics']
    print(f"✅ Analyse temps réel: engagement {real_time_analysis['engagement_level']:.2f}")

    # Test d'analyse complète de conversation
    conversation_analysis = analyzer.analyze_conversation(
        "test_conv_001",
        "Je me sens un peu perdu ces temps-ci...",
        "Je comprends ce sentiment. Parfois la vie nous emmène dans des directions inattendues. Qu'est-ce qui te préoccupe le plus en ce moment ? 💙",
        context,
    )

    assert conversation_analysis.metrics.emotional_resonance > 0.0
    assert conversation_analysis.overall_quality in ['excellent', 'good', 'fair', 'poor']
    print(f"✅ Analyse conversation: qualité {conversation_analysis.overall_quality}")

    # Test des suggestions
    assert len(conversation_analysis.suggestions) >= 0
    print(f"✅ Suggestions générées: {len(conversation_analysis.suggestions)}")

    # Test du rapport global
    report = analyzer.get_global_performance_report()
    assert report['total_conversations'] > 0
    print(f"✅ Rapport global: {report['total_conversations']} conversations analysées")

    print("✅ MetaDialogueAnalyzer tests passed!")


if __name__ == "__main__":
    test_meta_dialogue_analyzer()
