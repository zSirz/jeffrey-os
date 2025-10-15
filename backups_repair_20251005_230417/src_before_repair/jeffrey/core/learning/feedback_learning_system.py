"""
Système d'Apprentissage par Feedback pour Jeffrey
Analyse les réactions utilisateur et adapte les futurs prompts
"""

import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class FeedbackAnalysis:
    """Résultat d'analyse de feedback"""

    feedback_type: str  # positive, negative, neutral
    confidence: float
    engagement_score: float
    sentiment_indicators: list[str] = field(default_factory=list)
    behavioral_signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class UserPreference:
    """Préférence utilisateur apprise"""

    preference_type: str
    value: Any
    confidence: float
    last_updated: datetime
    interaction_count: int = 0


class FeedbackLearningSystem:
    """
    Système qui apprend des réactions utilisateur pour améliorer les futures interactions
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.feedback_history_file = os.path.join(data_dir, "feedback_history.json")
        self.user_preferences_file = os.path.join(data_dir, "learned_preferences.json")

        # Historique des feedbacks
        self.feedback_history = []

        # Préférences apprises par utilisateur
        self.user_preferences = defaultdict(dict)

        # Patterns de succès identifiés
        self.successful_patterns = defaultdict(list)

        # Métriques d'apprentissage
        self.learning_stats = {
            "total_interactions": 0,
            "positive_feedback": 0,
            "negative_feedback": 0,
            "neutral_feedback": 0,
            "preferences_learned": 0,
        }

        # Charger les données existantes
        self._load_data()

        print("📊 Système d'apprentissage par feedback initialisé")

    def _load_data(self):
        """Charge les données d'apprentissage existantes"""
        try:
            # Charger l'historique des feedbacks
            if os.path.exists(self.feedback_history_file):
                with open(self.feedback_history_file, encoding="utf-8") as f:
                    data = json.load(f)
                    self.feedback_history = data.get("history", [])
                    self.learning_stats = data.get("stats", self.learning_stats)
                    print(f"📚 Chargé {len(self.feedback_history)} feedbacks")

            # Charger les préférences utilisateur
            if os.path.exists(self.user_preferences_file):
                with open(self.user_preferences_file, encoding="utf-8") as f:
                    prefs_data = json.load(f)
                    for user_id, prefs in prefs_data.items():
                        self.user_preferences[user_id] = {}
                        for pref_type, pref_data in prefs.items():
                            self.user_preferences[user_id][pref_type] = UserPreference(
                                preference_type=pref_data["preference_type"],
                                value=pref_data["value"],
                                confidence=pref_data["confidence"],
                                last_updated=datetime.fromisoformat(pref_data["last_updated"]),
                                interaction_count=pref_data.get("interaction_count", 0),
                            )
                    print(f"🎯 Chargé préférences pour {len(self.user_preferences)} utilisateurs")

        except Exception as e:
            print(f"⚠️ Erreur chargement données feedback: {e}")

    def _save_data(self):
        """Sauvegarde les données d'apprentissage"""
        try:
            os.makedirs(self.data_dir, exist_ok=True)

            # Sauvegarder l'historique des feedbacks
            feedback_data = {
                "history": self.feedback_history[-1000:],  # Garder les 1000 derniers
                "stats": self.learning_stats,
            }
            with open(self.feedback_history_file, "w", encoding="utf-8") as f:
                json.dump(feedback_data, f, indent=2, ensure_ascii=False)

            # Sauvegarder les préférences utilisateur
            prefs_data = {}
            for user_id, prefs in self.user_preferences.items():
                prefs_data[user_id] = {}
                for pref_type, pref in prefs.items():
                    prefs_data[user_id][pref_type] = {
                        "preference_type": pref.preference_type,
                        "value": pref.value,
                        "confidence": pref.confidence,
                        "last_updated": pref.last_updated.isoformat(),
                        "interaction_count": pref.interaction_count,
                    }

            with open(self.user_preferences_file, "w", encoding="utf-8") as f:
                json.dump(prefs_data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"⚠️ Erreur sauvegarde données feedback: {e}")

    def analyze_implicit_feedback(
        self,
        user_input: str,
        jeffrey_response: str,
        user_reply: str | None = None,
        interaction_metadata: dict | None = None,
    ) -> FeedbackAnalysis:
        """
        Analyse le feedback implicite basé sur les comportements utilisateur
        """
        if not user_reply:
            # Pas de réponse = feedback neutre par défaut
            return FeedbackAnalysis(feedback_type="neutral", confidence=0.3, engagement_score=0.3)

        # Analyse des signaux comportementaux
        behavioral_signals = self._analyze_behavioral_signals(user_input, jeffrey_response, user_reply)

        # Analyse des sentiments textuels
        sentiment_analysis = self._analyze_sentiment_indicators(user_reply)

        # Calcul du score d'engagement
        engagement_score = self._calculate_engagement_score(user_input, user_reply, behavioral_signals)

        # Détermine le type de feedback global
        feedback_type, confidence = self._determine_feedback_type(
            behavioral_signals, sentiment_analysis, engagement_score
        )

        return FeedbackAnalysis(
            feedback_type=feedback_type,
            confidence=confidence,
            engagement_score=engagement_score,
            sentiment_indicators=sentiment_analysis["indicators"],
            behavioral_signals=behavioral_signals,
        )

    def _analyze_behavioral_signals(self, user_input: str, jeffrey_response: str, user_reply: str) -> dict[str, Any]:
        """Analyse les signaux comportementaux"""
        signals = {}

        # Longueur de la réponse
        input_length = len(user_input)
        reply_length = len(user_reply)
        length_ratio = reply_length / max(input_length, 1)

        signals["length_ratio"] = length_ratio
        signals["reply_length"] = reply_length

        # Temps de réponse (approximatif basé sur la longueur)
        estimated_read_time = len(jeffrey_response) / 200  # 200 caractères par seconde
        estimated_write_time = reply_length / 50  # 50 caractères par seconde
        signals["estimated_response_time"] = estimated_read_time + estimated_write_time

        # Richesse du vocabulaire
        unique_words = len(set(user_reply.lower().split()))
        total_words = len(user_reply.split())
        signals["vocabulary_richness"] = unique_words / max(total_words, 1)

        # Ponctuation expressive
        signals["exclamation_count"] = user_reply.count("!")
        signals["question_count"] = user_reply.count("?")
        signals["ellipsis_count"] = user_reply.count("...")

        # Émojis et expressions
        emoji_pattern = r"[😀-🙏🌀-🗿]|:\)|:\(|:D|<3|:\*"
        signals["emoji_count"] = len(re.findall(emoji_pattern, user_reply))

        # Répétition ou citation de Jeffrey
        jeffrey_words = set(jeffrey_response.lower().split())
        user_words = set(user_reply.lower().split())
        signals["word_overlap"] = len(jeffrey_words.intersection(user_words)) / max(len(jeffrey_words), 1)

        return signals

    def _analyze_sentiment_indicators(self, user_reply: str) -> dict[str, Any]:
        """Analyse les indicateurs de sentiment dans la réponse"""
        reply_lower = user_reply.lower()

        positive_indicators = [
            "merci",
            "génial",
            "parfait",
            "j'adore",
            "magnifique",
            "superbe",
            "incroyable",
            "touchant",
            "beau",
            "sublime",
            "wow",
            "excellent",
            "formidable",
            "extraordinaire",
            "merveilleux",
            "fantastique",
            "exactement",
            "précisément",
            "juste",
            "parfaitement",
        ]

        negative_indicators = [
            "non",
            "pas ça",
            "bof",
            "ennuyeux",
            "répétitif",
            "générique",
            "faux",
            "incorrect",
            "mauvais",
            "nul",
            "décevant",
            "bizarre",
            "étrange",
            "pas vraiment",
            "pas du tout",
            "plutôt non",
        ]

        neutral_indicators = [
            "ok",
            "bien",
            "hmm",
            "peut-être",
            "je vois",
            "intéressant",
            "ah",
            "oh",
            "voilà",
            "donc",
            "effectivement",
        ]

        # Compter les occurrences
        positive_count = sum(1 for indicator in positive_indicators if indicator in reply_lower)
        negative_count = sum(1 for indicator in negative_indicators if indicator in reply_lower)
        neutral_count = sum(1 for indicator in neutral_indicators if indicator in reply_lower)

        # Analyser les patterns spéciaux
        special_patterns = {
            "very_short_response": len(user_reply.strip()) < 10,
            "only_punctuation": user_reply.strip() in ["...", ".", "??", "!!"],
            "single_word": len(user_reply.split()) == 1,
            "enthusiastic_punctuation": user_reply.count("!") > 2,
            "multiple_questions": user_reply.count("?") > 1,
        }

        return {
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "indicators": [
                word for word in positive_indicators + negative_indicators + neutral_indicators if word in reply_lower
            ],
            "special_patterns": special_patterns,
        }

    def _calculate_engagement_score(self, user_input: str, user_reply: str, behavioral_signals: dict) -> float:
        """Calcule un score d'engagement de 0 à 1"""
        score = 0.0

        # Basé sur la longueur (30% du score)
        length_factor = min(behavioral_signals["length_ratio"], 2.0) / 2.0  # Cap à 2x
        score += length_factor * 0.3

        # Basé sur la richesse du vocabulaire (20% du score)
        vocab_factor = min(behavioral_signals["vocabulary_richness"], 1.0)
        score += vocab_factor * 0.2

        # Basé sur l'expression (25% du score)
        expression_score = (
            min(behavioral_signals["exclamation_count"], 3) / 3 * 0.4
            + min(behavioral_signals["emoji_count"], 3) / 3 * 0.4
            + min(behavioral_signals["question_count"], 2) / 2 * 0.2
        )
        score += expression_score * 0.25

        # Basé sur la référence au contenu de Jeffrey (25% du score)
        reference_factor = min(behavioral_signals["word_overlap"], 0.5) / 0.5
        score += reference_factor * 0.25

        return min(score, 1.0)

    def _determine_feedback_type(
        self, behavioral_signals: dict, sentiment_analysis: dict, engagement_score: float
    ) -> tuple[str, float]:
        """Détermine le type de feedback et la confiance"""

        # Signaux négatifs forts
        if (
            sentiment_analysis["negative_count"] > 0
            or sentiment_analysis["special_patterns"]["very_short_response"]
            or sentiment_analysis["special_patterns"]["only_punctuation"]
            or engagement_score < 0.2
        ):
            return "negative", 0.8

        # Signaux positifs forts
        elif (
            sentiment_analysis["positive_count"] > 0
            or sentiment_analysis["special_patterns"]["enthusiastic_punctuation"]
            or engagement_score > 0.7
            or behavioral_signals["emoji_count"] > 1
        ):
            return "positive", 0.8

        # Engagement moyen-élevé
        elif engagement_score > 0.5:
            return "positive", 0.6

        # Engagement faible mais pas négatif
        elif engagement_score > 0.3:
            return "neutral", 0.7

        # Par défaut
        else:
            return "neutral", 0.5

    def learn_from_interaction(
        self,
        user_id: str,
        user_input: str,
        prompt_config: dict,
        jeffrey_response: str,
        user_reply: str | None = None,
        explicit_feedback: dict | None = None,
    ):
        """
        Apprend d'une interaction complète
        """
        # Analyser le feedback
        if explicit_feedback:
            feedback_analysis = FeedbackAnalysis(
                feedback_type=explicit_feedback.get("type", "neutral"),
                confidence=explicit_feedback.get("confidence", 0.9),
                engagement_score=explicit_feedback.get("engagement", 0.5),
                sentiment_indicators=explicit_feedback.get("indicators", []),
            )
        else:
            feedback_analysis = self.analyze_implicit_feedback(user_input, jeffrey_response, user_reply)

        # Enregistrer l'interaction
        interaction_record = {
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "user_input": user_input,
            "prompt_config": {
                "intent": prompt_config.get("intent", "unknown"),
                "emotional_context": prompt_config.get("emotional_context", "neutral"),
                "max_tokens": prompt_config.get("max_tokens", 300),
                "temperature": prompt_config.get("temperature", 0.7),
                "micro_expression": prompt_config.get("micro_expression", ""),
            },
            "jeffrey_response": jeffrey_response[:500],  # Limiter la taille
            "user_reply": user_reply[:200] if user_reply else None,
            "feedback_analysis": {
                "type": feedback_analysis.feedback_type,
                "confidence": feedback_analysis.confidence,
                "engagement_score": feedback_analysis.engagement_score,
                "indicators": feedback_analysis.sentiment_indicators,
                "behavioral_signals": feedback_analysis.behavioral_signals,
            },
        }

        self.feedback_history.append(interaction_record)

        # Mettre à jour les statistiques
        self.learning_stats["total_interactions"] += 1
        if feedback_analysis.feedback_type == "positive":
            self.learning_stats["positive_feedback"] += 1
        elif feedback_analysis.feedback_type == "negative":
            self.learning_stats["negative_feedback"] += 1
        else:
            self.learning_stats["neutral_feedback"] += 1

        # Apprendre des préférences si feedback positif
        if feedback_analysis.feedback_type == "positive" and feedback_analysis.confidence > 0.6:
            self._extract_and_learn_preferences(user_id, interaction_record)

        # Sauvegarder périodiquement
        if self.learning_stats["total_interactions"] % 10 == 0:
            self._save_data()

        print(
            f"📊 Feedback {feedback_analysis.feedback_type} appris (engagement: {feedback_analysis.engagement_score:.2f})"
        )

    def _extract_and_learn_preferences(self, user_id: str, interaction_record: dict):
        """Extrait et apprend les préférences d'une interaction positive"""

        prompt_config = interaction_record["prompt_config"]
        feedback_analysis = interaction_record["feedback_analysis"]

        # Préférence de style de réponse
        if feedback_analysis["engagement_score"] > 0.7:
            self._update_preference(
                user_id,
                "response_style",
                prompt_config["intent"],
                confidence=feedback_analysis["confidence"],
            )

        # Préférence de longueur de réponse
        response_length = len(interaction_record["jeffrey_response"])
        if response_length > 300:
            length_category = "long"
        elif response_length > 150:
            length_category = "medium"
        else:
            length_category = "short"

        self._update_preference(
            user_id,
            "preferred_response_length",
            length_category,
            confidence=feedback_analysis["confidence"] * 0.8,
        )

        # Préférence de température (créativité)
        if feedback_analysis["engagement_score"] > 0.8:
            self._update_preference(
                user_id,
                "preferred_temperature",
                prompt_config["temperature"],
                confidence=feedback_analysis["confidence"] * 0.7,
            )

        # Préférence de micro-expressions
        if prompt_config["micro_expression"] and feedback_analysis["engagement_score"] > 0.6:
            self._update_preference(
                user_id,
                "liked_micro_expressions",
                prompt_config["micro_expression"],
                confidence=feedback_analysis["confidence"] * 0.6,
                is_list=True,
            )

        # Préférence de contexte émotionnel
        if feedback_analysis["engagement_score"] > 0.7:
            self._update_preference(
                user_id,
                "preferred_emotional_context",
                prompt_config["emotional_context"],
                confidence=feedback_analysis["confidence"] * 0.8,
            )

    def _update_preference(
        self,
        user_id: str,
        preference_type: str,
        value: Any,
        confidence: float,
        is_list: bool = False,
    ):
        """Met à jour une préférence utilisateur"""

        if preference_type not in self.user_preferences[user_id]:
            if is_list:
                self.user_preferences[user_id][preference_type] = UserPreference(
                    preference_type=preference_type,
                    value=[value],
                    confidence=confidence,
                    last_updated=datetime.now(),
                    interaction_count=1,
                )
            else:
                self.user_preferences[user_id][preference_type] = UserPreference(
                    preference_type=preference_type,
                    value=value,
                    confidence=confidence,
                    last_updated=datetime.now(),
                    interaction_count=1,
                )
            self.learning_stats["preferences_learned"] += 1
        else:
            existing_pref = self.user_preferences[user_id][preference_type]

            if is_list:
                # Ajouter à la liste si pas déjà présent
                if value not in existing_pref.value:
                    existing_pref.value.append(value)
                    # Garder seulement les 5 derniers
                    existing_pref.value = existing_pref.value[-5:]
            else:
                # Mettre à jour la valeur avec moyennage de confiance
                total_weight = existing_pref.confidence * existing_pref.interaction_count + confidence
                new_count = existing_pref.interaction_count + 1
                existing_pref.confidence = total_weight / new_count

                # Favoriser les nouvelles valeurs si confiance élevée
                if confidence > existing_pref.confidence * 1.2:
                    existing_pref.value = value

            existing_pref.interaction_count += 1
            existing_pref.last_updated = datetime.now()

    def get_user_preferences(self, user_id: str) -> dict[str, Any]:
        """Récupère les préférences apprises pour un utilisateur"""
        if user_id not in self.user_preferences:
            return self._get_default_preferences()

        prefs = {}
        for pref_type, pref in self.user_preferences[user_id].items():
            # Seulement retourner les préférences avec une confiance suffisante
            if pref.confidence > 0.3:
                prefs[pref_type] = pref.value

        # Compléter avec les valeurs par défaut si nécessaire
        default_prefs = self._get_default_preferences()
        for key, value in default_prefs.items():
            if key not in prefs:
                prefs[key] = value

        return prefs

    def _get_default_preferences(self) -> dict[str, Any]:
        """Préférences par défaut"""
        return {
            "response_style": "balanced",
            "preferred_response_length": "medium",
            "preferred_temperature": 0.7,
            "liked_micro_expressions": ["*présence attentive*"],
            "preferred_emotional_context": "curiosité_bienveillante",
        }

    def get_learning_recommendations(self, user_id: str) -> dict[str, Any]:
        """Génère des recommandations basées sur l'apprentissage"""
        user_prefs = self.get_user_preferences(user_id)

        # Analyser l'historique récent de l'utilisateur
        recent_interactions = [
            interaction for interaction in self.feedback_history[-50:] if interaction["user_id"] == user_id
        ]

        recommendations = {
            "prompt_adjustments": {},
            "tone_suggestions": [],
            "content_suggestions": [],
            "engagement_tips": [],
        }

        if not recent_interactions:
            return recommendations

        # Analyser les patterns de succès
        positive_interactions = [i for i in recent_interactions if i["feedback_analysis"]["type"] == "positive"]

        if positive_interactions:
            # Recommandations basées sur les succès
            successful_intents = Counter([i["prompt_config"]["intent"] for i in positive_interactions])
            most_successful = successful_intents.most_common(1)
            if most_successful:
                recommendations["content_suggestions"].append(
                    f"L'utilisateur répond bien aux intentions de type: {most_successful[0][0]}"
                )

            # Température optimale
            successful_temps = [i["prompt_config"]["temperature"] for i in positive_interactions]
            if successful_temps:
                avg_temp = sum(successful_temps) / len(successful_temps)
                recommendations["prompt_adjustments"]["temperature"] = round(avg_temp, 2)

            # Longueur optimale
            successful_lengths = [i["prompt_config"]["max_tokens"] for i in positive_interactions]
            if successful_lengths:
                avg_length = sum(successful_lengths) / len(successful_lengths)
                recommendations["prompt_adjustments"]["max_tokens"] = int(avg_length)

        # Recommandations d'engagement
        avg_engagement = sum([i["feedback_analysis"]["engagement_score"] for i in recent_interactions]) / len(
            recent_interactions
        )

        if avg_engagement < 0.5:
            recommendations["engagement_tips"] = [
                "Essayer des micro-expressions plus personnalisées",
                "Augmenter l'interactivité avec des questions",
                "Référencer plus les conversations passées",
            ]

        return recommendations

    def get_learning_stats(self) -> dict[str, Any]:
        """Retourne les statistiques d'apprentissage"""
        stats = self.learning_stats.copy()

        if stats["total_interactions"] > 0:
            stats["positive_rate"] = stats["positive_feedback"] / stats["total_interactions"]
            stats["negative_rate"] = stats["negative_feedback"] / stats["total_interactions"]
            stats["neutral_rate"] = stats["neutral_feedback"] / stats["total_interactions"]

        stats["users_with_preferences"] = len(self.user_preferences)

        # Engagement moyen récent
        recent_engagements = [
            interaction["feedback_analysis"]["engagement_score"]
            for interaction in self.feedback_history[-20:]
            if "feedback_analysis" in interaction
        ]
        if recent_engagements:
            stats["recent_avg_engagement"] = sum(recent_engagements) / len(recent_engagements)

        return stats
