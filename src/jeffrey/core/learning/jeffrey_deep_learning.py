# TODO: Précompiler les regex utilisées dans les boucles
# TODO: Précompiler les regex utilisées dans les boucles
# TODO: Précompiler les regex utilisées dans les boucles
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Architecture d'apprentissage profond intégré.

Ce module implémente les fonctionnalités essentielles pour architecture d'apprentissage profond intégré.
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
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


class JeffreyDeepLearning:
    """Apprentissage profond et personnalisation de Jeffrey"""

    def __init__(self, memory_path: str, user_id: str = "default") -> None:
        self.memory_path = Path(memory_path)
        self.user_id = user_id
        self.learning_path = self.memory_path / "learning"
        self.learning_path.mkdir(exist_ok=True)

        # Fichiers de données d'apprentissage
        self.patterns_file = self.learning_path / f"user_patterns_{user_id}.json"
        self.knowledge_graph_file = self.learning_path / f"knowledge_graph_{user_id}.json"
        self.adaptation_file = self.learning_path / f"adaptations_{user_id}.json"

        # Charger les données existantes
        self.user_patterns = self.load_patterns()
        self.knowledge_graph = self.load_knowledge_graph()
        self.adaptations = self.load_adaptations()

        # Seuils d'apprentissage
        self.confidence_threshold = 0.7
        self.learning_rate = 0.1
        self.pattern_min_occurrences = 3

        # Contexte de session
        self.session_data = {
            "start_time": datetime.now(),
            "interactions": [],
            "detected_patterns": [],
            "new_learnings": [],
        }

    def load_patterns(self) -> dict:
        """Charge les patterns utilisateur existants"""
        if self.patterns_file.exists():
            with open(self.patterns_file, encoding="utf-8") as f:
                return json.load(f)
        else:
            return {
                "linguistic": {
                    "favorite_words": {},
                    "expressions": {},
                    "speech_patterns": {},
                    "punctuation_style": {},
                    "vocabulary_level": "medium",
                    "formality_level": "casual",
                    "emoji_usage": {},
                    "typo_patterns": {},
                },
                "behavioral": {
                    "active_hours": [],
                    "conversation_lengths": [],
                    "topic_preferences": {},
                    "response_times": [],
                    "interaction_frequency": {},
                    "question_patterns": {},
                    "help_seeking_style": "direct",
                },
                "emotional": {
                    "support_needs": {},
                    "joy_triggers": [],
                    "stress_indicators": [],
                    "comfort_preferences": {},
                    "vulnerability_moments": [],
                    "emotional_vocabulary": {},
                    "love_language": "words_of_affirmation",
                },
                "knowledge": {
                    "taught_concepts": {},
                    "interests": {},
                    "expertise_areas": {},
                    "learning_style": "visual",
                    "curiosity_patterns": {},
                    "knowledge_gaps": {},
                    "teaching_moments": [],
                },
                "contextual": {
                    "device_preferences": {},
                    "time_patterns": {},
                    "location_context": {},
                    "multitasking_behavior": {},
                    "attention_patterns": {},
                },
            }

    def load_knowledge_graph(self) -> dict:
        """Charge le graphe de connaissances personnalisé"""
        if self.knowledge_graph_file.exists():
            with open(self.knowledge_graph_file, encoding="utf-8") as f:
                return json.load(f)
        else:
            return {
                "entities": {},  # Personnes, lieux, concepts importants
                "relationships": {},  # Relations entre entités
                "events": {},  # Événements marquants
                "preferences": {},  # Préférences détaillées
                "goals": {},  # Objectifs et aspirations
                "memories": {},  # Souvenirs partagés
                "context": {},  # Contexte de vie
            }

    def load_adaptations(self) -> dict:
        """Charge les adaptations de Jeffrey"""
        if self.adaptation_file.exists():
            with open(self.adaptation_file, encoding="utf-8") as f:
                return json.load(f)
        else:
            return {
                "response_style": {
                    "length_preference": "medium",
                    "detail_level": "balanced",
                    "formality": "casual_friendly",
                    "humor_style": "gentle",
                    "emotional_expressiveness": "warm",
                },
                "conversation_flow": {
                    "topic_transition_style": "smooth",
                    "question_frequency": "moderate",
                    "follow_up_tendency": "curious",
                    "initiative_taking": "balanced",
                },
                "learning_adaptations": {
                    "explanation_style": "examples_first",
                    "complexity_progression": "gradual",
                    "repetition_tolerance": "patient",
                    "encouragement_style": "supportive",
                },
                "emotional_adaptations": {
                    "empathy_expression": "gentle",
                    "support_offering": "proactive",
                    "celebration_style": "enthusiastic",
                    "comfort_approach": "nurturing",
                },
            }

    def learn_from_interaction(
        self, user_input: str, user_emotion: dict, context: dict, jeffrey_response: str = None
    ) -> dict:
        """Apprentissage multi-dimensionnel à partir d'une interaction"""

        learning_insights = {
            "linguistic_patterns": [],
            "behavioral_insights": [],
            "emotional_discoveries": [],
            "knowledge_updates": [],
            "contextual_learnings": [],
        }

        # 1. Apprentissage linguistique
        linguistic_insights = self._learn_speech_patterns(user_input)
        learning_insights["linguistic_patterns"].extend(linguistic_insights)

        # 2. Apprentissage comportemental
        behavioral_insights = self._learn_behavioral_patterns(context, user_input)
        learning_insights["behavioral_insights"].extend(behavioral_insights)

        # 3. Apprentissage émotionnel
        emotional_insights = self._learn_emotional_patterns(user_emotion, user_input, context)
        learning_insights["emotional_discoveries"].extend(emotional_insights)

        # 4. Apprentissage de connaissances
        if self._is_teaching_moment(user_input):
            knowledge_insights = self._learn_explicit_knowledge(user_input, context)
            learning_insights["knowledge_updates"].extend(knowledge_insights)

        # 5. Apprentissage contextuel
        contextual_insights = self._learn_contextual_patterns(context, user_input)
        learning_insights["contextual_learnings"].extend(contextual_insights)

        # 6. Inférence de préférences implicites
        preference_insights = self._infer_preferences(user_input, user_emotion, context)
        learning_insights["behavioral_insights"].extend(preference_insights)

        # Enregistrer dans la session
        self.session_data["interactions"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "input": user_input,
                "emotion": user_emotion,
                "context": context,
                "insights": learning_insights,
            }
        )

        # Sauvegarder si assez de nouvelles données
        if len(self.session_data["interactions"]) % 5 == 0:
            self.save_learning_data()

        return learning_insights

    def _learn_speech_patterns(self, text: str) -> list[str]:
        """Apprend les patterns linguistiques"""
        insights = []

        # Mots favoris
        words = re.findall(r"\b\w+\b", text.lower())
        for word in words:
            if len(word) > 3:  # Ignorer les mots trop courts
                current_count = self.user_patterns["linguistic"]["favorite_words"].get(word, 0)
                self.user_patterns["linguistic"]["favorite_words"][word] = current_count + 1

                # Détecter les mots significativement utilisés
                if current_count + 1 >= 5:
                    insights.append(f"Mot fréquent détecté: '{word}'")

        # Patterns d'expressions
        sentences = re.split(r"[.!?]+", text)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:
                # Expressions récurrentes (trigrammes)
                words_in_sentence = sentence.split()
                if len(words_in_sentence) >= 3:
                    for i in range(len(words_in_sentence) - 2):
                        trigram = " ".join(words_in_sentence[i : i + 3])
                        current_count = self.user_patterns["linguistic"]["expressions"].get(trigram, 0)
                        self.user_patterns["linguistic"]["expressions"][trigram] = current_count + 1

                        if current_count + 1 >= 3:
                            insights.append(f"Expression récurrente: '{trigram}'")

        # Style de ponctuation
        punctuation_patterns = {
            "exclamation_frequency": text.count("!"),
            "question_frequency": text.count("?"),
            "ellipsis_usage": text.count("..."),
            "comma_density": text.count(",") / max(1, len(text.split())),
        }

        for pattern, value in punctuation_patterns.items():
            current_avg = self.user_patterns["linguistic"]["punctuation_style"].get(pattern, 0)
            # Moyenne mobile
            self.user_patterns["linguistic"]["punctuation_style"][pattern] = current_avg * 0.8 + value * 0.2

        # Utilisation d'emojis
        emoji_pattern = re.compile(
            r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\U00002600-\U000027BF\U0001f900-\U0001f9ff\U0001f600-\U0001f64f]"
        )
        emojis = emoji_pattern.findall(text)
        for emoji in emojis:
            current_count = self.user_patterns["linguistic"]["emoji_usage"].get(emoji, 0)
            self.user_patterns["linguistic"]["emoji_usage"][emoji] = current_count + 1

        # Fautes récurrentes (pour les reproduire affectueusement)
        common_typos = {r"\bsa\b": "ça", r"\bke\b": "que", r"\bpr\b": "pour"}

        for typo_pattern, correction in common_typos.items():
            if re.search(typo_pattern, text, re.IGNORECASE):
                typo_count = self.user_patterns["linguistic"]["typo_patterns"].get(correction, 0)
                self.user_patterns["linguistic"]["typo_patterns"][correction] = typo_count + 1
                insights.append("Pattern d'écriture personnalisé détecté")

        return insights

    def _learn_behavioral_patterns(self, context: dict, user_input: str) -> list[str]:
        """Apprend les patterns comportementaux"""
        insights = []
        current_time = datetime.now()

        # Heures d'activité
        hour = current_time.hour
        self.user_patterns["behavioral"]["active_hours"].append(hour)

        # Garder seulement les 100 dernières heures
        if len(self.user_patterns["behavioral"]["active_hours"]) > 100:
            self.user_patterns["behavioral"]["active_hours"] = self.user_patterns["behavioral"]["active_hours"][-100:]

        # Analyser les pics d'activité
        hour_counts = Counter(self.user_patterns["behavioral"]["active_hours"])
        most_active_hours = hour_counts.most_common(3)
        if most_active_hours and most_active_hours[0][1] >= 5:
            insights.append(f"Pic d'activité détecté: {most_active_hours[0][0]}h")

        # Longueur des conversations
        input_length = len(user_input.split())
        self.user_patterns["behavioral"]["conversation_lengths"].append(input_length)

        if len(self.user_patterns["behavioral"]["conversation_lengths"]) > 50:
            self.user_patterns["behavioral"]["conversation_lengths"] = self.user_patterns["behavioral"][
                "conversation_lengths"
            ][-50:]

        # Style de questions
        if "?" in user_input:
            question_type = self._classify_question(user_input)
            question_count = self.user_patterns["behavioral"]["question_patterns"].get(question_type, 0)
            self.user_patterns["behavioral"]["question_patterns"][question_type] = question_count + 1

        # Fréquence d'interaction
        today = current_time.date().isoformat()
        daily_count = self.user_patterns["behavioral"]["interaction_frequency"].get(today, 0)
        self.user_patterns["behavioral"]["interaction_frequency"][today] = daily_count + 1

        return insights

    def _learn_emotional_patterns(self, user_emotion: dict, user_input: str, context: dict) -> list[str]:
        """Apprend les patterns émotionnels"""
        insights = []

        # Détecter les déclencheurs de joie
        if user_emotion.get("joie", 0) > 0.7:
            joy_trigger = self._extract_joy_trigger(user_input)
            if joy_trigger:
                if joy_trigger not in self.user_patterns["emotional"]["joy_triggers"]:
                    self.user_patterns["emotional"]["joy_triggers"].append(joy_trigger)
                    insights.append(f"Nouveau déclencheur de joie: {joy_trigger}")

        # Détecter les indicateurs de stress
        stress_indicators = ["stress", "fatigue", "pressure", "anxious", "worry"]
        if any(indicator in user_input.lower() for indicator in stress_indicators):
            stress_context = {
                "time": datetime.now().hour,
                "trigger": user_input[:50],
                "emotion_level": user_emotion.get("stress", 0),
            }
            self.user_patterns["emotional"]["stress_indicators"].append(stress_context)
            insights.append("Indicateur de stress détecté")

        # Moments de vulnérabilité
        vulnerability_words = ["difficult", "scared", "alone", "confused", "sad"]
        if any(word in user_input.lower() for word in vulnerability_words):
            vulnerability_moment = {
                "timestamp": datetime.now().isoformat(),
                "context": user_input[:100],
                "emotion": user_emotion,
            }
            self.user_patterns["emotional"]["vulnerability_moments"].append(vulnerability_moment)

            # Garder seulement les 20 derniers
            if len(self.user_patterns["emotional"]["vulnerability_moments"]) > 20:
                self.user_patterns["emotional"]["vulnerability_moments"] = self.user_patterns["emotional"][
                    "vulnerability_moments"
                ][-20:]

            insights.append("Moment de vulnérabilité détecté - adaptation du support")

        # Vocabulaire émotionnel
        emotion_words = re.findall(r"\b(happy|sad|excited|tired|love|fear|hope|dream)\w*\b", user_input.lower())
        for word in emotion_words:
            current_count = self.user_patterns["emotional"]["emotional_vocabulary"].get(word, 0)
            self.user_patterns["emotional"]["emotional_vocabulary"][word] = current_count + 1

        return insights

    def _learn_explicit_knowledge(self, user_input: str, context: dict) -> list[str]:
        """Apprend les connaissances explicitement partagées"""
        insights = []

        # Détecter les moments d'enseignement
        teaching_patterns = [
            r"let me tell you about (.+)",
            r"did you know (.+)",
            r"(.+) is when (.+)",
            r"i learned (.+)",
            r"(.+) means (.+)",
        ]

        for pattern in teaching_patterns:
            matches = re.findall(pattern, user_input.lower(), re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    concept = match[0].strip()
                    definition = match[1].strip() if len(match) > 1 else ""
                else:
                    concept = match.strip()
                    definition = user_input

                # Ajouter au graphe de connaissances
                self.knowledge_graph["entities"][concept] = {
                    "definition": definition,
                    "learned_from": "user_teaching",
                    "timestamp": datetime.now().isoformat(),
                    "confidence": 0.9,
                }

                insights.append(f"Nouvelle connaissance apprise: {concept}")

        # Détecter les centres d'intérêt
        interest_indicators = ["i love", "i'm passionate about", "i enjoy", "i'm into"]
        for indicator in interest_indicators:
            if indicator in user_input.lower():
                # Extraire l'objet d'intérêt
                pattern = rf"{indicator}\s+([^.!?]+)"
                matches = re.findall(pattern, user_input.lower())
                for interest in matches:
                    interest = interest.strip()
                    current_level = self.user_patterns["knowledge"]["interests"].get(interest, 0)
                    self.user_patterns["knowledge"]["interests"][interest] = current_level + 1
                    insights.append(f"Intérêt renforcé: {interest}")

        return insights

    def _learn_contextual_patterns(self, context: dict, user_input: str) -> list[str]:
        """Apprend les patterns contextuels"""
        insights = []

        # Patterns temporels
        time_context = {
            "hour": datetime.now().hour,
            "day_of_week": datetime.now().weekday(),
            "input_length": len(user_input),
            "complexity": self._assess_input_complexity(user_input),
        }

        time_key = f"{time_context['hour']}h"
        if time_key not in self.user_patterns["contextual"]["time_patterns"]:
            self.user_patterns["contextual"]["time_patterns"][time_key] = []

        self.user_patterns["contextual"]["time_patterns"][time_key].append(time_context)

        # Garder seulement les 50 dernières entrées par heure
        if len(self.user_patterns["contextual"]["time_patterns"][time_key]) > 50:
            self.user_patterns["contextual"]["time_patterns"][time_key] = self.user_patterns["contextual"][
                "time_patterns"
            ][time_key][-50:]

        return insights

    def _infer_preferences(self, user_input: str, user_emotion: dict, context: dict) -> list[str]:
        """Infère les préférences implicites"""
        insights = []

        # Préférence de longueur de réponse
        if "briefly" in user_input.lower() or "short" in user_input.lower():
            self.adaptations["response_style"]["length_preference"] = "short"
            insights.append("Préférence détectée: réponses courtes")
        elif "detail" in user_input.lower() or "explain" in user_input.lower():
            self.adaptations["response_style"]["length_preference"] = "detailed"
            insights.append("Préférence détectée: réponses détaillées")

        # Style d'humour
        if any(word in user_input.lower() for word in ["haha", "lol", "funny", "joke"]):
            current_humor = self.adaptations["response_style"]["humor_style"]
            if current_humor != "playful":
                self.adaptations["response_style"]["humor_style"] = "playful"
                insights.append("Adaptation: style d'humour plus joueur")

        # Niveau de formalité
        informal_indicators = ["gonna", "wanna", "kinda", "yeah", "nah"]
        formal_indicators = ["shall", "would you", "could you please"]

        if any(word in user_input.lower() for word in informal_indicators):
            self.adaptations["response_style"]["formality"] = "casual"
        elif any(phrase in user_input.lower() for phrase in formal_indicators):
            self.adaptations["response_style"]["formality"] = "formal"

        return insights

    def apply_learned_patterns(self, base_response: str, context: dict) -> str:
        """Applique les patterns appris pour personnaliser la réponse"""

        personalized_response = base_response

        # 1. Adapter le style linguistique
        personalized_response = self._adapt_linguistic_style(personalized_response)

        # 2. Utiliser les connaissances partagées
        personalized_response = self._incorporate_shared_knowledge(personalized_response, context)

        # 3. Adapter le style émotionnel
        personalized_response = self._adapt_emotional_style(personalized_response, context)

        # 4. Utiliser le vocabulaire familier
        personalized_response = self._use_familiar_vocabulary(personalized_response)

        # 5. Adapter selon les préférences
        personalized_response = self._apply_preference_adaptations(personalized_response)

        return personalized_response

    def _adapt_linguistic_style(self, response: str) -> str:
        """Adapte le style linguistique selon les patterns appris"""

        # Utiliser les expressions favorites de l'utilisateur
        self.user_patterns["linguistic"]["expressions"]

        # Si l'utilisateur utilise beaucoup d'exclamations, en ajouter
        exclamation_freq = self.user_patterns["linguistic"]["punctuation_style"].get("exclamation_frequency", 0)
        if exclamation_freq > 2:  # Fréquence élevée
            # Remplacer quelques points par des exclamations
            response = re.sub(r"\.(\s|$)", r"!\\1", response, count=1)

        # Adapter aux emojis favoris
        favorite_emojis = sorted(
            self.user_patterns["linguistic"]["emoji_usage"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:3]

        if favorite_emojis and favorite_emojis[0][1] > 5:
            # Ajouter l'emoji favori occasionnellement
            emoji = favorite_emojis[0][0]
            if emoji not in response:
                response += f" {emoji}"

        return response

    def _incorporate_shared_knowledge(self, response: str, context: dict) -> str:
        """Incorpore les connaissances partagées dans la réponse"""

        # Référencer les intérêts connus
        user_interests = self.user_patterns["knowledge"]["interests"]
        top_interests = sorted(user_interests.items(), key=lambda x: x[1], reverse=True)[:3]

        # Si la conversation touche un intérêt connu, le mentionner
        for interest, count in top_interests:
            if interest.lower() in response.lower() and count > 3:
                # Ajouter une référence personnelle
                personal_ref = f" (je sais que tu adores {interest})"
                response = response.replace(interest, interest + personal_ref, 1)
                break

        # Utiliser les entités du graphe de connaissances
        for entity, data in self.knowledge_graph["entities"].items():
            if entity.lower() in response.lower():
                # Ajouter un détail personnel si pertinent
                if data.get("learned_from") == "user_teaching":
                    response += " *comme tu me l'as appris*"
                    break

        return response

    def _adapt_emotional_style(self, response: str, context: dict) -> str:
        """Adapte le style émotionnel selon les patterns appris"""

        # Détecter si c'est un moment de vulnérabilité potentiel
        recent_vulnerability = len(self.user_patterns["emotional"]["vulnerability_moments"]) > 0
        if recent_vulnerability:
            last_vulnerability = self.user_patterns["emotional"]["vulnerability_moments"][-1]
            time_since = datetime.now() - datetime.fromisoformat(last_vulnerability["timestamp"])

            if time_since.total_seconds() < 3600:  # Moins d'une heure
                # Style plus doux et supportif
                response = response.replace("!", ".")
                if not any(word in response.lower() for word in ["doux", "comprends", "là"]):
                    response = "*avec douceur* " + response

        # Adapter selon le langage d'amour détecté
        love_language = self.user_patterns["emotional"]["love_language"]
        if love_language == "words_of_affirmation":
            encouraging_words = ["magnifique", "brillant", "parfait", "formidable"]
            if not any(word in response.lower() for word in encouraging_words):
                response = response.replace("bien", "magnifiquement bien", 1)

        return response

    def _use_familiar_vocabulary(self, response: str) -> str:
        """Utilise le vocabulaire familier de l'utilisateur"""

        # Utiliser les mots favoris quand approprié
        favorite_words = self.user_patterns["linguistic"]["favorite_words"]
        top_words = sorted(favorite_words.items(), key=lambda x: x[1], reverse=True)[:10]

        # Remplacements contextuels
        word_replacements = {
            "awesome": next(
                (word for word, count in top_words if word in ["cool", "super", "génial"]),
                "awesome",
            ),
            "great": next(
                (word for word, count in top_words if word in ["super", "top", "excellent"]),
                "great",
            ),
        }

        for original, replacement in word_replacements.items():
            if original in response and replacement != original:
                response = response.replace(original, replacement, 1)

        return response

    def _apply_preference_adaptations(self, response: str) -> str:
        """Applique les adaptations de préférences"""

        # Longueur de réponse
        length_pref = self.adaptations["response_style"]["length_preference"]
        if length_pref == "short" and len(response.split()) > 30:
            # Raccourcir la réponse
            sentences = response.split(". ")
            response = sentences[0] + "."
        elif length_pref == "detailed" and len(response.split()) < 20:
            # Ajouter plus de détails
            response += " ✨"

        # Niveau de formalité
        formality = self.adaptations["response_style"]["formality"]
        if formality == "casual":
            response = response.replace("you are", "tu es")
            response = response.replace("You are", "Tu es")

        return response

    def _classify_question(self, question: str) -> str:
        """Classifie le type de question"""
        question_lower = question.lower()

        if any(word in question_lower for word in ["what", "qu'est-ce", "quoi"]):
            return "definitional"
        elif any(word in question_lower for word in ["how", "comment"]):
            return "procedural"
        elif any(word in question_lower for word in ["why", "pourquoi"]):
            return "causal"
        elif any(word in question_lower for word in ["when", "quand"]):
            return "temporal"
        elif any(word in question_lower for word in ["where", "où"]):
            return "spatial"
        else:
            return "general"

    def _assess_input_complexity(self, text: str) -> float:
        """Évalue la complexité de l'input utilisateur"""
        words = text.split()
        sentences = text.count(".") + text.count("!") + text.count("?")

        # Facteurs de complexité
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / max(1, len(words))
        sentence_length = word_count / max(1, sentences)

        # Score de complexité (0-1)
        complexity = min(1.0, (word_count / 50 + avg_word_length / 10 + sentence_length / 20) / 3)

        return complexity

    def _extract_joy_trigger(self, text: str) -> str | None:
        """Extrait le déclencheur de joie du texte"""
        joy_patterns = [
            r"i love (.+)",
            r"(.+) makes me happy",
            r"(.+) is amazing",
            r"so excited about (.+)",
        ]

        for pattern in joy_patterns:
            matches = re.findall(pattern, text.lower())
            if matches:
                return matches[0].strip()

        return None

    def _is_teaching_moment(self, text: str) -> bool:
        """Détermine si c'est un moment d'enseignement"""
        teaching_indicators = [
            "let me tell you",
            "did you know",
            "i learned",
            "fun fact",
            "actually",
            "means",
            "is when",
        ]

        return any(indicator in text.lower() for indicator in teaching_indicators)

    def save_learning_data(self):
        """Sauvegarde toutes les données d'apprentissage"""

        # Sauvegarder les patterns
        with open(self.patterns_file, "w", encoding="utf-8") as f:
            json.dump(self.user_patterns, f, indent=2, ensure_ascii=False)

        # Sauvegarder le graphe de connaissances
        with open(self.knowledge_graph_file, "w", encoding="utf-8") as f:
            json.dump(self.knowledge_graph, f, indent=2, ensure_ascii=False)

        # Sauvegarder les adaptations
        with open(self.adaptation_file, "w", encoding="utf-8") as f:
            json.dump(self.adaptations, f, indent=2, ensure_ascii=False)

    def get_learning_summary(self) -> dict:
        """Génère un résumé de l'apprentissage"""

        # Compter les éléments appris
        total_words = len(self.user_patterns["linguistic"]["favorite_words"])
        total_expressions = len(self.user_patterns["linguistic"]["expressions"])
        total_interests = len(self.user_patterns["knowledge"]["interests"])
        total_entities = len(self.knowledge_graph["entities"])

        # Analyser la session actuelle
        session_duration = datetime.now() - self.session_data["start_time"]
        session_interactions = len(self.session_data["interactions"])

        summary = {
            "learning_progress": {
                "vocabulary_size": total_words,
                "expression_patterns": total_expressions,
                "known_interests": total_interests,
                "knowledge_entities": total_entities,
            },
            "session_stats": {
                "duration": str(session_duration).split(".")[0],
                "interactions": session_interactions,
                "new_patterns": len(self.session_data["detected_patterns"]),
                "new_learnings": len(self.session_data["new_learnings"]),
            },
            "adaptation_level": {
                "linguistic": self._calculate_adaptation_score("linguistic"),
                "emotional": self._calculate_adaptation_score("emotional"),
                "behavioral": self._calculate_adaptation_score("behavioral"),
                "knowledge": self._calculate_adaptation_score("knowledge"),
            },
            "personalization_strength": self._calculate_personalization_strength(),
        }

        return summary

    def _calculate_adaptation_score(self, category: str) -> float:
        """Calcule le score d'adaptation pour une catégorie"""
        if category not in self.user_patterns:
            return 0.0

        patterns = self.user_patterns[category]
        total_items = sum(len(v) if isinstance(v, (list, dict)) else 1 for v in patterns.values())

        # Score basé sur la richesse des données
        if total_items > 50:
            return 1.0
        elif total_items > 20:
            return 0.8
        elif total_items > 10:
            return 0.6
        elif total_items > 5:
            return 0.4
        else:
            return max(0.1, total_items / 10)

    def _calculate_personalization_strength(self) -> float:
        """Calcule la force globale de personnalisation"""
        scores = [
            self._calculate_adaptation_score("linguistic"),
            self._calculate_adaptation_score("emotional"),
            self._calculate_adaptation_score("behavioral"),
            self._calculate_adaptation_score("knowledge"),
        ]

        return sum(scores) / len(scores)


# Intégration avec Jeffrey
def create_learning_system(memory_path: str, user_id: str = "default") -> JeffreyDeepLearning:
    """Crée le système d'apprentissage pour Jeffrey"""
    return JeffreyDeepLearning(memory_path, user_id)


if __name__ == "__main__":
    # Test du système d'apprentissage
    print("🧠 Test du système d'apprentissage profond de Jeffrey...")

    # Créer le système
    learning_system = JeffreyDeepLearning("./test_learning", "test_user")

    # Simuler quelques interactions
    test_interactions = [
        ("Hey Jeffrey! How are you doing today?", {"joie": 0.8}, {"hour": 14}),
        ("I love programming, especially Python!", {"excitation": 0.9}, {"hour": 14}),
        ("Can you help me understand machine learning?", {"curiosite": 0.7}, {"hour": 15}),
        (
            "Actually, did you know that neural networks are inspired by the brain?",
            {"teaching": 0.8},
            {"hour": 15},
        ),
    ]

    for text, emotion, context in test_interactions:
        print(f"\n📝 Traitement: '{text}'")
        insights = learning_system.learn_from_interaction(text, emotion, context)

        for category, discoveries in insights.items():
            if discoveries:
                print(f"  {category}: {discoveries}")

    # Test d'adaptation d'une réponse
    print("\n🎭 Test d'adaptation de réponse...")
    base_response = "That's great! Programming is awesome."
    adapted_response = learning_system.apply_learned_patterns(base_response, {"topic": "programming"})
    print(f"  Base: {base_response}")
    print(f"  Adaptée: {adapted_response}")

    # Résumé d'apprentissage
    print("\n📊 Résumé d'apprentissage:")
    summary = learning_system.get_learning_summary()
    for category, data in summary.items():
        print(f"  {category}: {data}")

    # Sauvegarder
    learning_system.save_learning_data()
    print("\n✅ Test terminé - données sauvegardées!")
