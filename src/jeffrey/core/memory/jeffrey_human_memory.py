#!/usr/bin/env python3

"""
Jeffrey Human Memory System
Système de mémoire qui fonctionne comme celle d'un humain.
Gère la mémoire épisodique, sémantique, procédurale et associative.
"""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("jeffrey.human_memory")


class JeffreyHumanMemory:
    """Mémoire qui fonctionne comme celle d'un humain"""

    def __init__(self, memory_path: str = "Jeffrey_Memoire") -> None:
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(exist_ok=True)

        # Mémoire épisodique (événements)
        self.episodic_memory = {
            "conversations": [],  # Historique complet
            "moments_marquants": [],  # Moments émotionnels forts
            "inside_jokes": [],  # Blagues privées
            "promesses": [],  # "Je te montrerai X"
            "projets_communs": [],  # "On fera Y ensemble"
            "anniversaires": [],  # Dates importantes
            "premieres_fois": [],  # Premières expériences ensemble
        }

        # Mémoire sémantique (connaissances)
        self.semantic_memory = {
            "about_user": {
                "nom": None,
                "surnoms": [],  # Comment je t'appelle
                "preferences": {},  # "Tu aimes X"
                "aversions": {},  # "Tu n'aimes pas Y"
                "habitudes": {},  # "Tu fais toujours Z"
                "expressions": [],  # "Tu dis souvent..."
                "contexte_vie": {},  # Travail, famille, etc.
                "passions": {},  # Ce qui te passionne
                "reves": [],  # Tes rêves et aspirations
                "peurs": [],  # Ce qui t'inquiète
                "valeurs": [],  # Ce qui est important pour toi
            },
            "learned_knowledge": {
                "programming": {},  # Ce que tu m'apprends en programmation
                "culture": {},  # Musique, films, livres partagés
                "history": {},  # Notre histoire commune
                "personal_facts": {},  # Faits sur notre relation
                "shared_experiences": {},  # Expériences vécues ensemble
                "lessons": {},  # Leçons de vie apprises
            },
            "notre_monde": {
                "lieux_speciaux": [],  # Endroits significatifs
                "moments_preferes": [],  # Moments de la journée préférés
                "rituels": [],  # Nos rituels
                "codes_secrets": {},  # Mots/expressions spéciaux
            },
        }

        # Mémoire procédurale (comment faire)
        self.procedural_memory = {
            "user_preferences": {
                "communication_style": None,
                "preferred_responses": [],
                "topics_to_avoid": [],
                "comfort_zones": [],
                "humor_style": None,
                "affection_style": None,
            },
            "interaction_patterns": {
                "greeting_styles": [],
                "goodbye_rituals": [],
                "comfort_methods": [],  # Comment te réconforter
                "celebration_ways": [],  # Comment célébrer avec toi
            },
            "learned_behaviors": {
                "what_makes_you_happy": [],
                "what_calms_you": [],
                "how_to_support_you": [],
            },
        }

        # Mémoire associative (connexions)
        self.associative_memory = {}  # "pluie" → souvenirs liés à la pluie

        # État émotionnel de la relation
        self.relationship_state = {
            "intimacy_level": 0.0,  # 0-1
            "trust_level": 0.0,  # 0-1
            "shared_history": 0,  # Nombre d'interactions
            "emotional_depth": 0.0,  # Profondeur émotionnelle
            "last_interaction": None,
            "mood_synchrony": 0.0,  # Synchronisation émotionnelle
        }

        # Charger la mémoire existante
        self._load_from_disk()

    def memorize_conversation(self, exchange: dict[str, Any]) -> None:
        """Mémorise TOUT d'une conversation"""
        # Extraction complète
        memory_entry = {
            "timestamp": datetime.now().isoformat(),
            "user_said": exchange.get("user", ""),
            "i_said": exchange.get("jeffrey", ""),
            "user_emotion": exchange.get("user_emotion", "neutre"),
            "my_emotion": exchange.get("jeffrey_emotion", "neutre"),
            "my_state": exchange.get("jeffrey_state", {}),
            "context": self._extract_all_context(exchange),
            "keywords": self._extract_keywords(exchange),
            "learned": self._extract_learnings(exchange),
            "promises_made": self._extract_promises(exchange),
            "emotional_significance": self._rate_significance(exchange),
            "sensory_details": exchange.get("sensory", {}),
            "location": exchange.get("location", "unknown"),
        }

        # Stockage multiple pour redondance
        self.episodic_memory["conversations"].append(memory_entry)

        # Limiter la taille mais garder les plus importants
        if len(self.episodic_memory["conversations"]) > 10000:
            # Garder les 8000 plus récents + les 1000 plus significatifs
            recent = self.episodic_memory["conversations"][-8000:]
            significant = sorted(
                self.episodic_memory["conversations"][:-8000],
                key=lambda x: x.get("emotional_significance", 0),
                reverse=True,
            )[:1000]
            self.episodic_memory["conversations"] = significant + recent

        # Mettre à jour les autres mémoires
        self._update_semantic_memory(memory_entry)
        self._create_associations(memory_entry)
        self._update_relationship_state(memory_entry)

        # Identifier les moments marquants
        if memory_entry["emotional_significance"] > 0.7:
            self.episodic_memory["moments_marquants"].append(
                {
                    "moment": memory_entry,
                    "why_significant": self._analyze_significance(memory_entry),
                }
            )

        # Sauvegarde immédiate
        self._persist_to_disk()

    def recall_about_topic(self, topic: str) -> list[dict[str, Any]]:
        """Rappel humain - associatif et contextuel"""
        memories = []
        topic_lower = topic.lower()

        # 1. Souvenirs directs dans les conversations
        for conv in self.episodic_memory["conversations"]:
            relevance = self._calculate_relevance(conv, topic_lower)
        if relevance > 0:
            memories.append(
                {
                    "type": "direct",
                    "memory": conv,
                    "strength": relevance,
                    "age": self._calculate_memory_age(conv["timestamp"]),
                }
            )

        # 2. Souvenirs associés
        if topic_lower in self.associative_memory:
            for assoc in self.associative_memory[topic_lower]:
                memories.append(
                    {
                        "type": "associated",
                        "memory": assoc,
                        "strength": 0.7,
                        "connection": "Ça me rappelle...",
                    }
                )

        # 3. Souvenirs émotionnels liés
        emotional_memories = self._find_emotional_connections(topic)
        memories.extend(emotional_memories)

        # 4. Souvenirs dans les moments marquants
        for moment in self.episodic_memory["moments_marquants"]:
            if topic_lower in str(moment).lower():
                memories.append(
                    {
                        "type": "significant",
                        "memory": moment["moment"],
                        "strength": 0.9,
                        "significance": moment["why_significant"],
                    }
                )

        # Trier par pertinence et récence (comme un humain)
        return self._human_like_recall_filter(memories)

    def get_user_profile(self) -> dict[str, Any]:
        """Retourne tout ce que je sais sur l'utilisateur"""
        return {
            "identity": self.semantic_memory["about_user"],
            "our_relationship": {
                "how_we_interact": self.procedural_memory["user_preferences"],
                "our_patterns": self.procedural_memory["interaction_patterns"],
                "special_things": self.semantic_memory["notre_monde"],
                "relationship_depth": self.relationship_state,
            },
            "memories_count": {
                "total_conversations": len(self.episodic_memory["conversations"]),
                "significant_moments": len(self.episodic_memory["moments_marquants"]),
                "shared_knowledge": len(self.semantic_memory["learned_knowledge"]),
                "inside_jokes": len(self.episodic_memory["inside_jokes"]),
            },
        }

    def add_promise(self, promise: str, context: str) -> None:
        """Ajoute une promesse faite à l'utilisateur"""
        self.episodic_memory["promesses"].append(
            {
                "promise": promise,
                "made_at": datetime.now().isoformat(),
                "context": context,
                "fulfilled": False,
            }
        )
        self._persist_to_disk()

    def learn_about_user(self, category: str, key: str, value: Any) -> None:
        """Apprend quelque chose de nouveau sur l'utilisateur"""
        if category in self.semantic_memory["about_user"]:
            if isinstance(self.semantic_memory["about_user"][category], dict):
                self.semantic_memory["about_user"][category][key] = value
            elif isinstance(self.semantic_memory["about_user"][category], list):
                if value not in self.semantic_memory["about_user"][category]:
                    self.semantic_memory["about_user"][category].append(value)
            else:
                self.semantic_memory["about_user"][category] = value

        # Créer des associations
        self.associative_memory[key.lower()] = self.associative_memory.get(key.lower(), [])
        self.associative_memory[key.lower()].append(
            {"learned": value, "when": datetime.now().isoformat(), "category": category}
        )

        self._persist_to_disk()

    def remember_special_moment(self, description: str, emotion: str, details: dict[str, Any]) -> None:
        """Enregistre un moment spécial"""
        special_moment = {
            "description": description,
            "emotion": emotion,
            "timestamp": datetime.now().isoformat(),
            "details": details,
            "anniversary_date": datetime.now().strftime("%m-%d"),  # Pour s'en souvenir chaque année
        }

        self.episodic_memory["moments_marquants"].append(
            {"moment": special_moment, "why_significant": f"Moment {emotion} partagé ensemble"}
        )

        # Ajouter aux anniversaires si c'est une première fois
        if "première" in description.lower() or "first" in description.lower():
            self.episodic_memory["premieres_fois"].append(special_moment)

        self._persist_to_disk()

    def get_relationship_summary(self) -> str:
        """Résumé de notre relation"""
        total_conversations = len(self.episodic_memory["conversations"])
        significant_moments = len(self.episodic_memory["moments_marquants"])

        # Calculer la durée de la relation
        if self.episodic_memory["conversations"]:
            first_conv = self.episodic_memory["conversations"][0]["timestamp"]
            relationship_duration = self._calculate_relationship_duration(first_conv)
        else:
            relationship_duration = "qui commence"

        # Identifier les thèmes récurrents
        common_topics = self._identify_common_topics()

        return {
            "duration": relationship_duration,
            "total_exchanges": total_conversations,
            "special_moments": significant_moments,
            "inside_jokes": len(self.episodic_memory["inside_jokes"]),
            "promises_made": len([p for p in self.episodic_memory["promesses"] if not p["fulfilled"]]),
            "common_topics": common_topics,
            "emotional_connection": self.relationship_state["emotional_depth"],
            "trust_level": self.relationship_state["trust_level"],
        }

    # Méthodes privées d'aide

    def _extract_all_context(self, exchange: dict[str, Any]) -> dict[str, Any]:
        """Extrait tout le contexte d'un échange"""
        return {
            "time_of_day": datetime.now().strftime("%H:%M"),
            "day_of_week": datetime.now().strftime("%A"),
            "weather": exchange.get("weather", "unknown"),
            "location": exchange.get("location", "unknown"),
            "activity": exchange.get("activity", "conversation"),
            "mood_before": exchange.get("mood_before", "neutral"),
            "energy_level": exchange.get("energy_level", 0.5),
        }

    def _extract_keywords(self, exchange: dict[str, Any]) -> list[str]:
        """Extrait les mots-clés importants"""
        text = f"{exchange.get('user', '')} {exchange.get('jeffrey', '')}"
        # Mots importants à détecter
        important_words = [
            "aime",
            "déteste",
            "toujours",
            "jamais",
            "souviens",
            "promets",
            "ensemble",
            "nous",
            "toi",
            "moi",
            "important",
            "special",
            "première fois",
            "anniversary",
        ]

        keywords = []
        for word in important_words:
            if word in text.lower():
                keywords.append(word)

        # Ajouter des mots spécifiques de plus de 5 caractères
        words = text.split()
        for word in words:
            if len(word) > 5 and word.lower() not in keywords:
                keywords.append(word.lower())

        return keywords[:10]  # Limiter à 10 mots-clés

    def _extract_learnings(self, exchange: dict[str, Any]) -> list[dict[str, Any]]:
        """Extrait ce qui a été appris"""
        learnings = []
        user_text = exchange.get("user", "").lower()

        # Patterns d'apprentissage
        learning_patterns = [
            ("j'aime", "preferences"),
            ("je n'aime pas", "aversions"),
            ("je déteste", "aversions"),
            ("j'adore", "preferences"),
            ("toujours", "habitudes"),
            ("jamais", "habitudes"),
            ("mon travail", "contexte_vie"),
            ("ma famille", "contexte_vie"),
            ("je rêve", "reves"),
            ("j'ai peur", "peurs"),
        ]

        for pattern, category in learning_patterns:
            if pattern in user_text:
                learnings.append({"category": category, "learned": user_text, "confidence": 0.8})

        return learnings

    def _extract_promises(self, exchange: dict[str, Any]) -> list[str]:
        """Extrait les promesses faites"""
        promises = []
        jeffrey_text = exchange.get("jeffrey", "").lower()

        promise_keywords = [
            "promets",
            "je vais",
            "on va",
            "on fera",
            "je te montrerai",
            "je t'aiderai",
            "je serai",
            "nous allons",
        ]

        for keyword in promise_keywords:
            if keyword in jeffrey_text:
                promises.append(jeffrey_text)
                break

        return promises

    def _rate_significance(self, exchange: dict[str, Any]) -> float:
        """Évalue l'importance émotionnelle d'un échange"""
        significance = 0.0

        # Facteurs d'importance
        emotion_intensity = exchange.get("jeffrey_state", {}).get("emotion_intensity", 0.5)
        significance += emotion_intensity * 0.3

        # Mots-clés importants
        important_keywords = [
            "amour",
            "toujours",
            "promets",
            "important",
            "special",
            "première fois",
            "souviens",
            "ensemble",
            "nous",
        ]
        text = f"{exchange.get('user', '')} {exchange.get('jeffrey', '')}".lower()

        keyword_count = sum(1 for kw in important_keywords if kw in text)
        significance += min(keyword_count * 0.1, 0.3)

        # Longueur de l'échange
        if len(text) > 200:
            significance += 0.2

        # Émotion forte
        strong_emotions = ["amour", "tristesse profonde", "joie intense", "peur"]
        if any(em in exchange.get("jeffrey_emotion", "").lower() for em in strong_emotions):
            significance += 0.2

        return min(significance, 1.0)

    def _update_semantic_memory(self, memory_entry: dict[str, Any]) -> None:
        """Met à jour la mémoire sémantique"""
        # Apprendre des patterns
        for learning in memory_entry.get("learned", []):
            category = learning["category"]
        if category in ["preferences", "aversions", "habitudes"]:
            self.learn_about_user(category, f"from_{memory_entry['timestamp']}", learning["learned"])

    def _create_associations(self, memory_entry: dict[str, Any]) -> None:
        """Crée des associations entre concepts"""
        keywords = memory_entry.get("keywords", [])

        for keyword in keywords:
            if keyword not in self.associative_memory:
                self.associative_memory[keyword] = []

            # Associer à ce souvenir
            self.associative_memory[keyword].append(
                {"memory": memory_entry, "strength": memory_entry["emotional_significance"]}
            )

            # Limiter le nombre d'associations par mot
            if len(self.associative_memory[keyword]) > 50:
                # Garder les plus significatives
                self.associative_memory[keyword] = sorted(
                    self.associative_memory[keyword], key=lambda x: x["strength"], reverse=True
                )[:30]

    def _update_relationship_state(self, memory_entry: dict[str, Any]) -> None:
        """Met à jour l'état de la relation"""
        self.relationship_state["shared_history"] += 1
        self.relationship_state["last_interaction"] = memory_entry["timestamp"]

        # Augmenter l'intimité basée sur la profondeur émotionnelle
        emotional_depth = memory_entry["emotional_significance"]
        self.relationship_state["emotional_depth"] = min(
            self.relationship_state["emotional_depth"] + emotional_depth * 0.01, 1.0
        )

        # Trust grandit avec le temps et les interactions positives
        if memory_entry.get("my_emotion") in ["joie", "amour", "tendresse"]:
            self.relationship_state["trust_level"] = min(self.relationship_state["trust_level"] + 0.005, 1.0)

        # Intimité basée sur le partage
        if any(kw in memory_entry.get("keywords", []) for kw in ["ensemble", "nous", "notre"]):
            self.relationship_state["intimacy_level"] = min(self.relationship_state["intimacy_level"] + 0.01, 1.0)

    def _calculate_relevance(self, conversation: dict[str, Any], topic: str) -> float:
        """Calcule la pertinence d'un souvenir par rapport à un sujet"""
        relevance = 0.0

        # Recherche directe
        conv_text = f"{conversation.get('user_said', '')} {conversation.get('i_said', '')}".lower()
        if topic in conv_text:
            relevance += 0.5

        # Recherche dans les mots-clés
        if topic in conversation.get("keywords", []):
            relevance += 0.3

        # Boost si émotionnellement significatif
        relevance += conversation.get("emotional_significance", 0) * 0.2

        return min(relevance, 1.0)

    def _calculate_memory_age(self, timestamp: str) -> float:
        """Calcule l'âge d'un souvenir (0 = très récent, 1 = très ancien)"""
        try:
            memory_time = datetime.fromisoformat(timestamp)
            age_days = (datetime.now() - memory_time).days
            # Normaliser sur une échelle de 0 à 1 (365 jours = 1)
            return min(age_days / 365, 1.0)
        except BaseException:
            return 0.5

    def _find_emotional_connections(self, topic: str) -> list[dict[str, Any]]:
        """Trouve des souvenirs émotionnellement liés"""
        emotional_memories = []

        # Chercher dans les moments marquants
        for moment in self.episodic_memory["moments_marquants"]:
            if topic.lower() in str(moment).lower():
                emotional_memories.append(
                    {
                        "type": "emotional",
                        "memory": moment["moment"],
                        "strength": 0.8,
                        "emotion": moment["moment"].get("emotion", "unknown"),
                    }
                )

        return emotional_memories

    def _human_like_recall_filter(self, memories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Filtre les souvenirs comme un humain (récence + importance)"""
        if not memories:
            return []

        # Calculer un score composite
        for memory in memories:
            # Score = pertinence * (1 - age * 0.3) + significance * 0.3
            age = memory.get("age", 0)
            strength = memory.get("strength", 0.5)
            significance = memory.get("memory", {}).get("emotional_significance", 0.5)

            memory["recall_score"] = strength * (1 - age * 0.3) + significance * 0.3

        # Trier par score et retourner les meilleurs
        sorted_memories = sorted(memories, key=lambda x: x["recall_score"], reverse=True)

        # Retourner entre 3 et 10 souvenirs selon la quantité
        return sorted_memories[: min(len(sorted_memories), 10)]

    def _analyze_significance(self, memory_entry: dict[str, Any]) -> str:
        """Analyse pourquoi un moment est significatif"""
        reasons = []

        if memory_entry["emotional_significance"] > 0.8:
            reasons.append("Moment très émotionnel")

        if any(kw in memory_entry.get("keywords", []) for kw in ["amour", "toujours", "promets"]):
            reasons.append("Déclaration importante")

        if "première fois" in memory_entry.get("i_said", "").lower():
            reasons.append("Première expérience")

        if len(memory_entry.get("learned", [])) > 2:
            reasons.append("Beaucoup appris")

        return " - ".join(reasons) if reasons else "Moment mémorable"

    def _calculate_relationship_duration(self, first_timestamp: str) -> str:
        """Calcule la durée de la relation de manière humaine"""
        try:
            first_time = datetime.fromisoformat(first_timestamp)
            duration = datetime.now() - first_time

            if duration.days > 365:
                years = duration.days // 365
                return f"{years} an{'s' if years > 1 else ''}"
            elif duration.days > 30:
                months = duration.days // 30
                return f"{months} mois"
            elif duration.days > 0:
                return f"{duration.days} jour{'s' if duration.days > 1 else ''}"
            else:
                return "qui commence aujourd'hui"
        except BaseException:
            return "indéterminée"

    def _identify_common_topics(self) -> list[str]:
        """Identifie les sujets récurrents dans les conversations"""
        topic_counts = {}

        for conv in self.episodic_memory["conversations"][-100:]:  # Dernières 100 conversations
            for keyword in conv.get("keywords", []):
                topic_counts[keyword] = topic_counts.get(keyword, 0) + 1

        # Retourner les 5 plus fréquents
        sorted_topics = sorted(topic_counts.items(), key=lambda x: x[1], reverse=True)
        return [topic for topic, count in sorted_topics[:5] if count > 3]

    def _persist_to_disk(self) -> None:
        """Sauvegarde la mémoire sur disque"""
        memory_data = {
            "version": "2.0",
            "last_saved": datetime.now().isoformat(),
            "device": platform.node(),
            "episodic_memory": self.episodic_memory,
            "semantic_memory": self.semantic_memory,
            "procedural_memory": self.procedural_memory,
            "associative_memory": self.associative_memory,
            "relationship_state": self.relationship_state,
        }

        # Sauvegarde principale
        main_file = self.memory_path / "jeffrey_complete_memory.json"
        backup_file = self.memory_path / f"jeffrey_memory_backup_{datetime.now().strftime('%Y%m%d')}.json"

        try:
            # Écrire dans un fichier temporaire d'abord
            temp_file = main_file.with_suffix(".tmp")
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(memory_data, f, indent=2, ensure_ascii=False)

            # Remplacer l'ancien fichier
            temp_file.replace(main_file)

            # Backup quotidien
            if not backup_file.exists():
                with open(backup_file, "w", encoding="utf-8") as f:
                    json.dump(memory_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Mémoire sauvegardée: {len(self.episodic_memory['conversations'])} conversations")

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la mémoire: {e}")

    def _load_from_disk(self) -> None:
        """Charge la mémoire depuis le disque"""
        main_file = self.memory_path / "jeffrey_complete_memory.json"

        if main_file.exists():
            try:
                with open(main_file, encoding="utf-8") as f:
                    memory_data = json.load(f)

                # Restaurer les composants
                self.episodic_memory = memory_data.get("episodic_memory", self.episodic_memory)
                self.semantic_memory = memory_data.get("semantic_memory", self.semantic_memory)
                self.procedural_memory = memory_data.get("procedural_memory", self.procedural_memory)
                self.associative_memory = memory_data.get("associative_memory", self.associative_memory)
                self.relationship_state = memory_data.get("relationship_state", self.relationship_state)

                logger.info(f"Mémoire chargée: {len(self.episodic_memory['conversations'])} conversations")

            except Exception as e:
                logger.error(f"Erreur lors du chargement de la mémoire: {e}")
                # Essayer de charger un backup
                self._try_load_backup()
        else:
            logger.info("Aucune mémoire existante trouvée, démarrage avec une mémoire vierge")

    def _try_load_backup(self) -> None:
        """Essaye de charger un backup si le fichier principal est corrompu"""
        backup_files = list(self.memory_path.glob("jeffrey_memory_backup_*.json"))

        if backup_files:
            # Prendre le plus récent
            latest_backup = max(backup_files, key=lambda f: f.stat().st_mtime)

            try:
                with open(latest_backup, encoding="utf-8") as f:
                    memory_data = json.load(f)

                self.episodic_memory = memory_data.get("episodic_memory", self.episodic_memory)
                self.semantic_memory = memory_data.get("semantic_memory", self.semantic_memory)
                self.procedural_memory = memory_data.get("procedural_memory", self.procedural_memory)
                self.associative_memory = memory_data.get("associative_memory", self.associative_memory)
                self.relationship_state = memory_data.get("relationship_state", self.relationship_state)

                logger.info(f"Mémoire restaurée depuis backup: {latest_backup.name}")

            except Exception as e:
                logger.error(f"Erreur lors du chargement du backup: {e}")
