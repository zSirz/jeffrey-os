"""
Mémoire de travail simple pour Bundle 1
Module réel avec persistance basique
"""

import json
import logging
from collections import defaultdict, deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class WorkingMemory:
    """Mémoire de travail à court terme avec persistance simple"""

    def __init__(self):
        self.short_term = deque(maxlen=50)  # 50 derniers éléments
        self.context_memory = {}  # Contexte actuel
        self.user_facts = defaultdict(list)  # Faits sur les utilisateurs
        self.conversation_history = deque(maxlen=20)  # Historique conversation

        # Persistance
        self.storage_path = Path("data/memory/working_memory.json")
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.ttl_minutes = 30  # Time to live pour les mémoires
        self.importance_threshold = 0.5

        # Stats
        self.stats = {"recalls": 0, "stores": 0, "hits": 0, "misses": 0}

        self._load_memory()

    def initialize(self, config: dict[str, Any]):
        """Initialise la mémoire"""
        if "ttl_minutes" in config:
            self.ttl_minutes = config["ttl_minutes"]
        if "max_items" in config:
            self.short_term = deque(maxlen=config["max_items"])

        logger.info(f"✅ Working memory initialized (TTL={self.ttl_minutes}min)")
        return self

    def process(self, context: dict[str, Any]) -> dict[str, Any]:
        """Traite le contexte et gère la mémoire"""
        input_text = context.get("input", "")

        # Stocker l'interaction courante
        if input_text:
            self._store_interaction(input_text, context)

        # Rappeler les mémoires pertinentes
        memories = self._recall_relevant(input_text, context)
        if memories:
            context["memories"] = memories
            self.stats["hits"] += 1
        else:
            self.stats["misses"] += 1

        # Extraire et stocker les faits importants
        self._extract_facts(context)

        # Nettoyer les vieilles mémoires
        self._cleanup_old_memories()

        # Sauvegarder périodiquement
        if self.stats["stores"] % 10 == 0:
            self._save_memory()

        return context

    def recall(self, query: str) -> list[dict[str, Any]]:
        """Rappelle des mémoires basées sur une requête"""
        self.stats["recalls"] += 1
        return self._recall_relevant(query, {})

    def _store_interaction(self, text: str, context: dict[str, Any]):
        """Stocke une interaction en mémoire"""
        memory_item = {
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "intent": context.get("intent", "unknown"),
            "sentiment": context.get("sentiment", "neutral"),
            "entities": context.get("entities", []),
            "keywords": context.get("keywords", []),
            "importance": self._calculate_importance(context),
        }

        self.short_term.append(memory_item)
        self.conversation_history.append(
            {
                "input": text,
                "response": context.get("response", ""),
                "timestamp": memory_item["timestamp"],
            }
        )

        self.stats["stores"] += 1
        logger.debug(f"Stored memory item: {memory_item['text'][:50]}...")

    def _recall_relevant(self, query: str, context: dict[str, Any]) -> list[dict[str, Any]]:
        """Rappelle les mémoires pertinentes"""
        if not query:
            return []

        query_lower = query.lower()
        query_words = set(query_lower.split())
        relevant_memories = []

        # Chercher dans la mémoire à court terme
        for memory in self.short_term:
            relevance = self._calculate_relevance(memory, query_lower, query_words)
            if relevance > 0.3:  # Seuil de pertinence
                relevant_memories.append({**memory, "relevance": relevance})

        # Chercher dans les faits utilisateur
        for fact_list in self.user_facts.values():
            for fact in fact_list:
                if any(word in fact.get("text", "").lower() for word in query_words):
                    relevant_memories.append({"text": fact["text"], "type": "user_fact", "relevance": 0.8})

        # Trier par pertinence et limiter
        relevant_memories.sort(key=lambda x: x.get("relevance", 0), reverse=True)
        return relevant_memories[:5]  # Top 5

    def _calculate_relevance(self, memory: dict, query: str, query_words: set) -> float:
        """Calcule la pertinence d'une mémoire"""
        score = 0.0

        # Mots communs
        memory_text = memory.get("text", "").lower()
        memory_words = set(memory_text.split())
        common_words = query_words.intersection(memory_words)
        if common_words:
            score += len(common_words) / len(query_words) * 0.5

        # Mots-clés communs
        memory_keywords = set(memory.get("keywords", []))
        common_keywords = query_words.intersection(memory_keywords)
        if common_keywords:
            score += len(common_keywords) / len(query_words) * 0.3

        # Fraîcheur (plus récent = plus pertinent)
        if "timestamp" in memory:
            try:
                memory_time = datetime.fromisoformat(memory["timestamp"])
                age_minutes = (datetime.now() - memory_time).total_seconds() / 60
                if age_minutes < self.ttl_minutes:
                    freshness = 1.0 - (age_minutes / self.ttl_minutes)
                    score += freshness * 0.2
            except:
                pass

        return min(score, 1.0)

    def _calculate_importance(self, context: dict[str, Any]) -> float:
        """Calcule l'importance d'un élément"""
        importance = 0.5  # Base

        # Entités nommées augmentent l'importance
        if context.get("entities"):
            importance += 0.1 * min(len(context["entities"]), 3)

        # Questions sont importantes
        if context.get("intent") == "question":
            importance += 0.2

        # Sentiment fort augmente l'importance
        if context.get("sentiment") in ["positive", "negative"]:
            importance += 0.1

        # Noms propres très importants
        entities = context.get("entities", [])
        if any(e.get("type") == "person_name" for e in entities):
            importance += 0.3

        return min(importance, 1.0)

    def _extract_facts(self, context: dict[str, Any]):
        """Extrait et stocke les faits importants"""
        entities = context.get("entities", [])

        for entity in entities:
            if entity.get("type") == "person_name":
                # Stocker le fait qu'un nom a été mentionné
                self.user_facts["names"].append(
                    {
                        "text": f"L'utilisateur s'appelle {entity['value']}",
                        "value": entity["value"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )

            elif entity.get("type") == "email":
                self.user_facts["contacts"].append(
                    {
                        "text": f"Email: {entity['value']}",
                        "value": entity["value"],
                        "timestamp": datetime.now().isoformat(),
                    }
                )

        # Stocker les préférences détectées
        if context.get("sentiment") == "positive" and context.get("keywords"):
            for keyword in context.get("keywords", [])[:3]:
                self.user_facts["preferences"].append(
                    {
                        "text": f"Aime: {keyword}",
                        "value": keyword,
                        "timestamp": datetime.now().isoformat(),
                    }
                )

    def _cleanup_old_memories(self):
        """Nettoie les mémoires expirées"""
        cutoff_time = datetime.now() - timedelta(minutes=self.ttl_minutes)

        # Nettoyer les faits utilisateur
        for fact_type in self.user_facts:
            self.user_facts[fact_type] = [
                fact for fact in self.user_facts[fact_type] if datetime.fromisoformat(fact["timestamp"]) > cutoff_time
            ]

    def _save_memory(self):
        """Sauvegarde la mémoire sur disque"""
        try:
            data = {
                "short_term": list(self.short_term)[-20:],  # Garder les 20 derniers
                "user_facts": dict(self.user_facts),
                "stats": self.stats,
                "saved_at": datetime.now().isoformat(),
            }

            with open(self.storage_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug("Memory saved to disk")
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")

    def _load_memory(self):
        """Charge la mémoire depuis le disque"""
        if not self.storage_path.exists():
            return

        try:
            with open(self.storage_path) as f:
                data = json.load(f)

            # Restaurer mémoire court terme
            for item in data.get("short_term", []):
                self.short_term.append(item)

            # Restaurer faits utilisateur
            self.user_facts.update(data.get("user_facts", {}))

            # Restaurer stats
            self.stats.update(data.get("stats", {}))

            logger.info(f"Memory loaded from disk (saved at {data.get('saved_at')})")
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")

    def get_stats(self) -> dict[str, Any]:
        """Retourne les statistiques"""
        return {
            **self.stats,
            "short_term_size": len(self.short_term),
            "user_facts_count": sum(len(facts) for facts in self.user_facts.values()),
            "hit_rate": self.stats["hits"] / max(self.stats["hits"] + self.stats["misses"], 1),
        }

    def shutdown(self):
        """Arrêt propre avec sauvegarde"""
        self._save_memory()
        logger.info(f"Working memory shutdown. Stats: {self.get_stats()}")
