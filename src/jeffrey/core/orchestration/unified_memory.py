"""
Unified Memory System - Fusion des systèmes de mémoire redondants

Unifie:
- Mémoire contextuelle (conversation actuelle)
- Mémoire émotionnelle (états et patterns)
- Mémoire à long terme (apprentissages)
- Cache intelligent pour performance

Remplace: memory.py, memory_system.py, memory_manager.py
"""

import json
import logging
import os
from collections import OrderedDict, deque
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


class LRUCache:
    """Cache LRU simple pour optimiser les performances"""

    def __init__(self, maxsize: int = 100):
        self.cache = OrderedDict()
        self.maxsize = maxsize

    def get(self, key: str) -> Any | None:
        if key in self.cache:
            # Déplacer en fin pour marquer comme récemment utilisé
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.maxsize:
            # Supprimer le plus ancien
            self.cache.popitem(last=False)


class UnifiedMemory:
    """
    Système de mémoire unifié pour Jeffrey V1.1

    Fusionne tous les systèmes de mémoire redondants en une architecture cohérente.
    Gère contexte, émotions, apprentissages et relations de manière unifiée.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir

        # Mémoire contextuelle (conversation actuelle)
        self.context_window = 10
        self.current_context = deque(maxlen=self.context_window)

        # Mémoire émotionnelle
        self.emotional_traces: dict[str, dict] = {}
        self.emotional_patterns: dict[str, list] = {}

        # Mémoire à long terme
        self.long_term_patterns: dict[str, Any] = {}
        self.learned_preferences: dict[str, Any] = {}
        self.relationships: dict[str, dict] = {}

        # Cache intelligent
        self.cache = LRUCache(maxsize=100)

        # Statistiques
        self.stats = {"total_interactions": 0, "emotional_events": 0, "patterns_learned": 0}

        # Charger les données persistantes
        self._load_persistent_data()

    def _load_persistent_data(self):
        """Charge les données persistantes depuis le disque"""
        files_to_load = {
            "emotional_memory.json": "emotional_traces",
            "conversation_memory.json": "long_term_patterns",
            "relationships.json": "relationships",
            "jeffrey_learning.json": "learned_preferences",
        }

        for filename, attr in files_to_load.items():
            filepath = os.path.join(self.data_dir, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, encoding="utf-8") as f:
                        data = json.load(f)

                        # Gestion spéciale pour jeffrey_learning.json
                        if filename == "jeffrey_learning.json":
                            # Récupérer les données utilisateur stockées au niveau racine
                            user_data = {}
                            for key, value in data.items():
                                if key not in [
                                    "concepts",
                                    "patterns",
                                    "preferences",
                                    "last_update",
                                    "responses",
                                    "emotions",
                                    "learning_history",
                                    "user_info",
                                ]:
                                    # C'est probablement un user_id
                                    user_data[key] = value

                            # Fusionner avec les préférences existantes
                            if "preferences" in data:
                                user_data.update(data["preferences"])

                            setattr(self, attr, user_data)
                            logger.info(f"Loaded {filename}: {len(user_data)} users")
                        else:
                            setattr(self, attr, data)
                            logger.info(f"Loaded {filename}: {len(data)} entries")
                except Exception as e:
                    logger.error(f"Error loading {filename}: {e}")

    def save_persistent_data(self):
        """Sauvegarde les données persistantes sur le disque"""
        data_to_save = {
            "emotional_memory.json": self.emotional_traces,
            "conversation_memory.json": self.long_term_patterns,
            "relationships.json": self.relationships,
            "jeffrey_learning.json": self.learned_preferences,
        }

        for filename, data in data_to_save.items():
            filepath = os.path.join(self.data_dir, filename)
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Error saving {filename}: {e}")

    def update(self, message: str, emotion_state: dict, metadata: dict = None):
        """
        Met à jour la mémoire avec un nouveau message

        Args:
            message: Le message de l'utilisateur ou de Jeffrey
            emotion_state: État émotionnel actuel
            metadata: Métadonnées additionnelles (user_id, timestamp, etc.)
        """
        timestamp = datetime.now().isoformat()
        metadata = metadata or {}

        # Créer l'entrée mémoire
        memory_entry = {
            "message": message,
            "emotion": emotion_state,
            "timestamp": timestamp,
            "metadata": metadata,
        }

        # Mettre à jour le contexte
        self.current_context.append(memory_entry)

        # Mettre à jour les traces émotionnelles
        emotion_key = emotion_state.get("primary_emotion", "neutral")
        if emotion_key not in self.emotional_traces:
            self.emotional_traces[emotion_key] = {
                "count": 0,
                "last_seen": timestamp,
                "contexts": [],
            }

        self.emotional_traces[emotion_key]["count"] += 1
        self.emotional_traces[emotion_key]["last_seen"] = timestamp
        self.emotional_traces[emotion_key]["contexts"].append(message[:50])

        # Détecter des patterns
        self._detect_patterns(message, emotion_state)

        # Mettre à jour les statistiques
        self.stats["total_interactions"] += 1
        if emotion_state.get("intensity", 0) > 0.5:
            self.stats["emotional_events"] += 1

        # Invalider le cache si nécessaire
        self.cache.put(f"context_{timestamp}", memory_entry)

    def _detect_patterns(self, message: str, emotion_state: dict):
        """Détecte et enregistre des patterns d'interaction"""
        # Pattern simple : mots-clés récurrents avec émotions
        words = message.lower().split()
        emotion = emotion_state.get("primary_emotion", "neutral")

        for word in words:
            if len(word) > 4:  # Mots significatifs seulement
                pattern_key = f"{word}_{emotion}"
                if pattern_key not in self.emotional_patterns:
                    self.emotional_patterns[pattern_key] = []
                self.emotional_patterns[pattern_key].append(datetime.now())

                # Si pattern fréquent, l'enregistrer
                if len(self.emotional_patterns[pattern_key]) > 3:
                    self.long_term_patterns[pattern_key] = {
                        "word": word,
                        "emotion": emotion,
                        "frequency": len(self.emotional_patterns[pattern_key]),
                        "learned_at": datetime.now().isoformat(),
                    }
                    self.stats["patterns_learned"] += 1

    def retrieve(self, query_type: str = "all", limit: int = 10) -> list[dict]:
        """
        Récupère des informations de la mémoire

        Args:
            query_type: Type de requête (all, context, emotional, patterns)
            limit: Nombre maximum de résultats

        Returns:
            Liste des entrées mémoire pertinentes
        """
        # Vérifier le cache d'abord
        cache_key = f"retrieve_{query_type}_{limit}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        results = []

        if query_type in ["all", "context"]:
            # Récupérer le contexte récent
            results.extend(list(self.current_context)[-limit:])

        if query_type in ["all", "emotional"]:
            # Récupérer les traces émotionnelles récentes
            sorted_emotions = sorted(self.emotional_traces.items(), key=lambda x: x[1]["last_seen"], reverse=True)[
                :limit
            ]

            for emotion, data in sorted_emotions:
                results.append({"type": "emotional_trace", "emotion": emotion, "data": data})

        if query_type in ["all", "patterns"]:
            # Récupérer les patterns appris
            sorted_patterns = sorted(
                self.long_term_patterns.items(),
                key=lambda x: x[1].get("frequency", 0),
                reverse=True,
            )[:limit]

            for pattern, data in sorted_patterns:
                results.append({"type": "pattern", "pattern": pattern, "data": data})

        # Mettre en cache
        self.cache.put(cache_key, results)

        return results

    def get_emotional_summary(self, user_id: str = "default") -> dict:
        """Résumé de l'état émotionnel global"""
        summary = {
            "dominant_emotions": [],
            "emotional_stability": 0.0,
            "recent_mood": "neutral",
            "relationship_depth": 0,
        }

        # Émotions dominantes - Compatible avec l'ancien et nouveau format
        if self.emotional_traces:
            try:
                # Nouveau format avec structure complète
                sorted_emotions = sorted(
                    [(k, v) for k, v in self.emotional_traces.items() if isinstance(v, dict) and "count" in v],
                    key=lambda x: x[1]["count"],
                    reverse=True,
                )[:3]
                summary["dominant_emotions"] = [e[0] for e in sorted_emotions]
            except (KeyError, TypeError):
                # Format legacy ou format emotions simple
                if "emotions" in self.emotional_traces:
                    # Format avec clé "emotions"
                    emotions_data = self.emotional_traces["emotions"]
                    sorted_emotions = sorted(
                        emotions_data.items(),
                        key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0,
                        reverse=True,
                    )[:3]
                    summary["dominant_emotions"] = [e[0] for e in sorted_emotions]
                else:
                    # Format direct {emotion: value}
                    sorted_emotions = sorted(
                        [(k, v) for k, v in self.emotional_traces.items() if isinstance(v, (int, float))],
                        key=lambda x: x[1],
                        reverse=True,
                    )[:3]
                    summary["dominant_emotions"] = [e[0] for e in sorted_emotions]

        # Stabilité émotionnelle (moins de changements = plus stable)
        recent_emotions = []
        for entry in list(self.current_context)[-5:]:
            if "emotion" in entry:
                recent_emotions.append(entry["emotion"].get("primary_emotion"))

        if recent_emotions:
            unique_emotions = len(set(recent_emotions))
            summary["emotional_stability"] = 1.0 - (unique_emotions / len(recent_emotions))
            summary["recent_mood"] = recent_emotions[-1]

        # Profondeur relationnelle
        if user_id in self.relationships:
            rel = self.relationships[user_id]
            summary["relationship_depth"] = rel.get("depth", 0)

        return summary

    def update_relationship(self, user_id: str, interaction_quality: float):
        """Met à jour la relation avec un utilisateur"""
        if user_id not in self.relationships:
            self.relationships[user_id] = {
                "first_interaction": datetime.now().isoformat(),
                "depth": 0,
                "quality": 0.5,
                "interaction_count": 0,
            }

        rel = self.relationships[user_id]
        rel["interaction_count"] += 1
        rel["last_interaction"] = datetime.now().isoformat()

        # Moyenne pondérée pour la qualité
        rel["quality"] = (rel["quality"] * 0.8) + (interaction_quality * 0.2)

        # La profondeur augmente avec le temps et la qualité
        if rel["quality"] > 0.6:
            rel["depth"] = min(rel["depth"] + 0.05, 1.0)
        elif rel["quality"] < 0.4:
            rel["depth"] = max(rel["depth"] - 0.02, 0.0)

    def get_context_summary(self, user_id: str = None) -> str:
        """
        Retourne le résumé du contexte actuel

        Args:
            user_id: (Optionnel) Identifiant utilisateur (ignoré pour compatibilité)
        """
        if not self.current_context:
            return "Aucun contexte de conversation disponible."

        summary_parts = []

        # Derniers messages
        recent = list(self.current_context)[-3:]
        if recent:
            messages = [f"- {entry['message']}" for entry in recent]
            summary_parts.append("Messages récents:\n" + "\n".join(messages))

        # État émotionnel
        emotional_summary = self.get_emotional_summary()
        if emotional_summary["recent_mood"] != "neutral":
            summary_parts.append(f"Humeur actuelle: {emotional_summary['recent_mood']}")

        # Patterns détectés
        if self.long_term_patterns and isinstance(self.long_term_patterns, dict):
            try:
                patterns_list = list(self.long_term_patterns.values())
                if patterns_list:
                    top_pattern = max(
                        patterns_list,
                        key=lambda x: x.get("frequency", 0) if isinstance(x, dict) else 0,
                    )
                    summary_parts.append(f"Thème fréquent: {top_pattern.get('word', 'inconnu')}")
            except (TypeError, AttributeError):
                pass  # Ignore si format incorrect

        return "\n\n".join(summary_parts)

    def clear_old_data(self, days: int = 30):
        """Nettoie les données anciennes pour optimiser les performances"""
        cutoff_date = datetime.now() - timedelta(days=days)

        # Nettoyer les traces émotionnelles anciennes
        for emotion, data in self.emotional_traces.items():
            if "last_seen" in data:
                last_seen = datetime.fromisoformat(data["last_seen"])
                if last_seen < cutoff_date:
                    del self.emotional_traces[emotion]

    def search_memories(self, user_id: str, query: str) -> list[str]:
        """Recherche dans les VRAIS souvenirs de l'utilisateur"""
        results = []
        query_lower = query.lower()

        # Identifier le type de recherche (animal, personne, objet, etc.)
        search_keywords = []
        if "chien" in query_lower:
            search_keywords = ["chien", "dog", "toutou", "canin"]
        elif "chat" in query_lower:
            search_keywords = ["chat", "cat", "minou", "félin"]
        elif "hamster" in query_lower:
            search_keywords = ["hamster", "rongeur"]
        elif "frère" in query_lower or "soeur" in query_lower:
            search_keywords = ["frère", "soeur", "famille", "brother", "sister"]
        else:
            # Mots clés génériques de la requête
            search_keywords = [w for w in query_lower.split() if len(w) > 3]

        # Chercher dans les préférences apprises (plus structuré)
        if user_id in self.learned_preferences:
            user_prefs = self.learned_preferences.get(user_id, {})
            for key, value in user_prefs.items():
                # Chercher dans la clé ET la valeur
                if any(kw in key.lower() for kw in search_keywords):
                    # Format plus naturel pour les animaux
                    if "animal" in key or any(animal in key for animal in ["chien", "chat", "hamster"]):
                        if isinstance(value, dict) and "nom" in value:
                            results.append(f"Ton {key} s'appelle {value['nom']}")
                        elif isinstance(value, str):
                            results.append(f"Je me souviens que ton {key} s'appelle {value}")
                    else:
                        results.append(f"Je sais que {key}: {value}")

        # Chercher dans la mémoire conversationnelle (pour les infos partagées)
        if user_id in self.long_term_patterns:
            user_patterns = self.long_term_patterns.get(user_id, {})
            for pattern_key, pattern_data in user_patterns.items():
                # Chercher les patterns qui contiennent les mots clés
                if any(kw in str(pattern_data).lower() for kw in search_keywords):
                    # Extraire l'info pertinente si c'est structuré
                    if isinstance(pattern_data, dict):
                        if "nom" in pattern_data:
                            results.append(f"D'après mes souvenirs: {pattern_data['nom']}")
                        elif "info" in pattern_data:
                            results.append(pattern_data["info"])
                    # Éviter de retourner des questions
                    elif not any(q in str(pattern_data).lower() for q in ["comment s'appelle", "?"]):
                        results.append(f"Je me rappelle: {pattern_data}")

        # Si on n'a rien trouvé, retourner liste vide (dialogue_engine gérera)
        return results[:3]  # Top 3 résultats

    def save_fact(self, user_id: str, category: str, fact: str):
        """Sauvegarde un fait important (ex: nom d'un animal)"""
        if user_id not in self.learned_preferences:
            self.learned_preferences[user_id] = {}

        # Sauvegarder le fait de manière structurée
        self.learned_preferences[user_id][category] = fact

        # Sauvegarder immédiatement sur disque
        self.save_persistent_data()

        # Logger pour debug
        logger.info(f"Saved fact for {user_id}: {category} = {fact}")

    def get_all_memories(self, user_id: str) -> list[str]:
        """Récupère tous les souvenirs d'un utilisateur pour enrichir le contexte GPT"""
        memories = []

        # Contexte récent
        for entry in list(self.current_context)[-5:]:
            if entry.get("user_id") == user_id:
                memories.append(entry["message"])

        # Patterns à long terme
        if user_id in self.long_term_patterns:
            patterns = self.long_term_patterns[user_id]
            if isinstance(patterns, dict):
                for key, value in list(patterns.items())[:3]:
                    memories.append(f"{key}: {value}")

        # Préférences apprises
        if user_id in self.learned_preferences:
            prefs = self.learned_preferences[user_id]
            if isinstance(prefs, dict):
                for key, value in list(prefs.items())[:2]:
                    memories.append(f"Préférence: {key} = {value}")

        return memories

    def save_fact(self, user_id: str, category: str, fact: str):
        """Sauvegarde un fait spécifique pour un utilisateur"""
        if user_id not in self.learned_preferences:
            self.learned_preferences[user_id] = {}

        self.learned_preferences[user_id][category] = fact

        # Sauvegarder immédiatement
        self._save_learning_file()

        logger.info(f"Saved fact for {user_id}: {category} = {fact}")

    def _save_learning_file(self):
        """Sauvegarde le fichier jeffrey_learning.json en préservant la structure complète"""
        filepath = os.path.join(self.data_dir, "jeffrey_learning.json")

        try:
            # Charger le fichier existant pour préserver la structure
            existing_data = {}
            if os.path.exists(filepath):
                with open(filepath, encoding="utf-8") as f:
                    existing_data = json.load(f)

            # Mettre à jour avec les données utilisateur de learned_preferences
            for user_id, data in self.learned_preferences.items():
                existing_data[user_id] = data

            # Sauvegarder
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Error saving learning file: {e}")


# =============================================================================
# SINGLETON PATTERN - Accès Global à la Mémoire Unifiée
# =============================================================================

# Instance singleton globale
_unified_memory_instance = None


def get_unified_memory():
    """
    Retourne l'instance singleton de UnifiedMemory.

    Utilisé par AGIOrchestrator et autres composants pour accès centralisé
    à la mémoire unifiée du système Jeffrey.

    Returns:
        UnifiedMemory: Instance singleton de la mémoire unifiée

    Thread-safety:
        Cette implémentation n'est pas thread-safe. Pour une utilisation
        multi-thread, ajouter un threading.Lock().

    Example:
        >>> memory = get_unified_memory()
        >>> memory.update("Message", {"primary_emotion": "joy"})
    """
    global _unified_memory_instance

    if _unified_memory_instance is None:
        # Initialiser l'instance avec les paramètres par défaut
        _unified_memory_instance = UnifiedMemory(data_dir="data")
        logger.info("✅ UnifiedMemory singleton initialisé")

    return _unified_memory_instance


def reset_unified_memory():
    """
    Réinitialise le singleton (utile pour les tests).

    ⚠️ ATTENTION : Cette fonction efface l'instance en mémoire.
    Les données persistantes sur disque ne sont pas affectées.
    """
    global _unified_memory_instance
    _unified_memory_instance = None
    logger.info("🔄 UnifiedMemory singleton réinitialisé")


# Export explicite pour clarté
__all__ = ['UnifiedMemory', 'get_unified_memory', 'reset_unified_memory']
