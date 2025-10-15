from __future__ import annotations

import json

# V1.1 PATCH - Fix pour INT-COMP-04 (fuite mémoire)
import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any

# Import du nouveau module Sprint 191
# from core.memory.affective_link_resolver import AffectiveLinkResolver  # Module not yet available


logger = logging.getLogger("memory.manager")


class MemoryManager:
    """
    Gère la mémoire longue durée de Jeffrey :
    - Émotions vécues dans le temps
    - Souvenirs émotionnels
    - Événements marquants
    - Données à long terme
    - Phrases utilisateur
    """

    def __init__(
        self,
        memory_file: str = "data/memory/jeffrey_memory.json",
        read_only: bool = False,
    ):
        self.memory_file = memory_file
        self.read_only = read_only
        self.memory: dict[str, Any] = {
            "emotional_history": [],
            "events": [],
            "knowledge": {},
            "personality_traits": {},
            "important_people": {},
            "happy_memories": [],
            "user_phrases": [],
            "emotional_links": {},  # Nouveau champ Sprint 191
        }
        self._load_memory()

        # Initialisation du résolveur de liens affectifs (Sprint 191)
        self.affective_link_resolver = AffectiveLinkResolver(
            storage_path="data/memory/affective_links.json",
            max_topics=100,
            association_threshold=0.6,
        )

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, encoding="utf-8") as f:
                    self.memory = json.load(f)
            except Exception:
                pass

    def _save_memory(self):
        if self.read_only:
            return  # Ne pas sauvegarder en mode lecture seule

        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)

    # 🔹 Gestion des émotions dans le temps

    def log_emotion(self, emotion: str, intensity: float = 0.5, context: str | None = None):
        entry = {
            "emotion": emotion,
            "intensity": round(float(intensity), 2),
            "timestamp": datetime.now().isoformat(),
            "context": context or "",
        }
        self.memory["emotional_history"].append(entry)
        self._save_memory()

    def get_recent_emotions(self, limit: int = 10) -> list[dict[str, Any]]:
        return self.memory["emotional_history"][-limit:]

    # 🔹 Gestion des souvenirs émotionnels

    def add_memory_event(
        self,
        title: str,
        emotion: str,
        impact: float = 0.5,
        details: str | None = None,
    ):
        event = {
            "title": title,
            "emotion": emotion,
            "impact": round(float(impact), 2),
            "details": details or "",
            "timestamp": datetime.now().isoformat(),
        }
        self.memory["events"].append(event)
        self._save_memory()

    def get_all_events(self) -> list[dict[str, Any]]:
        return self.memory["events"]

    # 🔹 Ajout de connaissances ou traits persistants

    def remember_fact(self, key: str, value: Any):
        self.memory["knowledge"][key] = value
        self._save_memory()

    def remember_trait(self, trait: str, description: str):
        self.memory["personality_traits"][trait] = description
        self._save_memory()

    def get_knowledge(self) -> dict[str, Any]:
        return self.memory["knowledge"]

    def get_traits(self) -> dict[str, str]:
        return self.memory["personality_traits"]

    # 🔹 Relations avec des personnes importantes

    def link_person(self, name: str, relation: str, details: str | None = None):
        self.memory["important_people"][name] = {
            "relation": relation,
            "details": details or "",
            "last_updated": datetime.now().isoformat(),
        }
        self._save_memory()

    def get_linked_people(self) -> dict[str, Any]:
        return self.memory["important_people"]

    # 🔹 Export complet

    def export_memory(self) -> dict[str, Any]:
        return self.memory.copy()

    def clear_memory(self):
        self.memory = {
            "emotional_history": [],
            "events": [],
            "knowledge": {},
            "personality_traits": {},
            "important_people": {},
            "happy_memories": [],
        }
        self._save_memory()

    def describe_emotional_journal(self, limit: int = 10) -> str:
        """
        Génère un résumé lisible de l'historique émotionnel.
        """
        emotions = self.get_recent_emotions(limit)
        if not emotions:
            return "Aucune émotion enregistrée."

        lines = ["🧠 Journal émotionnel de Jeffrey (récent → ancien) :\n"]
        for e in reversed(emotions):
            date = datetime.fromisoformat(e["timestamp"]).strftime("%d/%m/%Y %H:%M")
            context = f" ➤ {e['context']}" if e["context"] else ""
            lines.append(f"[{date}] ({e['emotion']} – intensité {e['intensity']}){context}")

        return "\n".join(lines)

    def store_happy_memory(
        self,
        title: str,
        reason: str,
        source: str = "David",
        tags: list[str] | None = None,
    ):
        """
        Enregistre un souvenir heureux déclenché par une interaction positive.
        Peut être utilisé plus tard pour le réconfort émotionnel.
        """
        memory_entry = {
            "type": "souvenir_heureux",
            "title": title,
            "reason": reason,
            "source": source,
            "tags": tags or [],
            "timestamp": datetime.now().isoformat(),
        }
        self.memory.setdefault("happy_memories", []).append(memory_entry)
        self._save_memory()

    def get_happy_memories(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Retourne les derniers souvenirs heureux enregistrés.
        """
        return self.memory.get("happy_memories", [])[-limit:]

    # 🔹 Gestion des phrases utilisateur

    def store_user_phrase(self, phrase: str, emotion: str | None = None, affectif: bool = False) -> dict[str, Any]:
        """
        Stocke une phrase utilisateur dans la mémoire avec des métadonnées.

        Args:
            phrase (str): La phrase de l'utilisateur à stocker
            emotion (str, optional): L'émotion associée à la phrase
            affectif (bool): Si la phrase a un contenu affectif (je t'aime, merci, etc.)

        Returns:
            dict: L'entrée créée dans la mémoire
        """
        # Vérifier si la phrase est vide
        if not phrase or not phrase.strip():
            return {}

        # Nettoyer la phrase (supprimer les caractères spéciaux)
        phrase_clean = re.sub(r'[^\w\s.,!?\'"-:;]', "", phrase.strip())

        # Vérifier si la phrase contient du contenu affectif
        if not affectif:
            affectif_patterns = [
                r'\bje\s+t[\'"]aime\b',
                r"\bmerci\b",
                r'\bje\s+t[\'"]adore\b',
                r"\btu\s+es\s+g[eé]nial\b",
                r"\btu\s+es\s+super\b",
            ]
            affectif = any(re.search(pattern, phrase_clean.lower()) for pattern in affectif_patterns)

        # Vérifier si la phrase existe déjà pour éviter les doublons
        for existing_phrase in self.memory.get("user_phrases", []):
            if existing_phrase.get("phrase") == phrase_clean:
                # Mettre à jour le compteur d'occurrences et la date
                existing_phrase["occurrences"] += 1
                existing_phrase["last_seen"] = datetime.now().isoformat()
                self._save_memory()
                return existing_phrase

        # Créer l'entrée
        entry = {
            "phrase": phrase_clean,
            "timestamp": datetime.now().isoformat(),
            "last_seen": datetime.now().isoformat(),
            "emotion": emotion or "neutral",
            "affectif": affectif,
            "occurrences": 1,
            "tags": ["affectif"] if affectif else [],
        }

        # Ajouter des tags basés sur le contenu
        if "?" in phrase_clean:
            entry["tags"].append("question")
        if "!" in phrase_clean:
            entry["tags"].append("exclamation")
        if len(phrase_clean.split()) > 15:
            entry["tags"].append("longue")

        # Ajouter à la mémoire
        self.memory.setdefault("user_phrases", []).append(entry)
        self._save_memory()

        return entry

    def get_user_phrases(
        self, limit: int = 20, tag: str | None = None, affectif_only: bool = False
    ) -> list[dict[str, Any]]:
        """
        Récupère les phrases utilisateur stockées.

        Args:
            limit (int): Nombre maximum de phrases à récupérer
            tag (str, optional): Filtrer par tag spécifique
            affectif_only (bool): Récupérer uniquement les phrases affectives

        Returns:
            list: Liste des phrases correspondant aux critères
        """
        phrases = self.memory.get("user_phrases", [])

        # Appliquer les filtres
        if tag:
            phrases = [p for p in phrases if tag in p.get("tags", [])]
        if affectif_only:
            phrases = [p for p in phrases if p.get("affectif", False)]

        # Trier par date décroissante
        phrases.sort(key=lambda x: x.get("last_seen", ""), reverse=True)

        return phrases[:limit]

    def mark_as_favorite_phrase(self, phrase: str, emotion_tags: dict[str, float] | None = None) -> bool:
        """
        Marque une phrase comme favorite pour pouvoir la rejouer plus tard.

        Args:
            phrase (str): La phrase à marquer comme favorite
            emotion_tags (Dict[str, float], optional): Tags émotionnels associés à la phrase

        Returns:
            bool: True si la phrase a été marquée comme favorite, False sinon
        """
        phrases = self.memory.get("user_phrases", [])

        # Vérifier si la phrase existe déjà
        for p in phrases:
            if p.get("phrase") == phrase:
                # Marquer comme favorite si ce n'est pas déjà le cas
                if "is_favorite" not in p or not p["is_favorite"]:
                    p["is_favorite"] = True
                    p["favorite_date"] = datetime.now().isoformat()

                    # Ajouter le tag "favorite" s'il n'existe pas déjà
                    if "favorite" not in p.get("tags", []):
                        p.setdefault("tags", []).append("favorite")

                    # Ajouter les tags émotionnels si fournis
                    if emotion_tags:
                        p["emotion_tags"] = emotion_tags

                    self._save_memory()
                    return True
                return False  # Déjà marquée comme favorite

        # Si la phrase n'existe pas, l'ajouter comme nouvelle phrase favorite
        if phrase and phrase.strip():
            entry = {
                "phrase": phrase,
                "timestamp": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "favorite_date": datetime.now().isoformat(),
                "is_favorite": True,
                "emotion": "neutral",
                "affectif": True,
                "occurrences": 1,
                "tags": ["favorite", "affectif"],
            }

            # Ajouter les tags émotionnels si fournis
            if emotion_tags:
                entry["emotion_tags"] = emotion_tags

            self.memory.setdefault("user_phrases", []).append(entry)
            self._save_memory()
            return True

        return False

    # 🔹 Gestion des liens affectifs entre sujets (Sprint 191)

    def analyze_emotional_links(self, text: str, emotion: str, intensity: float = 0.5) -> dict[str, Any]:
        """
        Analyse un texte pour détecter les sujets et leurs liens émotionnels.
        Utilise le AffectiveLinkResolver pour construire un graphe émotionnel.

        Args:
            text (str): Texte à analyser
            emotion (str): Émotion dominante associée au texte
            intensity (float): Intensité de l'émotion (0-1)

        Returns:
            Dict[str, Any]: Résultat de l'analyse avec graphe émotionnel et sujets détectés
        """
        # Déléguer l'analyse au résolveur de liens
        result = self.affective_link_resolver.process_text(text, emotion, intensity)

        # Stocker les résultats dans la mémoire
        if result and result["topics"]:
            # Mettre à jour le graphe émotionnel dans la mémoire
            self.memory["emotional_links"] = {
                "last_updated": datetime.now().isoformat(),
                "topics": result["topics"],
                "graph": result["graph"],
            }
            self._save_memory()

        return result

    def get_topic_emotions(self, topic: str) -> dict[str, Any]:
        """
        Récupère les émotions associées à un sujet spécifique.

        Args:
            topic (str): Sujet à analyser

        Returns:
            Dict[str, Any]: Informations émotionnelles sur le sujet
        """
        return self.affective_link_resolver.get_topic_emotion(topic)

    def get_related_topics(self, topic: str, threshold: float = 0.5) -> list[dict[str, Any]]:
        """
        Trouve les sujets liés émotionnellement à un sujet donné.

        Args:
            topic (str): Sujet de référence
            threshold (float): Seuil minimal de force du lien

        Returns:
            List[Dict[str, Any]]: Liste des sujets liés avec leur force et émotions partagées
        """
        return self.affective_link_resolver.get_related_topics(topic, threshold)

    def get_emotional_graph(self) -> dict[str, Any]:
        """
        Récupère le graphe émotionnel complet des liens entre sujets.

        Returns:
            Dict[str, Any]: Graphe émotionnel avec nœuds et liens
        """
        # Résoudre les liens pour le graphe complet
        return self.affective_link_resolver.resolve_links()

    def get_emotional_insights(self) -> list[dict[str, Any]]:
        """
        Récupère des insights sur les associations émotionnelles entre sujets.

        Returns:
            List[Dict[str, Any]]: Liste d'insights émotionnels
        """
        return self.affective_link_resolver.get_emotional_insights()

    def clean_outdated_topics(self, days_threshold: int = 30) -> int:
        """
        Nettoie les sujets qui n'ont pas été mentionnés depuis longtemps.

        Args:
            days_threshold (int): Nombre de jours d'inactivité avant suppression

        Returns:
            int: Nombre de sujets supprimés
        """
        return self.affective_link_resolver.clean_old_topics(days_threshold)

    def get_favorite_phrases(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Récupère les phrases marquées comme favorites.

        Args:
            limit (int): Nombre maximum de phrases à récupérer

        Returns:
            list: Liste des phrases favorites
        """
        phrases = self.memory.get("user_phrases", [])
        favorites = [p for p in phrases if p.get("is_favorite", False)]

        # Trier par date de marquage comme favorite
        favorites.sort(key=lambda x: x.get("favorite_date", ""), reverse=True)

        return favorites[:limit]

    # V1.1 PATCH - Méthodes de nettoyage mémoire pour résoudre INT-COMP-04
    def cleanup_old_emotional_history(self, max_days=30):
        """
        Nettoie l'historique émotionnel en conservant uniquement
        les entrées plus récentes que max_days.

        Args:
            max_days (int): Nombre de jours à conserver

        Returns:
            int: Nombre d'entrées supprimées
        """
        if not self.memory.get("emotional_history"):
            return 0

        cutoff_date = (datetime.now() - timedelta(days=max_days)).isoformat()
        before_count = len(self.memory["emotional_history"])

        # Conserver uniquement les entrées plus récentes que la date limite
        self.memory["emotional_history"] = [
            entry for entry in self.memory["emotional_history"] if entry.get("timestamp", "") >= cutoff_date
        ]

        # Sauvegarder les changements
        self._save_memory()

        removed = before_count - len(self.memory["emotional_history"])
        logger.info(f"Nettoyage mémoire: {removed} entrées d'historique émotionnel supprimées")
        return removed

    def trim_user_phrases(self, max_phrases=500, keep_favorite=True):
        """
        Limite le nombre de phrases utilisateur stockées pour éviter
        une croissance illimitée de la mémoire.

        Args:
            max_phrases (int): Nombre maximum de phrases à conserver
            keep_favorite (bool): Si True, conserve toutes les phrases favorites

        Returns:
            int: Nombre de phrases supprimées
        """
        if not self.memory.get("user_phrases"):
            return 0

        before_count = len(self.memory.get("user_phrases", []))

        # Séparer les phrases favorites des phrases normales
        if keep_favorite:
            favorites = [p for p in self.memory.get("user_phrases", []) if p.get("is_favorite", False)]
            regulars = [p for p in self.memory.get("user_phrases", []) if not p.get("is_favorite", False)]
        else:
            favorites = []
            regulars = self.memory.get("user_phrases", [])

        # Trier les phrases normales par date (plus récentes en premier)
        regulars.sort(key=lambda x: x.get("last_seen", ""), reverse=True)

        # Limiter le nombre de phrases normales
        max_regular = max(0, max_phrases - len(favorites))
        regulars = regulars[:max_regular]

        # Recombiner les phrases
        self.memory["user_phrases"] = favorites + regulars

        # Sauvegarder les changements
        self._save_memory()

        removed = before_count - len(self.memory["user_phrases"])
        if removed > 0:
            logger.info(f"Nettoyage mémoire: {removed} phrases utilisateur supprimées")

        return removed

    def optimize_memory(self):
        """
        Effectue une optimisation complète de la mémoire en:
        - Nettoyant l'historique émotionnel ancien
        - Limitant le nombre de phrases utilisateur
        - Actualisant les liens affectifs

        Returns:
            dict: Statistiques de nettoyage
        """
        logger.info("Démarrage de l'optimisation mémoire...")
        stats = {
            "emotional_history_removed": self.cleanup_old_emotional_history(max_days=60),
            "user_phrases_removed": self.trim_user_phrases(max_phrases=300),
            "topics_cleaned": self.clean_outdated_topics(days_threshold=90),
            "timestamp": datetime.now().isoformat(),
        }

        # Force la sauvegarde des données nettoyées
        self._save_memory()

        logger.info(f"Optimisation mémoire terminée: {stats}")
        return stats

    def get_affective_user_phrases(self, emotion: str | None = None) -> list[dict[str, Any]]:
        """
        Récupère les phrases utilisateur à contenu affectif, optionnellement filtrées par émotion.

        Args:
            emotion (str, optional): L'émotion spécifique à rechercher

        Returns:
            List[Dict[str, Any]]: Liste des phrases affectives correspondantes
        """
        phrases = self.memory.get("user_phrases", [])

        # Filtrer d'abord par contenu affectif
        affective_phrases = [p for p in phrases if p.get("affectif", False)]

        # Si une émotion est spécifiée, filtrer davantage
        if emotion:
            emotion = emotion.lower()
            filtered_phrases = []

            for phrase in affective_phrases:
                # Vérifier l'émotion principale
                if phrase.get("emotion", "").lower() == emotion:
                    filtered_phrases.append(phrase)
                    continue

                # Vérifier dans les tags émotionnels
                emotion_tags = phrase.get("emotion_tags", {})
                if emotion in emotion_tags and emotion_tags[emotion] > 0.5:
                    filtered_phrases.append(phrase)
                    continue

                # Vérifier dans les tags généraux
                if "tags" in phrase and emotion in phrase["tags"]:
                    filtered_phrases.append(phrase)
                    continue

                # Recherche dans le contenu de la phrase
                phrase_text = phrase.get("phrase", "").lower()
                emotion_keywords = {
                    "amour": ["aime", "adore", "amour", "chéri"],
                    "joie": ["content", "heureux", "heureuse", "joie", "bonheur"],
                    "nostalgie": ["manque", "souvenir", "nostalgie"],
                    "tristesse": ["triste", "peine", "désolé"],
                    "surprise": ["surprise", "étonné", "stupéfait"],
                    "gratitude": [
                        "merci",
                        "reconnaissance",
                        "remercie",
                        "reconnaissant",
                    ],
                }

                if emotion in emotion_keywords and any(keyword in phrase_text for keyword in emotion_keywords[emotion]):
                    filtered_phrases.append(phrase)

            return filtered_phrases

        # Sans filtre d'émotion, retourner toutes les phrases affectives
        # Tri par date décroissante
        affective_phrases.sort(key=lambda x: x.get("last_seen", ""), reverse=True)

        return affective_phrases
