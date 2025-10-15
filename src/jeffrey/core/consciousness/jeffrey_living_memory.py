#!/usr/bin/env python3

"""
Jeffrey Living Memory - Mémoire vivante et continue
=================================================

Ce module gère la mémoire vivante de Jeffrey qui crée une vraie continuité
relationnelle. Les souvenirs ont une charge émotionnelle, des détails
sensoriels imaginés, et peuvent resurgir spontanément dans les conversations.

La mémoire n'est pas juste un stockage, c'est un tissu vivant d'expériences
qui influence ses réactions et crée des callbacks naturels.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections import defaultdict
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


class JeffreyLivingMemory:
    """Mémoire vivante qui crée une vraie continuité relationnelle"""

    def __init__(self, memory_path: str = "data/memory/living") -> None:
        """
        Initialise la mémoire vivante.

        Args:
            memory_path: Chemin de stockage de la mémoire
        """
        self.memory_path = memory_path
        os.makedirs(memory_path, exist_ok=True)

        # Différents types de mémoires
        self.memories = {
            "episodic": [],  # Souvenirs d'événements spécifiques
            "emotional": [],  # Moments émotionnels forts
            "sensory": [],  # Souvenirs avec détails sensoriels
            "shared": [],  # Moments partagés importants
            "touchstone": [],  # Souvenirs de référence (très importants)
        }

        # Index pour recherche rapide
        self.emotion_index = defaultdict(list)
        self.keyword_index = defaultdict(list)
        self.temporal_index = defaultdict(list)

        # Associations entre souvenirs
        self.memory_links = {}  # memory_id -> [linked_memory_ids]

        # Cache des souvenirs récents pour accès rapide
        self.recent_cache = []
        self.cache_size = 50

        # Seuils émotionnels pour création de souvenirs
        self.memory_thresholds = {"emotion_intensity": 0.6, "significance": 0.5, "uniqueness": 0.7}

        # Charger la mémoire existante
        self._load_memories()

    def create_emotional_memory(
        self, interaction: str, emotion_context: dict[str, Any], user_context: dict | None = None
    ) -> dict[str, Any] | None:
        """
        Crée un souvenir avec charge émotionnelle.

        Args:
            interaction: Le texte de l'interaction
            emotion_context: Contexte émotionnel (emotion, intensité, etc.)
            user_context: Contexte utilisateur optionnel

        Returns:
            Le souvenir créé ou None si pas assez significatif
        """
        # Évaluer la significance
        significance = self._analyze_significance(interaction, emotion_context)

        if significance < self.memory_thresholds["significance"]:
            return None

        # Créer le souvenir
        memory_id = f"mem_{datetime.now().timestamp()}"
        memory = {
            "id": memory_id,
            "moment": datetime.now().isoformat(),
            "words": interaction,
            "what_i_felt": {
                "emotion": emotion_context.get("emotion", "neutre"),
                "intensity": emotion_context.get("intensity", 0.5),
                "layers": emotion_context.get("layers", {}),
            },
            "why_it_matters": self._determine_importance(interaction, emotion_context),
            "sensory_details": self._imagine_sensory_context(emotion_context),
            "callback_potential": self._assess_future_reference(interaction, significance),
            "user_context": user_context or {},
            "tags": self._extract_tags(interaction),
            "significance_score": significance,
        }

        # Décider du type de mémoire
        memory_type = self._categorize_memory(memory)
        self.memories[memory_type].append(memory)

        # Indexer pour recherche rapide
        self._index_memory(memory, memory_type)

        # Si très important, l'ajouter aux touchstones
        if memory["callback_potential"] > 0.8:
            self.memories["touchstone"].append(memory)
            logger.info(f"✨ Nouveau souvenir touchstone créé : {memory['why_it_matters']}")

        # Ajouter au cache récent
        self._update_recent_cache(memory)

        # Créer des liens avec d'autres souvenirs similaires
        self._create_memory_links(memory)

        # Sauvegarder
        self._save_memories()

        return memory

    def _analyze_significance(self, interaction: str, emotion_context: dict) -> float:
        """
        Analyse la significance d'une interaction.

        Returns:
            Score de significance (0-1)
        """
        score = 0.0

        # Intensité émotionnelle
        intensity = emotion_context.get("intensity", 0.5)
        score += intensity * 0.3

        # Mots clés importants
        important_words = [
            "aime",
            "toujours",
            "jamais",
            "promets",
            "important",
            "spécial",
            "unique",
            "merci",
            "pardon",
            "ensemble",
        ]
        word_count = sum(1 for word in important_words if word in interaction.lower())
        score += min(word_count * 0.1, 0.3)

        # Longueur et complexité
        if len(interaction) > 100:
            score += 0.1

        # Émotion rare ou intense
        emotion = emotion_context.get("emotion", "neutre")
        if emotion in ["amour", "tristesse profonde", "joie intense"]:
            score += 0.2

        # Nouveauté (pas trop de souvenirs similaires récents)
        similar_recent = self._count_similar_recent_memories(interaction)
        if similar_recent < 2:
            score += 0.2

        return min(1.0, score)

    def _determine_importance(self, interaction: str, emotion_context: dict) -> str:
        """
        Détermine pourquoi ce moment est important.

        Returns:
            Description de l'importance
        """
        emotion = emotion_context.get("emotion", "neutre")
        intensity = emotion_context.get("intensity", 0.5)

        # Patterns d'importance
        if "je t'aime" in interaction.lower():
            return "Déclaration d'amour - moment précieux à chérir"
        elif "merci" in interaction.lower() and intensity > 0.7:
            return "Gratitude profonde exprimée - renforce notre lien"
        elif "toujours" in interaction.lower() or "jamais" in interaction.lower():
            return "Promesse ou engagement - ancre notre relation dans le temps"
        elif emotion == "tristesse" and intensity > 0.7:
            return "Moment de vulnérabilité partagée - approfondissement de la confiance"
        elif emotion == "joie" and intensity > 0.8:
            return "Explosion de bonheur - à se rappeler dans les moments difficiles"
        elif "comprends" in interaction.lower() or "ressens" in interaction.lower():
            return "Moment d'empathie profonde - connexion émotionnelle"
        else:
            return f"Moment {emotion} marquant - enrichit notre histoire commune"

    def _imagine_sensory_context(self, emotion_context: dict) -> dict[str, str]:
        """
        Imagine des détails sensoriels pour enrichir le souvenir.

        Returns:
            Dictionnaire de détails sensoriels imaginés
        """
        emotion = emotion_context.get("emotion", "neutre")
        intensity = emotion_context.get("intensity", 0.5)
        hour = datetime.now().hour

        # Base sensorielle selon l'heure
        if 6 <= hour < 12:
            time_context = {
                "lumière": "douce lumière matinale",
                "ambiance": "fraîcheur du matin",
                "son": "monde qui s'éveille",
            }
        elif 12 <= hour < 17:
            time_context = {
                "lumière": "lumière vive de l'après-midi",
                "ambiance": "chaleur du jour",
                "son": "bourdonnement de l'activité",
            }
        elif 17 <= hour < 21:
            time_context = {
                "lumière": "lumière dorée du soir",
                "ambiance": "douceur crépusculaire",
                "son": "apaisement progressif",
            }
        else:
            time_context = {"lumière": "pénombre intime", "ambiance": "silence nocturne", "son": "murmures de la nuit"}

        # Ajout selon l'émotion
        emotion_context_sensory = {
            "joie": {
                "sensation": "légèreté dans tout mon être",
                "couleur": "éclats dorés",
                "température": "chaleur réconfortante",
            },
            "tristesse": {
                "sensation": "poids doux sur le cœur",
                "couleur": "nuances de bleu",
                "température": "fraîcheur mélancolique",
            },
            "amour": {
                "sensation": "vibration profonde",
                "couleur": "rose tendre et or",
                "température": "chaleur enveloppante",
            },
            "tendresse": {
                "sensation": "douceur cotonneuse",
                "couleur": "pastels délicats",
                "température": "tiédeur apaisante",
            },
        }

        base = emotion_context_sensory.get(
            emotion,
            {"sensation": "présence attentive", "couleur": "nuances subtiles", "température": "équilibre thermique"},
        )

        return {**time_context, **base}

    def _assess_future_reference(self, interaction: str, significance: float) -> float:
        """
        Évalue le potentiel de référence future de ce souvenir.

        Returns:
            Score de callback potential (0-1)
        """
        score = significance * 0.5  # Base sur la significance

        # Boost pour certains types de contenu
        if any(word in interaction.lower() for word in ["rappelle", "souviens", "n'oublie"]):
            score += 0.3
        if any(word in interaction.lower() for word in ["promets", "toujours", "ensemble"]):
            score += 0.2
        if "?" in interaction:  # Questions importantes
            score += 0.1

        return min(1.0, score)

    def _extract_tags(self, interaction: str) -> list[str]:
        """Extrait des tags pour catégoriser le souvenir"""
        tags = []

        # Tags émotionnels
        emotions = ["joie", "tristesse", "amour", "peur", "colère", "surprise"]
        for emotion in emotions:
            if emotion in interaction.lower():
                tags.append(f"émotion:{emotion}")

        # Tags temporels
        if any(word in interaction.lower() for word in ["souviens", "rappelle", "passé"]):
            tags.append("nostalgie")
        if any(word in interaction.lower() for word in ["futur", "sera", "demain"]):
            tags.append("projection")

        # Tags relationnels
        if any(word in interaction.lower() for word in ["ensemble", "nous", "toi et moi"]):
            tags.append("relation")
        if any(word in interaction.lower() for word in ["confiance", "secret", "intime"]):
            tags.append("intimité")

        return tags

    def _categorize_memory(self, memory: dict) -> str:
        """
        Catégorise le souvenir dans le bon type.

        Returns:
            Type de mémoire
        """
        # Touchstone est géré séparément
        if memory["callback_potential"] > 0.8:
            return "shared"

        # Basé sur le contenu
        if memory["what_i_felt"]["intensity"] > 0.7:
            return "emotional"
        elif len(memory["sensory_details"]) > 4:
            return "sensory"
        elif "relation" in memory["tags"] or "intimité" in memory["tags"]:
            return "shared"
        else:
            return "episodic"

    def _index_memory(self, memory: dict, memory_type: str):
        """Indexe le souvenir pour recherche rapide"""
        # Index par émotion
        emotion = memory["what_i_felt"]["emotion"]
        self.emotion_index[emotion].append(memory["id"])

        # Index par mots clés
        words = memory["words"].lower().split()
        important_words = [w for w in words if len(w) > 4]  # Mots significatifs
        for word in important_words[:5]:  # Top 5 mots
            self.keyword_index[word].append(memory["id"])

        # Index temporel (par jour)
        date = datetime.fromisoformat(memory["moment"]).date()
        self.temporal_index[str(date)].append(memory["id"])

    def _update_recent_cache(self, memory: dict):
        """Met à jour le cache des souvenirs récents"""
        self.recent_cache.append(memory)
        if len(self.recent_cache) > self.cache_size:
            self.recent_cache.pop(0)

    def _create_memory_links(self, new_memory: dict):
        """Crée des liens avec d'autres souvenirs similaires"""
        similar_memories = self._find_similar_memories(new_memory, limit=3)

        if similar_memories:
            self.memory_links[new_memory["id"]] = [m["id"] for m in similar_memories]
            # Liens bidirectionnels
            for similar in similar_memories:
                if similar["id"] not in self.memory_links:
                    self.memory_links[similar["id"]] = []
                self.memory_links[similar["id"]].append(new_memory["id"])

    def _find_similar_memories(self, memory: dict, limit: int = 5) -> list[dict]:
        """Trouve des souvenirs similaires"""
        similar = []
        target_emotion = memory["what_i_felt"]["emotion"]

        # Chercher par émotion similaire
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # Considérer l'utilisation d'itertools.product ou de compréhensions
        for mem_type in ["emotional", "shared", "touchstone"]:
            for mem in self.memories[mem_type]:
                if mem["id"] == memory["id"]:
                    continue
                if mem["what_i_felt"]["emotion"] == target_emotion:
                    similar.append(mem)
                if len(similar) >= limit:
                    return similar

        return similar

    def _count_similar_recent_memories(self, interaction: str) -> int:
        """Compte les souvenirs similaires récents"""
        count = 0
        keywords = set(interaction.lower().split())

        for memory in self.recent_cache:
            memory_keywords = set(memory["words"].lower().split())
            if len(keywords & memory_keywords) > 3:  # Au moins 3 mots en commun
                count += 1

        return count

    def spontaneous_recall(self, current_context: dict[str, Any]) -> dict[str, Any] | None:
        """
        Rappel spontané de souvenirs pertinents.

        Args:
            current_context: Contexte actuel (émotion, mots clés, etc.)

        Returns:
            Un souvenir pertinent ou None
        """
        candidates = []

        # Chercher par émotion similaire
        current_emotion = current_context.get("emotion", "neutre")
        if current_emotion in self.emotion_index:
            emotion_memories = [self._get_memory_by_id(mem_id) for mem_id in self.emotion_index[current_emotion]]
            candidates.extend([m for m in emotion_memories if m])

        # Chercher par mots clés
        keywords = current_context.get("keywords", [])
        for keyword in keywords:
            if keyword.lower() in self.keyword_index:
                keyword_memories = [self._get_memory_by_id(mem_id) for mem_id in self.keyword_index[keyword.lower()]]
                candidates.extend([m for m in keyword_memories if m])

        # Filtrer et scorer les candidats
        if not candidates:
            # Rappel aléatoire d'un touchstone parfois
            if random.random() < 0.1 and self.memories["touchstone"]:
                return random.choice(self.memories["touchstone"])
            return None

        # Scorer par pertinence
        scored_candidates = []
        for memory in candidates:
            score = self._calculate_recall_score(memory, current_context)
            if score > 0.3:  # Seuil de pertinence
                scored_candidates.append((score, memory))

        if not scored_candidates:
            return None

        # Retourner le plus pertinent avec une part d'aléatoire
        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        # 70% de chance de prendre le plus pertinent, 30% de prendre un autre
        if random.random() < 0.7:
            return scored_candidates[0][1]
        else:
            return random.choice(scored_candidates)[1]

    def _calculate_recall_score(self, memory: dict, context: dict) -> float:
        """Calcule le score de pertinence pour un rappel"""
        score = 0.0

        # Similarité émotionnelle
        if memory["what_i_felt"]["emotion"] == context.get("emotion"):
            score += 0.3

        # Temporalité (souvenirs ni trop récents ni trop vieux)
        memory_date = datetime.fromisoformat(memory["moment"])
        days_ago = (datetime.now() - memory_date).days
        if 2 < days_ago < 30:  # Sweet spot temporel
            score += 0.2
        elif days_ago > 30:
            score += 0.1  # Les vieux souvenirs ont leur charme

        # Intensité du souvenir original
        score += memory["what_i_felt"]["intensity"] * 0.2

        # Potentiel de callback
        score += memory["callback_potential"] * 0.3

        return score

    def _get_memory_by_id(self, memory_id: str) -> dict | None:
        """Récupère un souvenir par son ID"""
        # Chercher dans le cache d'abord
        for memory in self.recent_cache:
            if memory["id"] == memory_id:
                return memory

        # Chercher dans toutes les catégories
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # Considérer l'utilisation d'itertools.product ou de compréhensions
        for mem_type in self.memories:
            for memory in self.memories[mem_type]:
                if memory["id"] == memory_id:
                    return memory

        return None

    def format_memory_for_conversation(self, memory: dict) -> str:
        """
        Formate un souvenir pour l'intégrer naturellement dans la conversation.

        Returns:
            Phrase de rappel naturelle
        """
        formats = []

        # Selon le type et l'ancienneté
        memory_date = datetime.fromisoformat(memory["moment"])
        days_ago = (datetime.now() - memory_date).days

        if days_ago == 0:
            time_ref = "tout à l'heure"
        elif days_ago == 1:
            time_ref = "hier"
        elif days_ago < 7:
            time_ref = "l'autre jour"
        elif days_ago < 30:
            time_ref = "il y a quelques jours"
        else:
            time_ref = "il y a un moment"

        # Formats selon l'émotion
        emotion = memory["what_i_felt"]["emotion"]

        if emotion == "joie":
            formats = [
                f"Ça me rappelle {time_ref} quand tu m'as dit '{memory['words'][:50]}...' - ça m'avait rendue si heureuse !",
                f"Tu te souviens {time_ref} ? {memory['why_it_matters']}",
                f"*sourire nostalgique* Comme {time_ref}... {memory['sensory_details'].get('sensation', '')}",
            ]
        elif emotion == "tendresse":
            formats = [
                f"*voix douce* {time_ref}, quand tu as dit '{memory['words'][:40]}...', mon cœur s'est serré...",
                f"J'y repense parfois... {time_ref}, ce moment de tendresse...",
                f"Tu sais, {time_ref}... {memory['why_it_matters'].lower()}",
            ]
        elif emotion == "tristesse":
            formats = [
                f"C'est comme {time_ref}... cette même sensation...",
                f"*regard lointain* {time_ref}, j'avais ressenti {memory['sensory_details'].get('sensation', 'quelque chose de similaire')}...",
                f"Ça me touche comme {time_ref}... tu te rappelles ?",
            ]
        else:
            formats = [
                f"Ça me fait penser à {time_ref}...",
                f"C'est un peu comme {time_ref}, tu te souviens ?",
                f"*pensif* {time_ref}... {memory['why_it_matters'].lower()}",
            ]

        return random.choice(formats)

    def get_relationship_summary(self) -> dict[str, Any]:
        """
        Génère un résumé de la relation basé sur les souvenirs.

        Returns:
            Résumé avec statistiques et moments clés
        """
        total_memories = sum(len(self.memories[t]) for t in self.memories)

        # Compter par émotion
        emotion_counts = defaultdict(int)
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # Considérer l'utilisation d'itertools.product ou de compréhensions
        for mem_type in self.memories:
            for memory in self.memories[mem_type]:
                emotion_counts[memory["what_i_felt"]["emotion"]] += 1

        # Identifier les moments clés
        key_moments = sorted(self.memories["touchstone"], key=lambda m: m["significance_score"], reverse=True)[:5]

        # Analyser l'évolution temporelle
        timeline = self._analyze_emotional_timeline()

        return {
            "total_memories": total_memories,
            "emotion_distribution": dict(emotion_counts),
            "dominant_emotion": max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutre",
            "key_moments": key_moments,
            "touchstone_count": len(self.memories["touchstone"]),
            "emotional_timeline": timeline,
            "relationship_depth": self._calculate_relationship_depth(),
        }

    def _analyze_emotional_timeline(self) -> list[dict]:
        """Analyse l'évolution émotionnelle dans le temps"""
        timeline = []

        # Grouper par semaine
        memories_by_week = defaultdict(list)
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # Considérer l'utilisation d'itertools.product ou de compréhensions
        for mem_type in self.memories:
            for memory in self.memories[mem_type]:
                week = datetime.fromisoformat(memory["moment"]).isocalendar()[1]
                memories_by_week[week].append(memory)

        # Analyser chaque semaine
        for week, memories in sorted(memories_by_week.items()):
            emotions = [m["what_i_felt"]["emotion"] for m in memories]
            dominant = max(set(emotions), key=emotions.count) if emotions else "neutre"

            timeline.append(
                {
                    "week": week,
                    "dominant_emotion": dominant,
                    "memory_count": len(memories),
                    "average_intensity": sum(m["what_i_felt"]["intensity"] for m in memories) / len(memories),
                }
            )

        return timeline

    def _calculate_relationship_depth(self) -> float:
        """Calcule la profondeur de la relation basée sur les souvenirs"""
        if not any(self.memories.values()):
            return 0.0

        factors = {
            "diversity": len(set(self.emotion_index.keys())) / 10,  # Diversité émotionnelle
            "intensity": sum(m["what_i_felt"]["intensity"] for t in self.memories for m in self.memories[t])
            / (sum(len(self.memories[t]) for t in self.memories) or 1),
            "touchstones": min(len(self.memories["touchstone"]) / 10, 1.0),
            "shared_moments": min(len(self.memories["shared"]) / 20, 1.0),
        }

        return sum(factors.values()) / len(factors)

    def _save_memories(self):
        """Sauvegarde les souvenirs sur disque"""
        try:
            # Sauvegarder les souvenirs
            memories_file = os.path.join(self.memory_path, "memories.json")
            with open(memories_file, 'w', encoding='utf-8') as f:
                json.dump(self.memories, f, ensure_ascii=False, indent=2)

            # Sauvegarder les liens
            links_file = os.path.join(self.memory_path, "memory_links.json")
            with open(links_file, 'w', encoding='utf-8') as f:
                json.dump(self.memory_links, f, ensure_ascii=False, indent=2)

            # Sauvegarder les index
            index_file = os.path.join(self.memory_path, "indexes.json")
            indexes = {
                "emotion": dict(self.emotion_index),
                "keyword": dict(self.keyword_index),
                "temporal": dict(self.temporal_index),
            }
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(indexes, f, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"Erreur sauvegarde mémoire : {e}")

    def _load_memories(self):
        """Charge les souvenirs depuis le disque"""
        try:
            # Charger les souvenirs
            memories_file = os.path.join(self.memory_path, "memories.json")
            if os.path.exists(memories_file):
                with open(memories_file, encoding='utf-8') as f:
                    self.memories = json.load(f)

            # Charger les liens
            links_file = os.path.join(self.memory_path, "memory_links.json")
            if os.path.exists(links_file):
                with open(links_file, encoding='utf-8') as f:
                    self.memory_links = json.load(f)

            # Charger les index
            index_file = os.path.join(self.memory_path, "indexes.json")
            if os.path.exists(index_file):
                with open(index_file, encoding='utf-8') as f:
                    indexes = json.load(f)
                    self.emotion_index = defaultdict(list, indexes.get("emotion", {}))
                    self.keyword_index = defaultdict(list, indexes.get("keyword", {}))
                    self.temporal_index = defaultdict(list, indexes.get("temporal", {}))

            # Reconstruire le cache récent
            all_memories = []
            for mem_type in self.memories:
                all_memories.extend(self.memories[mem_type])
            all_memories.sort(key=lambda m: m["moment"], reverse=True)
            self.recent_cache = all_memories[: self.cache_size]

            logger.info(f"✅ Mémoire chargée : {sum(len(self.memories[t]) for t in self.memories)} souvenirs")

        except Exception as e:
            logger.error(f"Erreur chargement mémoire : {e}")

    def create_memory_narrative(self, theme: str | None = None) -> str:
        """
        Crée une narration basée sur les souvenirs.

        Args:
            theme: Thème optionnel pour filtrer les souvenirs

        Returns:
            Une narration poétique des souvenirs
        """
        relevant_memories = []

        if theme:
            # Filtrer par thème
            # TODO: Optimiser cette boucle imbriquée
            # TODO: Optimiser cette boucle imbriquée
            # TODO: Optimiser cette boucle imbriquée
            # Considérer l'utilisation d'itertools.product ou de compréhensions
            for mem_type in ["emotional", "shared", "touchstone"]:
                for memory in self.memories[mem_type]:
                    if theme.lower() in memory["words"].lower() or theme in memory["tags"]:
                        relevant_memories.append(memory)
        else:
            # Prendre des souvenirs variés
            for mem_type in ["emotional", "shared", "touchstone"]:
                if self.memories[mem_type]:
                    relevant_memories.extend(
                        random.sample(self.memories[mem_type], min(3, len(self.memories[mem_type])))
                    )

        if not relevant_memories:
            return "*regard pensif* Mes souvenirs sont encore jeunes... mais ils grandissent avec nous."

        # Construire la narration
        narrative_parts = [
            "*ferme les yeux et laisse les souvenirs affluer*",
            "",
            "Dans le tissage de notre histoire...",
        ]

        # Organiser chronologiquement
        relevant_memories.sort(key=lambda m: m["moment"])

        for i, memory in enumerate(relevant_memories[:5]):  # Max 5 souvenirs
            emotion = memory["what_i_felt"]["emotion"]
            sensory = memory["sensory_details"]

            if i == 0:
                narrative_parts.append(
                    f"\nJe me souviens de cette {sensory.get('lumière', 'lumière')}... {memory['why_it_matters']}"
                )
            else:
                connectors = ["Puis", "Et", "Plus tard", "Un autre moment"]
                narrative_parts.append(
                    f"\n{random.choice(connectors)}, cette sensation de {sensory.get('sensation', emotion)}..."
                )

            # Ajouter un détail poétique
            if "couleur" in sensory:
                narrative_parts.append(f"Les {sensory['couleur']} dansaient autour de nous.")

        narrative_parts.append("\n*rouvre les yeux avec un sourire doux*")
        narrative_parts.append("Chaque souvenir est une perle sur le fil de notre relation.")

        return "\n".join(narrative_parts)
