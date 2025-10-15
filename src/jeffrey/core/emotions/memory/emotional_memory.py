#!/usr/bin/env python

"""
emotional_memory.py - Gestion de la mémoire émotionnelle pour Jeffrey

Ce module définit la classe EmotionalMemory utilisée par Jeffrey pour stocker,
manipuler et enrichir sa mémoire émotionnelle. Cette mémoire constitue le socle
des réactions émotionnelles contextuelles et permet à Jeffrey de maintenir une
continuité émotionnelle à travers les interactions avec l'utilisateur.

La mémoire émotionnelle enregistre les expériences affectives significatives,
permettant ainsi de créer un historique personnel qui influence les réactions
futures et la "personnalité" émergente de Jeffrey.

Sprint 11: Mise en œuvre d'une mémoire émotionnelle réaliste avec mécanismes d'oubli,
de renforcement et de feedback utilisateur pour simuler une mémoire humaine.

Sprint 12: Ajout de fonctionnalités de visualisation et d'exportation des données
émotionnelles pour faciliter la compréhension de l'état affectif.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from typing import Any


class EmotionalMemory:
    """
    Classe de gestion de la mémoire émotionnelle pour Jeffrey.

    Cette classe permet de stocker, charger, manipuler et enrichir
    la mémoire émotionnelle de Jeffrey. Elle constitue un composant
    fondamental de son système émotionnel, permettant une persistance
    des expériences affectives.

    Attributes:
        data (Dict): Dictionnaire contenant les données de mémoire émotionnelle
        filepath (str): Chemin du fichier de sauvegarde de la mémoire
        io_manager: Gestionnaire d'entrée/sortie pour charger/sauvegarder
        logger: Logger pour tracer les opérations sur la mémoire
    """

    def __init__(self, filepath: str = "data/memory/emotional_memory.json", io_manager=None) -> None:
        """
        Initialise la mémoire émotionnelle.

        Args:
            filepath (str): Chemin du fichier de sauvegarde
            io_manager: Gestionnaire d'entrée/sortie pour les opérations fichier
        """
        self.logger = logging.getLogger("jeffrey.emotional_memory")
        self.filepath = filepath
        self.io_manager = io_manager

        # Structure de données par défaut
        self.data = {
            "memories": [],
            "stats": {
                "total_entries": 0,
                "emotion_counts": {},
                "last_updated": datetime.now().isoformat(),
            },
            "metadata": {"version": "1.0", "created": datetime.now().isoformat()},
        }

        # Charger les données si disponibles
        try:
            self.load_memory()
        except Exception as e:
            self.logger.warning(f"Impossible de charger la mémoire émotionnelle: {e}")
            self.logger.info("Initialisation d'une nouvelle mémoire émotionnelle")

    def load_memory(self) -> bool:
        """
        Charge les données de mémoire depuis un fichier JSON.

        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            if self.io_manager:
                # Utiliser le gestionnaire d'E/S
                memory_data = self.io_manager.load_data(self.filepath, default_data={})
            else:
                # Chargement direct du fichier
                with open(self.filepath, encoding="utf-8") as f:
                    memory_data = json.load(f)

            if memory_data:
                self.data = memory_data
                self.logger.info(f"Mémoire émotionnelle chargée: {len(self.data['memories'])} entrées")
                return True
            return False
        except FileNotFoundError:
            self.logger.info(f"Fichier de mémoire émotionnelle non trouvé: {self.filepath}")
            return False
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la mémoire émotionnelle: {e}")
            return False

    def save_memory(self) -> bool:
        """
        Sauvegarde la mémoire émotionnelle actuelle.

        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        try:
            # Mettre à jour les métadonnées
            self.data["stats"]["last_updated"] = datetime.now().isoformat()
            self.data["stats"]["total_entries"] = len(self.data["memories"])

            if self.io_manager:
                # Utiliser le gestionnaire d'E/S
                self.io_manager.save_data(self.data, self.filepath)
            else:
                # Sauvegarde directe dans le fichier
                with open(self.filepath, "w", encoding="utf-8") as f:
                    json.dump(self.data, f, ensure_ascii=False, indent=2)

            self.logger.info(f"Mémoire émotionnelle sauvegardée: {len(self.data['memories'])} entrées")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde de la mémoire émotionnelle: {e}")
            return False

    def inject_memory(
        self,
        emotion: str,
        intensity: float,
        context: dict[str, Any],
        valence: float = 0.0,
        source: str = "interaction",
    ) -> dict[str, Any]:
        """
        Ajoute une entrée dans le journal émotionnel.

        Args:
            emotion (str): Émotion ressentie
            intensity (float): Intensité de l'émotion (0.0 à 1.0)
            context (Dict): Contexte associé à cette émotion
            valence (float): Valence de l'émotion (-1.0 à 1.0)
            source (str): Source de cette mémoire émotionnelle

        Returns:
            Dict: Entrée créée dans la mémoire émotionnelle
        """
        # Créer une nouvelle entrée avec les nouveaux champs du Sprint 11
        entry = {
            "id": f"em_{len(self.data['memories']) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "emotion": emotion,
            "intensity": max(0.0, min(1.0, intensity)),  # Limiter entre 0 et 1
            "valence": max(-1.0, min(1.0, valence)),  # Limiter entre -1 et 1
            "context": context,
            "source": source,
            "reactivation_count": 0,
            "last_reactivated": None,
            # Sprint 11: Nouveaux champs pour une mémoire plus réaliste
            "score": 1.0,  # Importance émotionnelle initiale
            "last_used": datetime.now().isoformat(),  # Date de dernière utilisation
            "lifespan": 90,  # Durée de vie estimée en jours
        }

        # Ajouter à la mémoire
        self.data["memories"].append(entry)

        # Mettre à jour les statistiques
        if emotion in self.data["stats"]["emotion_counts"]:
            self.data["stats"]["emotion_counts"][emotion] += 1
        else:
            self.data["stats"]["emotion_counts"][emotion] = 1

        self.logger.info(f"Mémoire émotionnelle injectée: {emotion} (intensité: {intensity:.2f})")

        # Sauvegarder après injection
        self.save_memory()

        return entry

    def clear_memory(self, save_backup: bool = True) -> bool:
        """
        Réinitialise la mémoire émotionnelle.

        Args:
            save_backup (bool): Si True, sauvegarde une copie avant effacement

        Returns:
            bool: True si l'opération a réussi, False sinon
        """
        try:
            if save_backup and self.data["memories"]:
                backup_path = f"{self.filepath}.backup.{datetime.now().strftime('%Y%m%d%H%M%S')}.json"

                if self.io_manager:
                    self.io_manager.save_data(self.data, backup_path)
                else:
                    with open(backup_path, "w", encoding="utf-8") as f:
                        json.dump(self.data, f, ensure_ascii=False, indent=2)

                self.logger.info(f"Sauvegarde de la mémoire émotionnelle créée: {backup_path}")

            # Réinitialiser les données
            self.data = {
                "memories": [],
                "stats": {
                    "total_entries": 0,
                    "emotion_counts": {},
                    "last_updated": datetime.now().isoformat(),
                },
                "metadata": {
                    "version": "1.0",
                    "created": datetime.now().isoformat(),
                    "reset": True,
                },
            }

            # Sauvegarder l'état vide
            self.save_memory()

            self.logger.warning("Mémoire émotionnelle réinitialisée")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la réinitialisation de la mémoire émotionnelle: {e}")
            return False

    def get_memories_by_emotion(self, emotion: str, limit: int = 5) -> list[dict[str, Any]]:
        """
        Récupère les souvenirs associés à une émotion spécifique.

        Args:
            emotion (str): Émotion recherchée
            limit (int): Nombre maximum de résultats à retourner

        Returns:
            List[Dict]: Liste des souvenirs correspondants
        """
        matching_memories = [m for m in self.data["memories"] if m["emotion"] == emotion]
        matching_memories.sort(key=lambda x: x["intensity"], reverse=True)
        return matching_memories[:limit]

    def get_recent_memories(self, limit: int = 10) -> list[dict[str, Any]]:
        """
        Récupère les souvenirs les plus récents.

        Args:
            limit (int): Nombre maximum de résultats à retourner

        Returns:
            List[Dict]: Liste des souvenirs récents
        """
        sorted_memories = sorted(self.data["memories"], key=lambda x: x["timestamp"], reverse=True)
        return sorted_memories[:limit]

    def reactivate_memory(self, memory_id: str) -> dict[str, Any] | None:
        """
        Réactive un souvenir spécifique pour renforcer sa trace.

        Args:
            memory_id (str): Identifiant du souvenir à réactiver

        Returns:
            Dict or None: Le souvenir réactivé ou None si non trouvé
        """
        for memory in self.data["memories"]:
            if memory["id"] == memory_id:
                memory["reactivation_count"] += 1
                memory["last_reactivated"] = datetime.now().isoformat()
                self.save_memory()
                return memory

        return None

    def reactivate_similar_memory(
        self, primary: str = None, valence: float = 0.0, threshold: float = 0.3
    ) -> dict[str, Any] | None:
        """
        Réactive un souvenir ayant une émotion similaire si trouvé dans la mémoire.

        Args:
            primary (str): Émotion primaire à rechercher (optionnel)
            valence (float): Valence émotionnelle actuelle
            threshold (float): Écart maximal autorisé pour considérer un souvenir comme similaire

        Returns:
            Dict or None: Le souvenir réactivé ou None
        """
        # Utiliser primary si fourni, sinon utiliser l'ancien paramètre par convention
        emotion = primary if primary is not None else valence

        candidates = []
        for memory in self.data["memories"]:
            if emotion and memory["emotion"] == emotion:
                candidates.append(memory)
            elif abs(memory["valence"] - valence) <= threshold:
                candidates.append(memory)

        if not candidates:
            return None

        # Choisir le plus récent ou celui avec le score le plus élevé (pondération)
        for candidate in candidates:
            # Calculer un score combiné basé sur la récence et l'importance
            recency_weight = 0.6
            score_weight = 0.4

            # Convertir la date en objet datetime
            last_used = (
                datetime.fromisoformat(candidate["last_used"])
                if "last_used" in candidate
                else datetime.fromisoformat(candidate["timestamp"])
            )
            days_ago = (datetime.now() - last_used).days
            recency_score = 1.0 / (1.0 + days_ago / 10)  # Plus récent = plus haut score

            candidate["_temp_score"] = (recency_weight * recency_score) + (score_weight * candidate.get("score", 1.0))

        # Sélectionner le candidat avec le score combiné le plus élevé
        selected = sorted(candidates, key=lambda m: m["_temp_score"], reverse=True)[0]

        # Incrémenter le score du souvenir et mettre à jour last_used (Sprint 11)
        memory_id = selected["id"]
        for memory in self.data["memories"]:
            if memory["id"] == memory_id:
                memory["score"] = memory.get("score", 1.0) + 1.0  # Incrémenter le score
                memory["last_used"] = datetime.now().isoformat()  # Mettre à jour la date de dernière utilisation
                break

        return self.reactivate_memory(memory_id)

    def decay_old_memories(self, decay_threshold_days: int = 30, min_score: float = 0.5) -> int:
        """
        Supprime ou réduit l'importance des souvenirs anciens ou peu importants.

        Args:
            decay_threshold_days (int): Nombre de jours après lesquels commencer la dégradation
            min_score (float): Score minimal en dessous duquel un souvenir peut être supprimé

        Returns:
            int: Nombre de souvenirs affectés (supprimés ou dégradés)
        """
        affected_count = 0
        now = datetime.now()
        memories_to_remove = []

        for memory in self.data["memories"]:
            # Obtenir la date de dernière utilisation
            last_used = datetime.fromisoformat(memory.get("last_used", memory["timestamp"]))

            # Obtenir la durée de vie du souvenir
            lifespan_days = memory.get("lifespan", 90)

            # Calculer l'âge en jours
            age_days = (now - last_used).days

            # Vérifier si le souvenir est trop ancien par rapport à sa durée de vie
            if age_days > lifespan_days:
                # Souvenir trop ancien - à supprimer
                memories_to_remove.append(memory)
                affected_count += 1
                self.logger.debug(f"Mémoire trop ancienne supprimée: {memory['id']}, âge: {age_days} jours")

            # Vérifier si le souvenir est en phase de dégradation
            elif age_days > decay_threshold_days:
                # Calculer le facteur de dégradation (plus vieux = plus de dégradation)
                decay_factor = (age_days - decay_threshold_days) / (lifespan_days - decay_threshold_days)
                decay_factor = min(0.9, max(0.1, decay_factor))  # Limiter entre 0.1 et 0.9

                # Dégrader progressivement le score
                old_score = memory.get("score", 1.0)
                new_score = old_score * (1.0 - (decay_factor * 0.2))  # Réduction maximale de 20% par cycle

                # Vérifier si le score est trop bas
                if new_score < min_score:
                    # Score trop bas - à supprimer
                    memories_to_remove.append(memory)
                    affected_count += 1
                    self.logger.debug(f"Mémoire à faible score supprimée: {memory['id']}, score: {new_score:.2f}")
                else:
                    # Mettre à jour le score
                    memory["score"] = new_score
                    affected_count += 1
                    self.logger.debug(f"Score de mémoire dégradé: {memory['id']}, nouveau score: {new_score:.2f}")

        # Supprimer les souvenirs marqués
        for memory in memories_to_remove:
            self.data["memories"].remove(memory)

        # Mettre à jour les statistiques
        self.data["stats"]["total_entries"] = len(self.data["memories"])

        # Recalculer les comptages d'émotions
        emotion_counts = {}
        for memory in self.data["memories"]:
            emotion = memory["emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        self.data["stats"]["emotion_counts"] = emotion_counts

        return affected_count

    def save_memory_if_needed(self, modified_entries: int = 1) -> None:
        """
        Sauvegarde la mémoire seulement si un certain nombre de modifications ont été faites.
        Déclenche également le processus de dégradation des souvenirs anciens.

        Args:
            modified_entries (int): Seuil de modifications déclenchant la sauvegarde
        """
        total = self.data["stats"]["total_entries"]
        if total % modified_entries == 0:
            self.logger.info("Sauvegarde conditionnelle déclenchée.")

            # Appliquer la dégradation/oubli des souvenirs (Sprint 11)
            if random.random() < 0.3:  # 30% de chance de déclencher le processus de dégradation
                affected = self.decay_old_memories()
                if affected > 0:
                    self.logger.info(f"Processus d'oubli déclenché: {affected} souvenirs affectés")

            # Sauvegarder la mémoire
            self.save_memory()

    def summarize_emotions(self) -> dict[str, int]:
        """
        Fournit une synthèse des émotions stockées dans la mémoire.

        Returns:
            Dict[str, int]: Comptage des émotions
        """
        return dict(self.data["stats"]["emotion_counts"])

    def get_emotional_stats(self) -> dict[str, Any]:
        """
        Génère un résumé statistique de l'état émotionnel.

        Returns:
            Dict: Résumé JSON des statistiques émotionnelles incluant:
                - total_memories: nombre total de souvenirs
                - average_score: score moyen des souvenirs
                - most_frequent: émotions les plus fréquentes (top 3)
                - recent_percentage: pourcentage de souvenirs récents (<7 jours)
                - valence_distribution: distribution des valences
                - intensity_stats: statistiques d'intensité émotionnelle
        """
        if not self.data["memories"]:
            return {
                "total_memories": 0,
                "average_score": 0,
                "most_frequent": {},
                "recent_percentage": 0,
                "valence_distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "intensity_stats": {"min": 0, "max": 0, "avg": 0},
            }

        now = datetime.now()
        total = len(self.data["memories"])

        # Calcul du score moyen
        avg_score = sum(memory.get("score", 1.0) for memory in self.data["memories"]) / total

        # Top émotions les plus fréquentes
        emotion_counts = self.data["stats"]["emotion_counts"]
        if emotion_counts:
            top_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:3]
            top_emotions_dict = {emotion: count for emotion, count in top_emotions}
        else:
            top_emotions_dict = {}

        # Pourcentage de souvenirs récents (moins de 7 jours)
        recent_count = 0
        for memory in self.data["memories"]:
            last_used = datetime.fromisoformat(memory.get("last_used", memory["timestamp"]))
            if (now - last_used).days < 7:
                recent_count += 1

        recent_percentage = (recent_count / total) * 100 if total > 0 else 0

        # Distribution des valences
        valence_distribution = {"positive": 0, "neutral": 0, "negative": 0}
        for memory in self.data["memories"]:
            valence = memory["valence"]
            if valence > 0.2:
                valence_distribution["positive"] += 1
            elif valence < -0.2:
                valence_distribution["negative"] += 1
            else:
                valence_distribution["neutral"] += 1

        # Convertir en pourcentages
        for key in valence_distribution:
            valence_distribution[key] = (valence_distribution[key] / total) * 100 if total > 0 else 0

        # Statistiques d'intensité
        intensities = [memory["intensity"] for memory in self.data["memories"]]
        intensity_stats = {
            "min": min(intensities) if intensities else 0,
            "max": max(intensities) if intensities else 0,
            "avg": sum(intensities) / len(intensities) if intensities else 0,
        }

        return {
            "total_memories": total,
            "average_score": round(avg_score, 2),
            "most_frequent": top_emotions_dict,
            "recent_percentage": round(recent_percentage, 1),
            "valence_distribution": {k: round(v, 1) for k, v in valence_distribution.items()},
            "intensity_stats": {k: round(v, 2) for k, v in intensity_stats.items()},
        }

    def add_feedback_to_memory(self, emotion_type: str, feedback_score: float) -> int:
        """
        Augmente ou réduit le score d'un souvenir selon un feedback positif ou négatif.

        Args:
            emotion_type (str): Type d'émotion à modifier
            feedback_score (float): Score de feedback entre 0.0 (très négatif) et 1.0 (très positif)

        Returns:
            int: Nombre de souvenirs modifiés
        """
        if not 0.0 <= feedback_score <= 1.0:
            feedback_score = max(0.0, min(1.0, feedback_score))
            self.logger.warning(f"Feedback score ajusté à {feedback_score} (doit être entre 0.0 et 1.0)")

        # Normaliser le score de feedback pour l'ajustement de -0.5 à +0.5
        adjustment = (feedback_score - 0.5) * 2  # -1.0 à +1.0
        adjustment = adjustment * 0.3  # Limiter l'impact à ±30%

        count_modified = 0

        # Trouver les souvenirs correspondant à ce type d'émotion
        for memory in self.data["memories"]:
            if memory["emotion"] == emotion_type:
                old_score = memory.get("score", 1.0)

                # Appliquer l'ajustement
                if adjustment > 0:
                    # Feedback positif: augmenter le score (max 2.0)
                    new_score = min(2.0, old_score * (1.0 + adjustment))
                else:
                    # Feedback négatif: diminuer le score (min 0.1)
                    new_score = max(0.1, old_score * (1.0 + adjustment))

                memory["score"] = new_score
                memory["feedback_applied"] = True
                memory["last_used"] = datetime.now().isoformat()
                count_modified += 1

                self.logger.debug(f"Score de mémoire ajusté: {memory['id']}, {old_score:.2f} → {new_score:.2f}")

        # Sauvegarder après modification
        if count_modified > 0:
            self.save_memory()

        return count_modified
