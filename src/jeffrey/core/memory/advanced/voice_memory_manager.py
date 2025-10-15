#!/usr/bin/env python
"""
voice_memory_manager.py - Gestionnaire de mémoire vocale pour Jeffrey

Ce module gère la mémoire vocale et les interactions de Jeffrey, permettant de:
- Enregistrer les interactions vocales et émotionnelles
- Stocker les données émotionnelles liées à la voix
- Journaliser les interactions vocales au format JSONL
- Calculer des statistiques sur les émotions dans différents contextes
"""

from __future__ import annotations

import datetime
import json
import logging

# V1.1 PATCH - Ajout de méthodes de nettoyage pour résoudre INT-COMP-04
import os
import uuid
from collections import defaultdict
from typing import Any

# V1.1 PATCH - Ajout du logger pour le nettoyage mémoire
logger = logging.getLogger("voice.memory")


class VoiceMemoryManager:
    """
    Gestionnaire de mémoire vocale et de journalisation des interactions.
    Combine les fonctionnalités des anciennes versions et ajoute la journalisation JSONL.
    """

    def __init__(self, filepath: str = "data/memory/voice_memory.json") -> None:
        """
        Initialise le gestionnaire de mémoire vocale.

        Args:
            filepath: Chemin vers le fichier de mémoire vocale
        """
        self.filepath = filepath
        self.log_file_path = "data/voice_logs/voice_interactions.jsonl"

        # Créer les répertoires nécessaires
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

        # Charger la mémoire persistante
        self.memory = self._load_memory()

        # Mémoire pour les enregistrements émotionnels détaillés
        self.emotional_records = defaultdict(list)

    def _load_memory(self) -> dict:
        """
        Charge la mémoire vocale depuis le fichier.

        Returns:
            Dict: Mémoire vocale chargée ou un dictionnaire vide en cas d'erreur
        """
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {}
        return {}

    def save(self) -> None:
        """
        Sauvegarde la mémoire vocale dans le fichier.
        """
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2)

    def update_emotion(self, emotion: str, intensity: float) -> None:
        """
        Met à jour la mémoire émotionnelle pour une émotion donnée.

        Args:
            emotion: Nom de l'émotion
            intensity: Intensité de l'émotion (0.0 à 1.0)
        """
        history = self.memory.get(emotion, {"count": 0, "avg_intensity": 0.0})
        count = history["count"] + 1
        new_avg = ((history["avg_intensity"] * history["count"]) + intensity) / count
        self.memory[emotion] = {"count": count, "avg_intensity": new_avg}
        self.save()

    def get_memory(self, emotion: str) -> dict:
        """
        Récupère la mémoire pour une émotion donnée.

        Args:
            emotion: Nom de l'émotion

        Returns:
            Dict: Informations sur l'émotion (nombre d'occurrences et intensité moyenne)
        """
        return self.memory.get(emotion, {"count": 0, "avg_intensity": 0.5})

    def log_voice_interaction(self, prompt: str, emotion_state: dict[str, Any] | None = None) -> None:
        """
        Journalise une interaction vocale au format JSONL.

        Args:
            prompt: Texte de l'interaction vocale
            emotion_state: État émotionnel associé à l'interaction (optionnel)
        """
        try:
            entry = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "text": prompt,
                "emotion_state": emotion_state or {},
            }

            with open(self.log_file_path, "a", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False)
                f.write("\n")

        except Exception as e:
            print(f"[Erreur] Impossible d'enregistrer l'interaction vocale : {e}")

    # ----- Fonctionnalités avancées de tracking émotionnel (v2) -----

    def add_emotion_record(self, emotion: str, score: float, context: str = "", interaction_type: str = "") -> str:
        """
        Ajoute une trace émotionnelle détaillée avec contexte et type d'interaction.

        Args:
            emotion: Nom de l'émotion
            score: Score de l'émotion (0.0 à 1.0)
            context: Contexte de l'interaction
            interaction_type: Type d'interaction (voix, texte, etc.)

        Returns:
            str: ID unique de l'enregistrement créé
        """
        record_id = str(uuid.uuid4())
        record = {
            "id": record_id,
            "score": score,
            "context": context,
            "interaction_type": interaction_type,
            "timestamp": datetime.datetime.now(),
        }

        self.emotional_records[emotion].append(record)

        # Mettre également à jour la mémoire agrégée
        self.update_emotion(emotion, score)

        return record_id

    def delete_record_by_id(self, emotion: str, record_id: str) -> bool:
        """
        Supprime un enregistrement émotionnel précis à partir de son ID.

        Args:
            emotion: Nom de l'émotion
            record_id: ID unique de l'enregistrement à supprimer

        Returns:
            bool: True si supprimé, False sinon
        """
        records = self.emotional_records.get(emotion, [])
        for i, record in enumerate(records):
            if record.get("id") == record_id:
                del records[i]
                return True
        return False

    def get_average_score(self, emotion: str) -> float:
        """
        Retourne le score moyen d'une émotion, tous contextes confondus.

        Args:
            emotion: Nom de l'émotion

        Returns:
            float: Score moyen (0.0 à 1.0)
        """
        records = self.emotional_records.get(emotion, [])
        if not records:
            return 0.0
        return sum([r["score"] for r in records]) / len(records)

    def get_contextual_emotion_score(self, emotion: str, context: str = "", interaction_type: str = "") -> float:
        """
        Calcule une moyenne filtrée selon le contexte et/ou type d'interaction.

        Args:
            emotion: Nom de l'émotion
            context: Contexte de l'interaction (optionnel)
            interaction_type: Type d'interaction (optionnel)

        Returns:
            float: Score moyen filtré (0.0 à 1.0)
        """
        filtered = [
            r["score"]
            for r in self.emotional_records.get(emotion, [])
            if (not context or r["context"] == context)
            and (not interaction_type or r["interaction_type"] == interaction_type)
        ]

        if not filtered:
            return 0.0
        return sum(filtered) / len(filtered)

    def reset_memory(self) -> None:
        """
        Réinitialise complètement la mémoire détaillée des émotions.
        """
        self.emotional_records.clear()

    def export_emotional_records(self) -> dict:
        """
        Exporte les enregistrements émotionnels détaillés avec timestamps sérialisés.

        Returns:
            Dict: Mémoire émotionnelle exportée
        """
        exported = {}
        for emotion, records in self.emotional_records.items():
            # Convertir l'objet datetime en chaîne ISO pour la sérialisation
            exported[emotion] = []
            for r in records:
                record_copy = r.copy()
                if isinstance(record_copy["timestamp"], datetime.datetime):
                    record_copy["timestamp"] = record_copy["timestamp"].isoformat()
                exported[emotion].append(record_copy)
        return exported

    def load_emotional_records(self, data: dict) -> None:
        """
        Charge des enregistrements émotionnels détaillés depuis un dictionnaire.

        Args:
            data: Dictionnaire d'enregistrements émotionnels
        """
        for emotion, records in data.items():
            self.emotional_records[emotion] = []
            for record in records:
                # Convertir le timestamp ISO en objet datetime
                if isinstance(record.get("timestamp"), str):
                    try:
                        record["timestamp"] = datetime.datetime.fromisoformat(record["timestamp"])
                    except ValueError:
                        # En cas d'erreur, utiliser la date/heure actuelle
                        record["timestamp"] = datetime.datetime.now()
                self.emotional_records[emotion].append(record)

    def get_recent_interactions(self, limit: int = 10) -> list[dict]:
        """
        Récupère les interactions vocales récentes depuis le fichier JSONL.

        Args:
            limit: Nombre maximal d'interactions à récupérer

        Returns:
            List[Dict]: Liste des interactions récentes
        """
        interactions = []

        try:
            if os.path.exists(self.log_file_path):
                with open(self.log_file_path, encoding="utf-8") as f:
                    lines = f.readlines()
                    # Prendre les 'limit' dernières lignes
                    for line in lines[-limit:]:
                        try:
                            interaction = json.loads(line.strip())
                            interactions.append(interaction)
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            print(f"[Erreur] Impossible de lire les interactions vocales : {e}")

        return interactions

    # V1.1 PATCH - Méthodes de nettoyage mémoire pour résoudre INT-COMP-04
    def cleanup_old_records(self, max_age_days=30):
        """
        Supprime les enregistrements émotionnels détaillés plus anciens que max_age_days.

        Args:
            max_age_days (int): Âge maximum en jours des enregistrements à conserver

        Returns:
            int: Nombre d'enregistrements supprimés
        """
        total_removed = 0
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=max_age_days)

        for emotion in list(self.emotional_records.keys()):
            before_count = len(self.emotional_records[emotion])

            # Ne conserver que les enregistrements plus récents que la date limite
            self.emotional_records[emotion] = [
                record
                for record in self.emotional_records[emotion]
                if record.get("timestamp", datetime.datetime.now()) > cutoff_date
            ]

            removed = before_count - len(self.emotional_records[emotion])
            total_removed += removed

            # Si tous les enregistrements ont été supprimés, supprimer l'émotion
            if not self.emotional_records[emotion]:
                del self.emotional_records[emotion]

        # Sauvegarder les changements si des enregistrements ont été supprimés
        if total_removed > 0:
            self.save_all_memories()
            logger.info(f"Nettoyage mémoire vocale: {total_removed} enregistrements émotionnels supprimés")

        return total_removed

    def limit_log_size(self, max_entries=1000):
        """
        Limite la taille du fichier de log des interactions vocales
        en conservant uniquement les entrées les plus récentes.

        Args:
            max_entries (int): Nombre maximum d'entrées à conserver

        Returns:
            int: Nombre d'entrées supprimées ou -1 en cas d'erreur
        """
        try:
            if not os.path.exists(self.log_file_path):
                return 0

            with open(self.log_file_path, encoding="utf-8") as f:
                lines = f.readlines()

            before_count = len(lines)
            if before_count <= max_entries:
                return 0

            # Conserver uniquement les max_entries dernières lignes
            with open(self.log_file_path, "w", encoding="utf-8") as f:
                for line in lines[-max_entries:]:
                    f.write(line)

            removed = before_count - max_entries
            logger.info(f"Nettoyage mémoire vocale: {removed} entrées de log supprimées")
            return removed
        except Exception as e:
            logger.error(f"[Erreur] Impossible de limiter la taille du fichier log: {e}")
            return -1

    def optimize_memory(self):
        """
        Effectue une optimisation complète de la mémoire vocale.

        Returns:
            dict: Statistiques d'optimisation
        """
        logger.info("Démarrage de l'optimisation mémoire vocale...")
        stats = {
            "records_removed": self.cleanup_old_records(max_age_days=60),
            "log_entries_removed": self.limit_log_size(max_entries=500),
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # Forcer un nettoyage complet
        self.save_all_memories()

        logger.info(f"Optimisation mémoire vocale terminée: {stats}")
        return stats

    def save_all_memories(self) -> bool:
        """
        Sauvegarde l'ensemble des mémoires vocales et émotionnelles.
        Cette méthode effectue une sauvegarde complète de toutes les données gérées par ce gestionnaire:
        - Mémoire vocale standard
        - Enregistrements émotionnels détaillés
        - Historique des interactions (optionnel)

        Returns:
            bool: True si la sauvegarde a réussi, False en cas d'erreur
        """
        try:
            # 1. Sauvegarde de la mémoire vocale principale
            self.save()

            # 2. Sauvegarde des enregistrements émotionnels détaillés
            emotional_records_path = os.path.join(os.path.dirname(self.filepath), "voice_emotional_records.json")
            os.makedirs(os.path.dirname(emotional_records_path), exist_ok=True)

            with open(emotional_records_path, "w", encoding="utf-8") as f:
                json.dump(self.export_emotional_records(), f, indent=2, ensure_ascii=False)

            # 3. Création d'une archive de l'historique des interactions
            # (pour éviter que le fichier JSONL ne devienne trop volumineux)
            try:
                if os.path.exists(self.log_file_path) and os.path.getsize(self.log_file_path) > 1024 * 1024:  # 1 Mo
                    archive_dir = os.path.join(os.path.dirname(self.log_file_path), "archives")
                    os.makedirs(archive_dir, exist_ok=True)

                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    archive_path = os.path.join(archive_dir, f"voice_{timestamp}.json")

                    # Lire l'ancien fichier JSONL
                    interactions = []
                    with open(self.log_file_path, encoding="utf-8") as f:
                        for line in f:
                            try:
                                interactions.append(json.loads(line.strip()))
                            except json.JSONDecodeError:
                                continue

                    # Sauvegarder dans l'archive au format JSON
                    with open(archive_path, "w", encoding="utf-8") as f:
                        json.dump(interactions, f, indent=2, ensure_ascii=False)

                    # Tronquer le fichier JSONL en gardant les 100 dernières entrées
                    with open(self.log_file_path, "w", encoding="utf-8") as f:
                        for interaction in interactions[-100:]:
                            json.dump(interaction, f, ensure_ascii=False)
                            f.write("\n")
            except Exception as archive_err:
                print(f"[Avertissement] Erreur lors de l'archivage des interactions vocales : {archive_err}")
                # On continue même en cas d'erreur d'archivage

            return True

        except Exception as e:
            print(f"[Erreur] Impossible de sauvegarder toutes les mémoires vocales : {e}")
            return False


# Instance singleton pour utilisation globale
voice_memory_manager = VoiceMemoryManager()
