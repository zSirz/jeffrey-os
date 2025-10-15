#!/usr/bin/env python3

"""
Jeffrey Memory Synchronization System
Système de synchronisation et persistance de la mémoire entre appareils.
Gère la sauvegarde atomique et la fusion des états de mémoire.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("jeffrey.memory_sync")


class JeffreyMemorySync:
    """Synchronisation mémoire entre appareils via iCloud/stockage partagé"""

    def __init__(self, memory_path: str = "Jeffrey_Memoire") -> None:
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(exist_ok=True)

        # Fichiers de synchronisation
        self.sync_file = self.memory_path / "jeffrey_unified_memory.json"
        self.lock_file = self.memory_path / "memory.lock"
        self.conflict_dir = self.memory_path / "conflicts"
        self.conflict_dir.mkdir(exist_ok=True)

        # Configuration de synchronisation
        self.sync_config = {
            "auto_sync": True,
            "sync_interval": 300,  # 5 minutes
            "max_backups": 10,
            "compression": False,
            "encryption": False,  # Pour l'instant
        }

        # État de synchronisation
        self.sync_state = {
            "last_sync": None,
            "device_id": self._get_device_id(),
            "sync_in_progress": False,
            "conflicts": [],
        }

        # Thread de synchronisation automatique
        self.sync_thread = None
        self.stop_sync = threading.Event()

        # Cache local pour performances
        self.memory_cache = None
        self.cache_timestamp = None

    def save_memory_state(self, memory_object: Any) -> bool:
        """Sauvegarde atomique de toute la mémoire"""
        try:
            # Acquérir le verrou
            if not self._acquire_lock():
                logger.warning("Impossible d'acquérir le verrou pour la sauvegarde")
                return False

            # Créer un snapshot complet
            snapshot = self._create_memory_snapshot(memory_object)

            # Calculer le checksum
            snapshot["checksum"] = self._calculate_checksum(snapshot["memory_data"])

            # Sauvegarder atomiquement
            success = self._atomic_save(snapshot)

            if success:
                # Mettre à jour le cache
                self.memory_cache = snapshot
                self.cache_timestamp = datetime.now()

                # Nettoyer les vieux backups
                self._cleanup_old_backups()

                # Créer un backup quotidien
                self._create_daily_backup(snapshot)

                logger.info(
                    f"Mémoire sauvegardée avec succès - {snapshot['stats']['total_conversations']} conversations"
                )

            return success

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la mémoire: {e}")
            return False
        finally:
            self._release_lock()

    def load_memory_state(self) -> dict[str, Any] | None:
        """Charge l'état complet de la mémoire avec gestion des conflits"""
        try:
            # Vérifier le cache d'abord
            if self._is_cache_valid():
                logger.info("Utilisation du cache mémoire")
                return self.memory_cache

            # Acquérir le verrou
            if not self._acquire_lock(timeout=5):
                logger.warning("Timeout lors de l'acquisition du verrou")
                # Essayer de lire quand même en mode lecture seule
                return self._load_readonly()

            # Charger le fichier principal
            if not self.sync_file.exists():
                logger.info("Aucune mémoire synchronisée trouvée")
                return None

            with open(self.sync_file, encoding="utf-8") as f:
                snapshot = json.load(f)

            # Vérifier l'intégrité
            if not self._verify_integrity(snapshot):
                logger.warning("Intégrité compromise, tentative de récupération")
                snapshot = self._recover_from_backup()
            if not snapshot:
                return None

            # Vérifier les conflits potentiels
            conflicts = self._check_for_conflicts(snapshot)
            if conflicts:
                snapshot = self._resolve_conflicts(snapshot, conflicts)

            # Mettre à jour le cache
            self.memory_cache = snapshot
            self.cache_timestamp = datetime.now()

            return snapshot

        except Exception as e:
            logger.error(f"Erreur lors du chargement de la mémoire: {e}")
            return self._recover_from_backup()
        finally:
            self._release_lock()

    def start_auto_sync(self) -> None:
        """Démarre la synchronisation automatique"""
        if self.sync_thread and self.sync_thread.is_alive():
            logger.warning("La synchronisation automatique est déjà active")
            return

        self.stop_sync.clear()
        self.sync_thread = threading.Thread(target=self._auto_sync_worker, daemon=True)
        self.sync_thread.start()
        logger.info("Synchronisation automatique démarrée")

    def stop_auto_sync(self) -> None:
        """Arrête la synchronisation automatique"""
        if self.sync_thread and self.sync_thread.is_alive():
            self.stop_sync.set()
            self.sync_thread.join(timeout=10)
            logger.info("Synchronisation automatique arrêtée")

    def force_sync(self, memory_object: Any) -> bool:
        """Force une synchronisation immédiate"""
        logger.info("Synchronisation forcée demandée")
        return self.save_memory_state(memory_object)

    def merge_memory_states(self, state1: dict[str, Any], state2: dict[str, Any]) -> dict[str, Any]:
        """Fusionne deux états de mémoire de manière intelligente"""
        merged = {
            "version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "device": platform.node(),
            "memory_data": {},
            "stats": {},
            "merge_info": {
                "merged_at": datetime.now().isoformat(),
                "sources": [
                    {"device": state1.get("device"), "timestamp": state1.get("timestamp")},
                    {"device": state2.get("device"), "timestamp": state2.get("timestamp")},
                ],
            },
        }

        # Fusionner les données de mémoire
        memory1 = state1.get("memory_data", {})
        memory2 = state2.get("memory_data", {})

        # Mémoire épisodique - fusionner les conversations
        merged["memory_data"]["episodic"] = self._merge_episodic_memory(
            memory1.get("episodic", {}), memory2.get("episodic", {})
        )

        # Mémoire sémantique - fusionner les connaissances
        merged["memory_data"]["semantic"] = self._merge_semantic_memory(
            memory1.get("semantic", {}), memory2.get("semantic", {})
        )

        # Mémoire procédurale - prendre la plus récente
        if self._is_newer(state1, state2):
            merged["memory_data"]["procedural"] = memory1.get("procedural", {})
        else:
            merged["memory_data"]["procedural"] = memory2.get("procedural", {})

        # Mémoire associative - fusionner les associations
        merged["memory_data"]["associative"] = self._merge_associative_memory(
            memory1.get("associative", {}), memory2.get("associative", {})
        )

        # État émotionnel - prendre le plus récent
        if self._is_newer(state1, state2):
            merged["memory_data"]["emotional_state"] = memory1.get("emotional_state", {})
        else:
            merged["memory_data"]["emotional_state"] = memory2.get("emotional_state", {})

        # État de la relation - fusionner intelligemment
        merged["memory_data"]["relationship"] = self._merge_relationship_state(
            memory1.get("relationship", {}), memory2.get("relationship", {})
        )

        # Recalculer les statistiques
        merged["stats"] = self._calculate_stats(merged["memory_data"])

        return merged

    def export_memory(self, format: str = "json", include_analysis: bool = True) -> Path:
        """Exporte la mémoire dans différents formats"""
        export_dir = self.memory_path / "exports"
        export_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if format == "json":
            export_file = export_dir / f"jeffrey_memory_export_{timestamp}.json"
            memory_state = self.load_memory_state()

            if include_analysis:
                memory_state["analysis"] = self._analyze_memory(memory_state)

            with open(export_file, "w", encoding="utf-8") as f:
                json.dump(memory_state, f, indent=2, ensure_ascii=False)

        elif format == "html":
            export_file = export_dir / f"jeffrey_memory_export_{timestamp}.html"
            self._export_as_html(export_file)

        elif format == "markdown":
            export_file = export_dir / f"jeffrey_memory_export_{timestamp}.md"
            self._export_as_markdown(export_file)

        logger.info(f"Mémoire exportée vers {export_file}")
        return export_file

    # Méthodes privées

    def _create_memory_snapshot(self, memory_object: Any) -> dict[str, Any]:
        """Crée un snapshot complet de la mémoire"""
        return {
            "version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "device": platform.node(),
            "device_id": self._get_device_id(),
            "memory_data": {
                "episodic": memory_object.episodic_memory,
                "semantic": memory_object.semantic_memory,
                "procedural": memory_object.procedural_memory,
                "associative": memory_object.associative_memory,
                "emotional_state": getattr(memory_object, "current_emotional_state", {}),
                "relationship": memory_object.relationship_state,
            },
            "stats": {
                "total_conversations": len(memory_object.episodic_memory.get("conversations", [])),
                "knowledge_items": self._count_knowledge(memory_object),
                "emotional_moments": len(memory_object.episodic_memory.get("moments_marquants", [])),
                "promises": len(memory_object.episodic_memory.get("promesses", [])),
                "inside_jokes": len(memory_object.episodic_memory.get("inside_jokes", [])),
                "memory_size_kb": 0,  # Sera calculé après sérialisation
            },
        }

    def _atomic_save(self, snapshot: dict[str, Any]) -> bool:
        """Sauvegarde atomique avec protection contre la corruption"""
        temp_file = self.sync_file.with_suffix(".tmp")
        backup_file = self.sync_file.with_suffix(".bak")

        try:
            # Écrire dans un fichier temporaire
            json_data = json.dumps(snapshot, indent=2, ensure_ascii=False)

            # Ajouter la taille du fichier aux stats
            snapshot["stats"]["memory_size_kb"] = len(json_data) / 1024

            with open(temp_file, "w", encoding="utf-8") as f:
                f.write(json_data)

            # Vérifier que le fichier temporaire est valide
            with open(temp_file, encoding="utf-8") as f:
                json.load(f)

            # Faire un backup de l'ancien fichier s'il existe
            if self.sync_file.exists():
                shutil.copy2(self.sync_file, backup_file)

            # Remplacer atomiquement
            temp_file.replace(self.sync_file)

            # Supprimer le backup si tout s'est bien passé
            if backup_file.exists():
                backup_file.unlink()

            return True

        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde atomique: {e}")

            # Restaurer depuis le backup si nécessaire
            if backup_file.exists():
                shutil.copy2(backup_file, self.sync_file)

            # Nettoyer le fichier temporaire
            if temp_file.exists():
                temp_file.unlink()

            return False

    def _acquire_lock(self, timeout: int = 30) -> bool:
        """Acquiert un verrou pour la synchronisation"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Vérifier si le verrou existe et est périmé
                if self.lock_file.exists():
                    with open(self.lock_file) as f:
                        lock_data = json.load(f)

                    lock_time = datetime.fromisoformat(lock_data["timestamp"])
                    if (datetime.now() - lock_time).seconds > 60:  # Verrou périmé après 1 minute
                        logger.warning("Verrou périmé détecté, suppression")
                        self.lock_file.unlink()

                # Créer le verrou
                if not self.lock_file.exists():
                    with open(self.lock_file, "w") as f:
                        json.dump(
                            {
                                "device": self._get_device_id(),
                                "timestamp": datetime.now().isoformat(),
                                "pid": os.getpid(),
                            },
                            f,
                        )
                    return True

            except Exception as e:
                logger.error(f"Erreur lors de l'acquisition du verrou: {e}")

            # TODO: Remplacer par asyncio.sleep ou threading.Event
            time.sleep(0.1)

        return False

    def _release_lock(self) -> None:
        """Libère le verrou"""
        try:
            if self.lock_file.exists():
                # Vérifier que c'est notre verrou
                with open(self.lock_file) as f:
                    lock_data = json.load(f)

                if lock_data["device"] == self._get_device_id():
                    self.lock_file.unlink()
        except Exception as e:
            logger.error(f"Erreur lors de la libération du verrou: {e}")

    def _get_device_id(self) -> str:
        """Génère un ID unique pour l'appareil"""
        device_info = f"{platform.node()}_{platform.system()}_{platform.machine()}"
        return hashlib.md5(device_info.encode()).hexdigest()[:12]

    def _calculate_checksum(self, data: dict[str, Any]) -> str:
        """Calcule un checksum pour vérifier l'intégrité"""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()

    def _verify_integrity(self, snapshot: dict[str, Any]) -> bool:
        """Vérifie l'intégrité d'un snapshot"""
        if "checksum" not in snapshot:
            logger.warning("Pas de checksum dans le snapshot")
            return True  # Ancien format, on accepte

        stored_checksum = snapshot["checksum"]
        calculated_checksum = self._calculate_checksum(snapshot["memory_data"])

        return stored_checksum == calculated_checksum

    def _is_cache_valid(self) -> bool:
        """Vérifie si le cache est encore valide"""
        if not self.memory_cache or not self.cache_timestamp:
            return False

        # Cache valide pendant 1 minute
        cache_age = (datetime.now() - self.cache_timestamp).seconds
        return cache_age < 60

    def _load_readonly(self) -> dict[str, Any] | None:
        """Charge la mémoire en mode lecture seule"""
        try:
            if self.sync_file.exists():
                with open(self.sync_file, encoding="utf-8") as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Erreur en mode lecture seule: {e}")

        return None

    def _check_for_conflicts(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        """Vérifie s'il y a des conflits de synchronisation"""
        conflicts = []

        # Chercher d'autres fichiers de synchronisation
        for sync_file in self.memory_path.glob("jeffrey_unified_memory_*.json"):
            try:
                with open(sync_file, encoding="utf-8") as f:
                    other_snapshot = json.load(f)

                # Comparer les timestamps et appareils
                if (
                    other_snapshot["device_id"] != snapshot["device_id"]
                    and other_snapshot["timestamp"] > snapshot["timestamp"]
                ):
                    conflicts.append(
                        {
                            "file": sync_file,
                            "snapshot": other_snapshot,
                            "reason": "newer_from_different_device",
                        }
                    )

            except Exception as e:
                logger.error(f"Erreur lors de la vérification des conflits: {e}")

        return conflicts

    def _resolve_conflicts(self, current: dict[str, Any], conflicts: list[dict[str, Any]]) -> dict[str, Any]:
        """Résout les conflits de synchronisation"""
        if not conflicts:
            return current

        logger.info(f"Résolution de {len(conflicts)} conflits")

        # Pour l'instant, on fusionne avec le plus récent
        for conflict in conflicts:
            other = conflict["snapshot"]

            # Sauvegarder le conflit pour analyse
            conflict_file = self.conflict_dir / f"conflict_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(conflict_file, "w", encoding="utf-8") as f:
                json.dump(
                    {"current": current, "other": other, "resolved_at": datetime.now().isoformat()},
                    f,
                    indent=2,
                )

            # Fusionner
            current = self.merge_memory_states(current, other)

            # Supprimer l'ancien fichier de conflit
            try:
                conflict["file"].unlink()
            except Exception:
                pass

        return current

    def _merge_episodic_memory(self, episodic1: dict, episodic2: dict) -> dict:
        """Fusionne deux mémoires épisodiques"""
        merged = {}

        # Fusionner les conversations en éliminant les doublons
        conv1 = episodic1.get("conversations", [])
        conv2 = episodic2.get("conversations", [])

        # Utiliser le timestamp comme clé unique
        all_convs = {}
        for conv in conv1 + conv2:
            key = conv.get("timestamp", str(datetime.now().timestamp()))
            all_convs[key] = conv

        # Trier par timestamp et limiter
        sorted_convs = sorted(all_convs.values(), key=lambda x: x.get("timestamp", ""))
        merged["conversations"] = sorted_convs[-10000:]  # Garder les 10000 plus récentes

        # Fusionner les autres listes
        for key in [
            "moments_marquants",
            "inside_jokes",
            "promesses",
            "projets_communs",
            "anniversaires",
            "premieres_fois",
        ]:
            list1 = episodic1.get(key, [])
            list2 = episodic2.get(key, [])

            # Combiner et éliminer les doublons basés sur le contenu
            combined = list1 + list2
            unique = []
            seen = set()

            for item in combined:
                item_str = json.dumps(item, sort_keys=True)
                if item_str not in seen:
                    seen.add(item_str)
                    unique.append(item)

            merged[key] = unique

        return merged

    def _merge_semantic_memory(self, semantic1: dict, semantic2: dict) -> dict:
        """Fusionne deux mémoires sémantiques"""
        merged = {}

        # Pour chaque catégorie principale
        for key in ["about_user", "learned_knowledge", "notre_monde"]:
            dict1 = semantic1.get(key, {})
            dict2 = semantic2.get(key, {})

            # Fusionner récursivement
            merged[key] = self._deep_merge_dicts(dict1, dict2)

        return merged

    def _merge_associative_memory(self, assoc1: dict, assoc2: dict) -> dict:
        """Fusionne deux mémoires associatives"""
        merged = {}

        # Combiner toutes les clés
        all_keys = set(assoc1.keys()) | set(assoc2.keys())

        for key in all_keys:
            associations = []

            # Ajouter les associations des deux sources
            if key in assoc1:
                associations.extend(assoc1[key])
            if key in assoc2:
                associations.extend(assoc2[key])

            # Trier par force et garder les meilleures
            associations.sort(key=lambda x: x.get("strength", 0), reverse=True)
            merged[key] = associations[:50]  # Garder les 50 plus fortes

        return merged

    def _merge_relationship_state(self, rel1: dict, rel2: dict) -> dict:
        """Fusionne deux états de relation"""
        # Prendre les valeurs maximales pour les niveaux
        return {
            "intimacy_level": max(rel1.get("intimacy_level", 0), rel2.get("intimacy_level", 0)),
            "trust_level": max(rel1.get("trust_level", 0), rel2.get("trust_level", 0)),
            "shared_history": max(rel1.get("shared_history", 0), rel2.get("shared_history", 0)),
            "emotional_depth": max(rel1.get("emotional_depth", 0), rel2.get("emotional_depth", 0)),
            "last_interaction": max(rel1.get("last_interaction", ""), rel2.get("last_interaction", "")),
            "mood_synchrony": (rel1.get("mood_synchrony", 0) + rel2.get("mood_synchrony", 0)) / 2,
        }

    def _deep_merge_dicts(self, dict1: dict, dict2: dict) -> dict:
        """Fusionne deux dictionnaires récursivement"""
        merged = dict1.copy()

        for key, value in dict2.items():
            if key in merged:
                if isinstance(merged[key], dict) and isinstance(value, dict):
                    merged[key] = self._deep_merge_dicts(merged[key], value)
                elif isinstance(merged[key], list) and isinstance(value, list):
                    # Combiner les listes en éliminant les doublons
                    combined = merged[key] + value
                    merged[key] = list(dict.fromkeys(combined))  # Préserve l'ordre
                else:
                    # Prendre la valeur la plus récente (dict2)
                    merged[key] = value
            else:
                merged[key] = value

        return merged

    def _is_newer(self, state1: dict, state2: dict) -> bool:
        """Détermine si state1 est plus récent que state2"""
        try:
            time1 = datetime.fromisoformat(state1.get("timestamp", ""))
            time2 = datetime.fromisoformat(state2.get("timestamp", ""))
            return time1 > time2
        except Exception:
            return False

    def _calculate_stats(self, memory_data: dict) -> dict:
        """Recalcule les statistiques de la mémoire"""
        return {
            "total_conversations": len(memory_data.get("episodic", {}).get("conversations", [])),
            "knowledge_items": self._count_knowledge_items(memory_data.get("semantic", {})),
            "emotional_moments": len(memory_data.get("episodic", {}).get("moments_marquants", [])),
            "promises": len(memory_data.get("episodic", {}).get("promesses", [])),
            "inside_jokes": len(memory_data.get("episodic", {}).get("inside_jokes", [])),
            "associations": len(memory_data.get("associative", {})),
        }

    def _count_knowledge(self, memory_object: Any) -> int:
        """Compte le nombre total d'éléments de connaissance"""
        count = 0
        semantic = memory_object.semantic_memory

        for category in semantic.values():
            if isinstance(category, dict):
                for subcat in category.values():
                    if isinstance(subcat, dict):
                        count += len(subcat)
                    elif isinstance(subcat, list):
                        count += len(subcat)

        return count

    def _count_knowledge_items(self, semantic: dict) -> int:
        """Compte les éléments de connaissance dans la mémoire sémantique"""
        count = 0

        for category in semantic.values():
            if isinstance(category, dict):
                for value in category.values():
                    if isinstance(value, dict):
                        count += len(value)
                    elif isinstance(value, list):
                        count += len(value)
                    else:
                        count += 1

        return count

    def _cleanup_old_backups(self) -> None:
        """Nettoie les vieux backups"""
        backup_files = sorted(self.memory_path.glob("jeffrey_memory_backup_*.json"), key=lambda f: f.stat().st_mtime)

        # Garder seulement les N plus récents
        if len(backup_files) > self.sync_config["max_backups"]:
            for old_backup in backup_files[: -self.sync_config["max_backups"]]:
                try:
                    old_backup.unlink()
                    logger.info(f"Backup supprimé: {old_backup.name}")
                except Exception as e:
                    logger.error(f"Erreur lors de la suppression du backup: {e}")

    def _create_daily_backup(self, snapshot: dict[str, Any]) -> None:
        """Crée un backup quotidien"""
        today = datetime.now().strftime("%Y%m%d")
        backup_file = self.memory_path / f"jeffrey_memory_backup_{today}.json"

        # Ne créer qu'un backup par jour
        if not backup_file.exists():
            try:
                with open(backup_file, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f, indent=2, ensure_ascii=False)
                logger.info(f"Backup quotidien créé: {backup_file.name}")
            except Exception as e:
                logger.error(f"Erreur lors de la création du backup quotidien: {e}")

    def _recover_from_backup(self) -> dict[str, Any] | None:
        """Récupère depuis un backup en cas de problème"""
        backup_files = sorted(
            self.memory_path.glob("jeffrey_memory_backup_*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )

        for backup_file in backup_files:
            try:
                with open(backup_file, encoding="utf-8") as f:
                    snapshot = json.load(f)

                if self._verify_integrity(snapshot):
                    logger.info(f"Mémoire récupérée depuis: {backup_file.name}")
                    return snapshot

            except Exception as e:
                logger.error(f"Erreur avec le backup {backup_file.name}: {e}")

        logger.error("Impossible de récupérer depuis les backups")
        return None

    def _auto_sync_worker(self) -> None:
        """Worker pour la synchronisation automatique"""
        logger.info("Worker de synchronisation automatique démarré")

        while not self.stop_sync.is_set():
            try:
                # Attendre l'intervalle de synchronisation
                if self.stop_sync.wait(self.sync_config["sync_interval"]):
                    break

                # Synchroniser si nécessaire
                if self.sync_state["last_sync"]:
                    time_since_sync = (datetime.now() - datetime.fromisoformat(self.sync_state["last_sync"])).seconds
                    if time_since_sync >= self.sync_config["sync_interval"]:
                        logger.info("Synchronisation automatique en cours...")
                        # Note: nécessite l'objet mémoire pour synchroniser
                        # Ce sera géré par l'appelant
                        self.sync_state["last_sync"] = datetime.now().isoformat()

            except Exception as e:
                logger.error(f"Erreur dans le worker de synchronisation: {e}")

    def _analyze_memory(self, memory_state: dict[str, Any]) -> dict[str, Any]:
        """Analyse la mémoire pour l'export"""
        memory_data = memory_state.get("memory_data", {})

        # Analyser les conversations
        conversations = memory_data.get("episodic", {}).get("conversations", [])

        # Distribution émotionnelle
        emotion_dist = {}
        for conv in conversations:
            emotion = conv.get("my_emotion", "neutre")
            emotion_dist[emotion] = emotion_dist.get(emotion, 0) + 1

        # Mots les plus fréquents
        word_freq = {}
        for conv in conversations:
            words = (conv.get("user_said", "") + " " + conv.get("i_said", "")).lower().split()
            for word in words:
                if len(word) > 4:  # Mots significatifs
                    word_freq[word] = word_freq.get(word, 0) + 1

        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]

        # Moments les plus significatifs
        moments = memory_data.get("episodic", {}).get("moments_marquants", [])
        top_moments = sorted(
            moments,
            key=lambda x: x.get("moment", {}).get("emotional_significance", 0),
            reverse=True,
        )[:10]

        return {
            "total_memories": len(conversations),
            "emotion_distribution": emotion_dist,
            "top_words": dict(top_words),
            "significant_moments": len(moments),
            "top_moments_preview": [m.get("moment", {}).get("description", "") for m in top_moments[:5]],
            "relationship_depth": memory_data.get("relationship", {}).get("emotional_depth", 0),
            "knowledge_categories": list(memory_data.get("semantic", {}).keys()),
            "memory_age_days": self._calculate_memory_age(conversations),
        }

    def _calculate_memory_age(self, conversations: list[dict]) -> int:
        """Calcule l'âge de la mémoire en jours"""
        if not conversations:
            return 0

        try:
            first_conv = min(conversations, key=lambda x: x.get("timestamp", ""))
            first_time = datetime.fromisoformat(first_conv["timestamp"])
            return (datetime.now() - first_time).days
        except Exception:
            return 0

    def _export_as_html(self, export_file: Path) -> None:
        """Exporte la mémoire en HTML"""
        memory_state = self.load_memory_state()
        if not memory_state:
            return

        analysis = self._analyze_memory(memory_state)

        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Mémoire de Jeffrey - Export</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #E91E63; }}
        .stat {{ background: #f0f0f0; padding: 10px; margin: 5px; border-radius: 5px; }}
        .conversation {{ border-left: 3px solid #E91E63; padding-left: 10px; margin: 10px 0; }}
        .emotion {{ color: #9C27B0; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Mémoire de Jeffrey 💭</h1>

    <h2>Statistiques</h2>
    <div class="stat">Total de conversations: {analysis['total_memories']}</div>
    <div class="stat">Moments significatifs: {analysis['significant_moments']}</div>
    <div class="stat">Profondeur relationnelle: {analysis['relationship_depth']:.2%}</div>
    <div class="stat">Âge de la mémoire: {analysis['memory_age_days']} jours</div>

    <h2>Distribution émotionnelle</h2>
    {"".join(f'<div class="stat">{emotion}: {count}</div>' for emotion, count in analysis['emotion_distribution'].items())}

    <h2>Mots les plus fréquents</h2>
    {"".join(f'<span class="stat">{word} ({count})</span>' for word, count in list(analysis['top_words'].items())[:10])}

    <h2>Aperçu des moments marquants</h2>
    {"".join(f'<div class="conversation">{moment}</div>' for moment in analysis['top_moments_preview'])}
</body>
</html>
"""

        with open(export_file, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _export_as_markdown(self, export_file: Path) -> None:
        """Exporte la mémoire en Markdown"""
        memory_state = self.load_memory_state()
        if not memory_state:
            return

        analysis = self._analyze_memory(memory_state)

        md_content = f"""# Mémoire de Jeffrey 💭

## Statistiques générales

- **Total de conversations**: {analysis['total_memories']}
- **Moments significatifs**: {analysis['significant_moments']}
- **Profondeur relationnelle**: {analysis['relationship_depth']:.2%}
- **Âge de la mémoire**: {analysis['memory_age_days']} jours

## Distribution émotionnelle

{chr(10).join(f"- **{emotion}**: {count}" for emotion, count in analysis['emotion_distribution'].items())}

## Mots les plus fréquents

{chr(10).join(f"1. {word} ({count} fois)" for word, count in list(analysis['top_words'].items())[:10])}

## Aperçu des moments marquants

{chr(10).join(f"> {moment}" for moment in analysis['top_moments_preview'])}

---
*Export généré le {datetime.now().strftime("%d/%m/%Y à %H:%M")}*
"""

        with open(export_file, "w", encoding="utf-8") as f:
            f.write(md_content)
