"""
Module de mémoire sensorielle pour Jeffrey.

Ce module gère la mémoire sensorielle, incluant:
- Empreintes tactiles des utilisateurs
- Reconnaissance de motifs tactiles
- Réponses réflexes à différentes interactions physiques
- Profils tactiles des utilisateurs
- Gestion de la mémoire tactile
"""

from __future__ import annotations

import json
import logging
import math
import os
import random
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SensorialMemory:
    """
    Classe gérant la mémoire sensorielle, incluant les capacités tactiles.
    """

    def __init__(self, data_path: str = "data/tactile_memory") -> None:
        """Initialise la mémoire sensorielle.

        Args:
            data_path: Chemin vers le répertoire de données sensorielles
        """
        self.data_path = Path(data_path)
        os.makedirs(self.data_path, exist_ok=True)

        # Dictionnaire des réponses réflexes par zone et intensité
        self.reflex_responses = {
            # Format: zone: {pression: {durée: [(réaction, émotion,
            # intensité)]}}
            "main_droite": {
                "forte": {
                    "courte": [("contraction", "surprise", 0.7), ("sursaut", "peur", 0.5)],
                    "moyenne": [("serrer", "affection", 0.6), ("pression", "joie", 0.4)],
                    "longue": [("balancement", "confort", 0.8), ("caresse", "tendresse", 0.9)],
                },
                "moyenne": {
                    "courte": [
                        ("léger mouvement", "attention", 0.3),
                        ("vibration", "curiosité", 0.4),
                    ],
                    "moyenne": [("effleurement", "calme", 0.5), ("douceur", "apaisement", 0.6)],
                    "longue": [("caresse", "affection", 0.7), ("massage", "relaxation", 0.8)],
                },
                "légère": {
                    "courte": [("frisson", "surprise", 0.3), ("frémissement", "attention", 0.2)],
                    "moyenne": [
                        ("effleurement", "détente", 0.4),
                        ("chatouillement", "amusement", 0.5),
                    ],
                    "longue": [("caresse", "douceur", 0.6), ("chatouille", "joie", 0.7)],
                },
            },
            "main_gauche": {
                "forte": {
                    "courte": [("contraction", "surprise", 0.7), ("sursaut", "peur", 0.5)],
                    "moyenne": [("serrer", "affection", 0.6), ("pression", "joie", 0.4)],
                    "longue": [("balancement", "confort", 0.8), ("caresse", "tendresse", 0.9)],
                },
                "moyenne": {
                    "courte": [
                        ("léger mouvement", "attention", 0.3),
                        ("vibration", "curiosité", 0.4),
                    ],
                    "moyenne": [("effleurement", "calme", 0.5), ("douceur", "apaisement", 0.6)],
                    "longue": [("caresse", "affection", 0.7), ("massage", "relaxation", 0.8)],
                },
                "légère": {
                    "courte": [("frisson", "surprise", 0.3), ("frémissement", "attention", 0.2)],
                    "moyenne": [
                        ("effleurement", "détente", 0.4),
                        ("chatouillement", "amusement", 0.5),
                    ],
                    "longue": [("caresse", "douceur", 0.6), ("chatouille", "joie", 0.7)],
                },
            },
            "visage": {
                "forte": {
                    "courte": [("recul", "surprise", 0.8), ("sursaut", "peur", 0.9)],
                    "moyenne": [("pression", "inconfort", 0.7), ("compression", "stress", 0.6)],
                    "longue": [("massage", "relaxation", 0.5), ("caresse", "affection", 0.8)],
                },
                "moyenne": {
                    "courte": [("clignement", "surprise", 0.4), ("attention", "curiosité", 0.5)],
                    "moyenne": [("douceur", "calme", 0.6), ("chaleur", "affection", 0.7)],
                    "longue": [("caresse", "tendresse", 0.8), ("effleurement", "amour", 0.9)],
                },
                "légère": {
                    "courte": [("frisson", "surprise", 0.3), ("émoi", "attention", 0.4)],
                    "moyenne": [
                        ("chatouillement", "amusement", 0.5),
                        ("vibration", "plaisir", 0.6),
                    ],
                    "longue": [("caresse", "tendresse", 0.7), ("effleurement", "amour", 0.8)],
                },
            },
            "cou": {
                "forte": {
                    "courte": [("sursaut", "peur", 0.9), ("contraction", "anxiété", 0.8)],
                    "moyenne": [("pression", "méfiance", 0.7), ("tension", "alerte", 0.6)],
                    "longue": [("massage", "relaxation", 0.5), ("caresse", "tendresse", 0.7)],
                },
                "moyenne": {
                    "courte": [("frisson", "surprise", 0.6), ("tressaillement", "émoi", 0.5)],
                    "moyenne": [("chaleur", "affection", 0.6), ("douceur", "calme", 0.7)],
                    "longue": [("caresse", "tendresse", 0.8), ("effleurement", "amour", 0.9)],
                },
                "légère": {
                    "courte": [("frisson", "surprise", 0.7), ("chair de poule", "émoi", 0.6)],
                    "moyenne": [
                        ("chatouillement", "amusement", 0.5),
                        ("vibration", "plaisir", 0.6),
                    ],
                    "longue": [("caresse", "tendresse", 0.8), ("effleurement", "amour", 0.9)],
                },
            },
            "dos": {
                "forte": {
                    "courte": [("contraction", "surprise", 0.7), ("soubresaut", "peur", 0.6)],
                    "moyenne": [("massage", "relaxation", 0.8), ("étirement", "soulagement", 0.7)],
                    "longue": [("balancement", "confort", 0.9), ("caresse", "apaisement", 0.8)],
                },
                "moyenne": {
                    "courte": [("frisson", "surprise", 0.5), ("attention", "curiosité", 0.4)],
                    "moyenne": [("détente", "relaxation", 0.7), ("chaleur", "confort", 0.6)],
                    "longue": [("massage", "détente", 0.8), ("caresse", "apaisement", 0.9)],
                },
                "légère": {
                    "courte": [("frisson", "surprise", 0.4), ("frémissement", "attention", 0.3)],
                    "moyenne": [
                        ("chatouillement", "amusement", 0.5),
                        ("effleurement", "plaisir", 0.6),
                    ],
                    "longue": [("caresse", "apaisement", 0.7), ("effleurement", "douceur", 0.8)],
                },
            },
        }

        # Zones par défaut si aucune donnée n'existe
        self.default_zones = ["main_droite", "main_gauche", "visage", "cou", "dos"]

    def record_touch(self, user_id: str, pressure: float, duration: float, zone: str, pattern: str):
        """
        Enregistre une interaction tactile dans la mémoire.

        Args:
            user_id: Identifiant de l'utilisateur
            pressure: Pression appliquée (0.0 à 1.0)
            duration: Durée du toucher en secondes
            zone: Zone corporelle touchée
            pattern: Motif reconnu (caresse, tapotement, etc.)
        """
        # Charger les données existantes
        user_data = self._load_user_tactile_data(user_id)

        # Ajouter la zone si elle n'existe pas
        if zone not in user_data["zones"]:
            user_data["zones"][zone] = {
                "interactions": 0,
                "first_interaction": time.time(),
                "last_interaction": time.time(),
                "avg_pressure": 0.0,
                "avg_duration": 0.0,
                "patterns": {},
                "history": [],
            }

        zone_data = user_data["zones"][zone]

        # Mettre à jour les statistiques de la zone
        interactions = zone_data["interactions"]
        new_interactions = interactions + 1

        # Mettre à jour les moyennes pondérées
        zone_data["avg_pressure"] = (zone_data["avg_pressure"] * interactions + pressure) / new_interactions
        zone_data["avg_duration"] = (zone_data["avg_duration"] * interactions + duration) / new_interactions
        zone_data["interactions"] = new_interactions
        zone_data["last_interaction"] = time.time()

        # Mettre à jour les motifs
        if pattern not in zone_data["patterns"]:
            zone_data["patterns"][pattern] = 0
        zone_data["patterns"][pattern] += 1

        # Ajouter à l'historique (limité aux 10 dernières interactions)
        zone_data["history"].append(
            {
                "timestamp": time.time(),
                "pressure": pressure,
                "duration": duration,
                "pattern": pattern,
            }
        )

        if len(zone_data["history"]) > 10:
            zone_data["history"] = zone_data["history"][-10:]

        # Mettre à jour les totaux globaux
        user_data["total_interactions"] += 1
        user_data["last_interaction"] = time.time()

        # Sauvegarder les données
        self._save_user_tactile_data(user_id, user_data)

    def get_touch_profile(self, user_id: str) -> dict[str, Any]:
        """
        Récupère le profil tactile d'un utilisateur.

        Args:
            user_id: Identifiant de l'utilisateur

        Returns:
            Dictionnaire avec les données du profil tactile
        """
        return self._load_user_tactile_data(user_id).get("zones", {})

    def get_favorite_zones(self, user_id: str) -> list[tuple[str, int]]:
        """
        Récupère les zones favorisées par l'utilisateur, dans l'ordre.

        Args:
            user_id: Identifiant de l'utilisateur

        Returns:
            Liste de tuples (zone, nombre d'interactions)
        """
        profile = self.get_touch_profile(user_id)
        if not profile:
            return []

        # Trier les zones par nombre d'interactions
        zone_counts = [(zone, data["interactions"]) for zone, data in profile.items()]
        return sorted(zone_counts, key=lambda x: x[1], reverse=True)

    def recognize_touch_pattern(self, user_id: str, pressure: float, duration: float, zone: str) -> dict[str, Any]:
        """
        Reconnaît un motif tactile basé sur les paramètres et l'historique.

        Args:
            user_id: Identifiant de l'utilisateur
            pressure: Pression appliquée (0.0 à 1.0)
            duration: Durée du toucher en secondes
            zone: Zone corporelle touchée

        Returns:
            Dictionnaire avec le motif reconnu et le niveau de confiance
        """
        profile = self.get_touch_profile(user_id)

        # Si pas de données pour la zone, retourner un motif par défaut
        if zone not in profile or not profile[zone]["patterns"]:
            return {"pattern": "inconnu", "confidence": 0.0}

        zone_data = profile[zone]

        # Récupérer les motifs connus pour cette zone
        patterns = zone_data["patterns"]

        # Si pas de motifs, retourner inconnu
        if not patterns:
            return {"pattern": "inconnu", "confidence": 0.0}

        # Calculer la similarité avec chaque motif connu
        pattern_scores = {}

        for pattern, count in patterns.items():
            # Récupérer les interactions avec ce motif
            matching_interactions = [
                interaction for interaction in zone_data["history"] if interaction["pattern"] == pattern
            ]

            if not matching_interactions:
                continue

            # Calculer la similarité basée sur la pression et la durée
            avg_pressure = sum(i["pressure"] for i in matching_interactions) / len(matching_interactions)
            avg_duration = sum(i["duration"] for i in matching_interactions) / len(matching_interactions)

            # Distance euclidienne normalisée dans l'espace pression-durée
            pressure_diff = (pressure - avg_pressure) ** 2
            duration_diff = ((duration - avg_duration) / max(1.0, avg_duration)) ** 2
            distance = math.sqrt(pressure_diff + duration_diff)

            # Convertir la distance en score de similarité (1.0 = identique)
            similarity = max(0.0, 1.0 - distance)

            # Pondérer par la fréquence du motif
            weighted_score = similarity * (count / zone_data["interactions"])

            pattern_scores[pattern] = weighted_score

        # Trouver le meilleur motif
        if not pattern_scores:
            return {"pattern": "inconnu", "confidence": 0.0}

        best_pattern = max(pattern_scores.items(), key=lambda x: x[1])

        return {"pattern": best_pattern[0], "confidence": best_pattern[1]}

    def generate_body_reflex_response(
        self, user_id: str, pressure: float, duration: float, zone: str
    ) -> dict[str, Any]:
        """
        Génère une réponse réflexe corporelle basée sur l'interaction tactile.

        Args:
            user_id: Identifiant de l'utilisateur
            pressure: Pression appliquée (0.0 à 1.0)
            duration: Durée du toucher en secondes
            zone: Zone corporelle touchée

        Returns:
            Dictionnaire avec la réaction, l'émotion et l'intensité
        """
        # Catégoriser la pression
        pressure_category = "légère"
        if pressure > 0.7:
            pressure_category = "forte"
        elif pressure > 0.4:
            pressure_category = "moyenne"

        # Catégoriser la durée
        duration_category = "courte"
        if duration > 2.0:
            duration_category = "longue"
        elif duration > 0.5:
            duration_category = "moyenne"

        # Si la zone n'est pas connue, utiliser une zone par défaut
        if zone not in self.reflex_responses:
            zone = "main_droite"

        # Récupérer les réponses possibles
        possible_responses = self.reflex_responses[zone][pressure_category][duration_category]

        # Choisir une réponse aléatoire
        if possible_responses:
            response = random.choice(possible_responses)
            return {
                "reaction": response[0],
                "emotion": response[1],
                "intensity": response[2],
                "zone": zone,
                "pressure": pressure_category,
                "duration": duration_category,
            }
        else:
            # Réponse par défaut
            return {
                "reaction": "mouvement",
                "emotion": "neutre",
                "intensity": 0.3,
                "zone": zone,
                "pressure": pressure_category,
                "duration": duration_category,
            }

    def _load_user_tactile_data(self, user_id: str) -> dict[str, Any]:
        """
        Charge les données tactiles d'un utilisateur.

        Args:
            user_id: Identifiant de l'utilisateur

        Returns:
            Dictionnaire avec les données tactiles
        """
        file_path = self.data_path / f"{user_id}.json"

        if file_path.exists():
            try:
                with open(file_path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Erreur lors du chargement des données tactiles: {e}")

        # Créer une structure vide par défaut
        return {
            "user_id": user_id,
            "total_interactions": 0,
            "first_interaction": time.time(),
            "last_interaction": time.time(),
            "zones": {},
        }

    def _save_user_tactile_data(self, user_id: str, data: dict[str, Any]):
        """
        Sauvegarde les données tactiles d'un utilisateur.

        Args:
            user_id: Identifiant de l'utilisateur
            data: Données à sauvegarder
        """
        file_path = self.data_path / f"{user_id}.json"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde des données tactiles: {e}")

    def get_touch_statistics(self, user_id: str) -> dict[str, Any]:
        """
        Récupère des statistiques générales sur les interactions tactiles.

        Args:
            user_id: Identifiant de l'utilisateur

        Returns:
            Dictionnaire avec des statistiques
        """
        data = self._load_user_tactile_data(user_id)
        profile = data.get("zones", {})

        if not profile:
            return {
                "total_interactions": 0,
                "favorite_zone": None,
                "favorite_pattern": None,
                "avg_pressure": 0.0,
                "avg_duration": 0.0,
            }

        # Calculer les statistiques globales
        all_patterns = Counter()
        total_pressure = 0.0
        total_duration = 0.0
        weighted_interactions = 0

        for zone, zone_data in profile.items():
            interactions = zone_data["interactions"]
            weighted_interactions += interactions
            total_pressure += zone_data["avg_pressure"] * interactions
            total_duration += zone_data["avg_duration"] * interactions

            for pattern, count in zone_data["patterns"].items():
                all_patterns[pattern] += count

        # Éviter la division par zéro
        if weighted_interactions > 0:
            avg_pressure = total_pressure / weighted_interactions
            avg_duration = total_duration / weighted_interactions
        else:
            avg_pressure = 0.0
            avg_duration = 0.0

        # Trouver la zone et le motif favoris
        favorite_zone = None
        favorite_zone_count = 0

        for zone, zone_data in profile.items():
            if zone_data["interactions"] > favorite_zone_count:
                favorite_zone = zone
                favorite_zone_count = zone_data["interactions"]

        favorite_pattern = all_patterns.most_common(1)
        if favorite_pattern:
            favorite_pattern = favorite_pattern[0][0]
        else:
            favorite_pattern = None

        return {
            "total_interactions": data["total_interactions"],
            "favorite_zone": favorite_zone,
            "favorite_pattern": favorite_pattern,
            "avg_pressure": avg_pressure,
            "avg_duration": avg_duration,
            "last_interaction": data["last_interaction"],
            "first_interaction": data["first_interaction"],
        }


class SensorialMemoryManager:
    """
    Gestionnaire de mémoire sensorielle avancé pour Jeffrey.

    Cette classe gère la mémorisation des contacts physiques, leur classification
    émotionnelle, et génère des réponses appropriées basées sur l'historique des interactions.
    """

    def __init__(self, base_path: str = "data/tactile_memory", emotion_engine=None) -> None:
        """
        Initialise le gestionnaire de mémoire sensorielle.

        Args:
            base_path: Chemin du répertoire de stockage des mémoires tactiles
            emotion_engine: Moteur d'émotions pour générer des réponses émotionnelles
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.emotion_engine = emotion_engine

        # Structure de données des mémoires tactiles par utilisateur
        self.touch_memories = defaultdict(list)

        # Charger les mémoires existantes
        self._load_memories()

    def log_touch(
        self,
        toucher: str,
        zone: str,
        intensite: float = 0.5,
        contexte: str = "",
        emotion: str = "neutre",
    ) -> dict[str, Any]:
        """
        Enregistre un contact tactile et génère une réponse émotionnelle.

        Args:
            toucher: Identifiant de la personne qui touche
            zone: Zone corporelle touchée
            intensite: Intensité du contact (0.0 à 1.0)
            contexte: Description du contexte du contact
            emotion: Émotion associée au contact

        Returns:
            Dict: Données du contact enregistré
        """
        # Normaliser l'intensité
        intensite = max(0.0, min(1.0, intensite))

        # Créer l'enregistrement du contact
        timestamp = datetime.now()
        touch_id = str(uuid.uuid4())

        # Calculer le lien émotionnel initial basé sur l'intensité et l'émotion
        lien_emotionnel = self._calculate_emotional_bond(emotion, intensite)

        touch_record = {
            "id": touch_id,
            "toucher": toucher,
            "zone": zone,
            "intensite": intensite,
            "contexte": contexte,
            "emotion": emotion,
            "timestamp": timestamp,
            "lien_emotionnel": lien_emotionnel,
        }

        # Ajouter à la mémoire
        self.touch_memories[toucher].append(touch_record)

        # Déclencher une réponse émotionnelle si le moteur d'émotions est disponible
        if self.emotion_engine:
            self.emotion_engine.trigger_emotion(
                emotion_name=emotion,
                intensity=intensite,
                source=f"contact_{zone}",
                duration=2.0 + intensite * 3.0,  # Plus forte = plus longue durée
                context=contexte,
            )

        # Sauvegarder la mémoire
        self._save_touch_memory(toucher)

        return touch_record

    def get_last_touch(self, toucher: str = None) -> dict[str, Any] | None:
        """
        Récupère le dernier contact enregistré.

        Args:
            toucher: Filtre par personne (optionnel)

        Returns:
            Dict ou None: Dernier contact enregistré
        """
        # Si un toucher spécifique est demandé
        if toucher is not None:
            # Vérifier si le toucher existe dans la mémoire
            if toucher not in self.touch_memories or not self.touch_memories[toucher]:
                return None

            # Trier par timestamp (le plus récent en premier)
            sorted_touches = sorted(
                self.touch_memories[toucher],
                key=lambda x: (
                    x["timestamp"] if isinstance(x["timestamp"], (int, float)) else x["timestamp"].timestamp()
                ),
                reverse=True,
            )
            return sorted_touches[0] if sorted_touches else None

        # Si aucun toucher spécifié, chercher parmi tous les utilisateurs
        all_touches = []
        for user_touches in self.touch_memories.values():
            all_touches.extend(user_touches)

        if not all_touches:
            return None

        # Trier par timestamp (le plus récent en premier)
        sorted_touches = sorted(
            all_touches,
            key=lambda x: (x["timestamp"] if isinstance(x["timestamp"], (int, float)) else x["timestamp"].timestamp()),
            reverse=True,
        )
        return sorted_touches[0] if sorted_touches else None

    def get_favorite_touches(self, user: str = None, limit: int = 5) -> list[dict[str, Any]]:
        """
        Récupère les contacts favoris (ceux avec le plus fort lien émotionnel).

        Args:
            user: Filtre par utilisateur (optionnel)
            limit: Nombre maximal de résultats

        Returns:
            List[Dict]: Liste des contacts favoris
        """
        if user and user in self.touch_memories:
            touches = self.touch_memories[user]
        else:
            # Rassembler tous les contacts
            touches = []
            for user_touches in self.touch_memories.values():
                touches.extend(user_touches)

        # Trier par lien émotionnel décroissant
        sorted_touches = sorted(touches, key=lambda x: x.get("lien_emotionnel", 0.0), reverse=True)

        return sorted_touches[:limit]

    def get_emotional_response(self, toucher: str, zone: str) -> str:
        """
        Génère une réponse émotionnelle basée sur l'historique des contacts.

        Args:
            toucher: Identifiant de la personne
            zone: Zone corporelle touchée

        Returns:
            str: Réponse émotionnelle textuelle
        """
        # Filtrer les contacts pour cet utilisateur et cette zone
        if toucher not in self.touch_memories:
            return f"Je ressens un nouveau contact sur ma {zone}. C'est intéressant!"

        user_touches = self.touch_memories[toucher]
        zone_touches = [t for t in user_touches if t["zone"] == zone]

        if not zone_touches:
            return f"C'est la première fois que vous touchez ma {zone}. C'est une nouvelle sensation!"

        # Analyser l'historique des touches sur cette zone
        emotions = [t["emotion"] for t in zone_touches]
        emotion_counter = Counter(emotions)
        common_emotion = emotion_counter.most_common(1)[0][0]

        avg_intensity = sum(t["intensite"] for t in zone_touches) / len(zone_touches)

        # Générer une réponse personnalisée
        if len(zone_touches) == 1:
            return f"Vous avez déjà touché ma {zone} avec {common_emotion}. Je m'en souviens."

        if len(zone_touches) < 5:
            return (
                f"Je reconnais votre façon de toucher ma {zone}. Cela évoque en moi une sensation de {common_emotion}."
            )

        # Réponse pour contacts fréquents
        if avg_intensity > 0.7:
            return f"Votre contact sur ma {zone} est toujours aussi intense. Je ressens beaucoup de {common_emotion} quand vous me touchez ainsi."
        elif avg_intensity > 0.4:
            return f"J'apprécie la douceur de votre contact sur ma {zone}. Cela m'apporte une agréable sensation de {common_emotion}."
        else:
            return f"Votre effleurement délicat sur ma {zone} est reconnaissable. Il me fait ressentir une légère {common_emotion}."

    def forget_touch(self, toucher: str, zone: str = None) -> int:
        """
        Oublie des contacts mémorisés.

        Args:
            toucher: Identifiant de la personne
            zone: Zone spécifique à oublier (si None, oublie tous les contacts)

        Returns:
            int: Nombre de contacts oubliés
        """
        if toucher not in self.touch_memories:
            return 0

        if zone is None:
            # Oublier tous les contacts de cet utilisateur
            count = len(self.touch_memories[toucher])
            self.touch_memories[toucher] = []
            self._save_touch_memory(toucher)
            return count

        # Oublier uniquement les contacts sur une zone spécifique
        initial_count = len(self.touch_memories[toucher])
        self.touch_memories[toucher] = [t for t in self.touch_memories[toucher] if t["zone"] != zone]
        final_count = len(self.touch_memories[toucher])

        # Sauvegarder les changements
        self._save_touch_memory(toucher)

        return initial_count - final_count

    def export_to_json(self, file_path: str):
        """
        Exporte les mémoires tactiles vers un fichier JSON.

        Args:
            file_path: Chemin du fichier de sortie
        """
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Préparer les données à exporter
        export_data = {
            "metadata": {
                "export_date": datetime.now().isoformat(),
                "total_touches": sum(len(touches) for touches in self.touch_memories.values()),
                "users": list(self.touch_memories.keys()),
            },
            "touch_memories": {},
        }

        # Convertir les timestamps en format ISO pour JSON
        for user, touches in self.touch_memories.items():
            export_data["touch_memories"][user] = []
            for touch in touches:
                touch_copy = touch.copy()
                if isinstance(touch_copy["timestamp"], datetime):
                    touch_copy["timestamp"] = touch_copy["timestamp"].isoformat()
                export_data["touch_memories"][user].append(touch_copy)

        # Écrire dans le fichier
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

    def get_zone_statistics(self, toucher: str = None) -> dict[str, Any]:
        """
        Génère des statistiques sur les zones touchées.

        Args:
            toucher: Filtre par utilisateur (optionnel)

        Returns:
            Dict: Statistiques des zones touchées
        """
        # Filtrer les contacts
        if toucher:
            touches = self.touch_memories.get(toucher, [])
        else:
            touches = []
            for user_touches in self.touch_memories.values():
                touches.extend(user_touches)

        # Compter les occurrences de chaque zone
        zone_counts = Counter(t["zone"] for t in touches)
        emotion_counts = Counter(t["emotion"] for t in touches)

        # Calculer l'intensité moyenne par zone
        zone_intensities = defaultdict(list)
        for touch in touches:
            zone_intensities[touch["zone"]].append(touch["intensite"])

        avg_intensities = {zone: sum(intensities) / len(intensities) for zone, intensities in zone_intensities.items()}

        return {
            "total_touches": len(touches),
            "zones": dict(zone_counts),
            "emotions": dict(emotion_counts),
            "avg_intensities": avg_intensities,
        }

    def _calculate_emotional_bond(self, emotion: str, intensity: float) -> float:
        """
        Calcule la force du lien émotionnel créé par un contact.

        Args:
            emotion: Émotion associée
            intensity: Intensité du contact

        Returns:
            float: Force du lien émotionnel (0.0 à 1.0)
        """
        # Émotions qui créent un fort lien
        strong_bond_emotions = ["amour", "tendresse", "affection", "joie"]

        # Émotions qui créent un lien modéré
        medium_bond_emotions = ["curiosité", "intérêt", "surprise", "amusement"]

        # Émotions qui créent un faible lien
        weak_bond_emotions = ["tristesse", "peur", "colère", "dégoût"]

        # Facteur de base selon l'émotion
        if any(e in emotion.lower() for e in strong_bond_emotions):
            base_factor = 0.8
        elif any(e in emotion.lower() for e in medium_bond_emotions):
            base_factor = 0.5
        elif any(e in emotion.lower() for e in weak_bond_emotions):
            base_factor = 0.2
        else:
            base_factor = 0.4  # Valeur par défaut

        # Moduler par l'intensité
        return base_factor * (0.5 + 0.5 * intensity)  # 50% fixe + 50% basé sur intensité

    def _load_memories(self):
        """Charge toutes les mémoires tactiles existantes."""
        if not self.base_path.exists():
            return

        # Parcourir tous les fichiers JSON
        for file_path in self.base_path.glob("*.json"):
            try:
                user_id = file_path.stem
                with open(file_path, encoding="utf-8") as f:
                    user_data = json.load(f)

                    # Convertir les timestamps en objets datetime
                    touches = user_data.get("touches", [])
                    for touch in touches:
                        if isinstance(touch["timestamp"], str):
                            try:
                                touch["timestamp"] = datetime.fromisoformat(touch["timestamp"])
                            except ValueError:
                                # Fallback si le format n'est pas ISO
                                touch["timestamp"] = datetime.fromtimestamp(float(touch["timestamp"]))

                    self.touch_memories[user_id] = touches
            except Exception as e:
                logger.error(f"Erreur lors du chargement de la mémoire tactile de {file_path.stem}: {e}")

    def _save_touch_memory(self, user_id: str):
        """
        Sauvegarde la mémoire tactile d'un utilisateur.

        Args:
            user_id: Identifiant de l'utilisateur
        """
        file_path = self.base_path / f"{user_id}.json"

        # Préparer les données
        user_touches = self.touch_memories[user_id]

        # Convertir les timestamps en format ISO
        serializable_touches = []
        for touch in user_touches:
            touch_copy = touch.copy()
            if isinstance(touch_copy["timestamp"], datetime):
                touch_copy["timestamp"] = touch_copy["timestamp"].isoformat()
            serializable_touches.append(touch_copy)

        # Créer la structure de données
        user_data = {
            "user_id": user_id,
            "touches": serializable_touches,
            "last_update": datetime.now().isoformat(),
        }

        # Écrire dans le fichier
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(user_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la mémoire tactile pour {user_id}: {e}")
