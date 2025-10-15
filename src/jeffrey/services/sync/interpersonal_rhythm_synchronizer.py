#!/usr/bin/env python3

"""
Module InterpersonalRhythmSynchronizer pour Jeffrey (Sprint 217).

Ce module analyse le rythme d'interaction (tempo, silence, type de réponses,
relances) et s'aligne sur le rythme relationnel préféré de l'utilisateur pour
moduler la vivacité ou la lenteur de Jeffrey de manière fluide.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import deque
from datetime import datetime
from typing import Any


class InterpersonalRhythmSynchronizer:
    """
    Analyse et synchronise le rythme conversationnel et relationnel entre
    Jeffrey et l'utilisateur.

    Ce module permet de moduler la vivacité ou la lenteur des réponses de Jeffrey
    en fonction du style d'interaction préféré de l'utilisateur, créant ainsi
    une harmonie rythmique entre les participants.
    """

    def __init__(self, storage_path: str | None = None, config_path: str | None = None):
        """
        Initialise le synchronisateur de rythmes interpersonnels.

        Args:
            storage_path: Chemin vers le fichier de stockage du modèle (optionnel)
            config_path: Chemin vers le fichier de configuration (optionnel)
        """
        self.logger = logging.getLogger(__name__)

        # Chemins de stockage et configuration
        self.storage_path = storage_path or os.path.join("data", "memory", "interaction_rhythm.json")
        self.config_path = config_path or os.path.join("config", "interaction_rhythms.json")

        # Charger la configuration
        self.reload_config()

        # Historique d'interactions
        # Limiter à 100 interactions récentes
        self.interaction_history = deque(maxlen=100)

        # Modèle de rythme de l'utilisateur
        self.user_rhythm_model = {
            "tempo": 0.5,  # Vitesse globale (0: très lent, 1: très rapide)
            # Niveau de détail (0: minimaliste, 1: très verbeux)
            "verbosity": 0.5,
            # Tolérance aux pauses (0: impatient, 1: apprécie les pauses)
            "pause_tolerance": 0.5,
            # Niveau d'initiative (0: réactif, 1: très proactif)
            "initiative_level": 0.5,
            # Cohérence du rythme (0: très variable, 1: très régulier)
            "consistency": 0.5,
            # Capacité multitâche (0: séquentiel, 1: parallèle)
            "multitasking": 0.5,
            # Changement de contexte (0: linéaire, 1: sauts fréquents)
            "context_switching": 0.5,
            # Préférence de profondeur (0: superficiel, 1: approfondi)
            "depth_preference": 0.5,
        }

        # Modèle de rythme courant de Jeffrey (peut être ajusté)
        self.jeffrey_rhythm_model = self.user_rhythm_model.copy()

        # Statistiques et métriques
        self.stats = {
            "interactions_analyzed": 0,
            "rhythm_adjustments": 0,
            "synchronization_events": 0,
            "current_synchronization_level": 0.5,
            "last_sync_update": datetime.now().isoformat(),
        }

        # Charger les données existantes
        self._load_data()

        self.logger.info("InterpersonalRhythmSynchronizer initialisé avec succès")

    def reload_config(self) -> bool:
        """
        Recharge la configuration du synchronisateur.

        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        # Configuration par défaut
        self.rhythm_prototypes = {
            "reserved": {
                "tempo": 0.3,
                "verbosity": 0.3,
                "pause_tolerance": 0.7,
                "initiative_level": 0.3,
                "consistency": 0.6,
                "multitasking": 0.3,
                "context_switching": 0.4,
                "depth_preference": 0.7,
            },
            "balanced": {
                "tempo": 0.5,
                "verbosity": 0.5,
                "pause_tolerance": 0.5,
                "initiative_level": 0.5,
                "consistency": 0.5,
                "multitasking": 0.5,
                "context_switching": 0.5,
                "depth_preference": 0.5,
            },
            "dynamic": {
                "tempo": 0.7,
                "verbosity": 0.7,
                "pause_tolerance": 0.3,
                "initiative_level": 0.7,
                "consistency": 0.4,
                "multitasking": 0.7,
                "context_switching": 0.7,
                "depth_preference": 0.3,
            },
            "analytical": {
                "tempo": 0.4,
                "verbosity": 0.6,
                "pause_tolerance": 0.6,
                "initiative_level": 0.4,
                "consistency": 0.8,
                "multitasking": 0.3,
                "context_switching": 0.3,
                "depth_preference": 0.9,
            },
            "spontaneous": {
                "tempo": 0.8,
                "verbosity": 0.5,
                "pause_tolerance": 0.2,
                "initiative_level": 0.8,
                "consistency": 0.3,
                "multitasking": 0.8,
                "context_switching": 0.8,
                "depth_preference": 0.4,
            },
        }

        self.rhythm_adaptation_policies = {
            "full_mirroring": {
                "description": "Ajuste complètement le rythme de Jeffrey pour correspondre à celui de l'utilisateur",
                "weight_self": 0.1,
                "weight_user": 0.9,
                "adaptation_speed": 0.8,
            },
            "gentle_alignment": {
                "description": "Ajuste doucement le rythme de Jeffrey vers celui de l'utilisateur",
                "weight_self": 0.3,
                "weight_user": 0.7,
                "adaptation_speed": 0.5,
            },
            "balanced_harmony": {
                "description": "Cherche un équilibre entre le rythme de Jeffrey et celui de l'utilisateur",
                "weight_self": 0.5,
                "weight_user": 0.5,
                "adaptation_speed": 0.4,
            },
            "subtle_influence": {
                "description": "Maintient principalement le rythme de Jeffrey avec une légère influence de l'utilisateur",
                "weight_self": 0.7,
                "weight_user": 0.3,
                "adaptation_speed": 0.3,
            },
            "rhythm_leadership": {
                "description": "Jeffrey maintient son rythme et guide doucement l'utilisateur",
                "weight_self": 0.9,
                "weight_user": 0.1,
                "adaptation_speed": 0.2,
            },
        }

        self.default_policy = "gentle_alignment"

        # Tentative de chargement de la configuration personnalisée
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, encoding="utf-8") as f:
                    config = json.load(f)

                if "rhythm_prototypes" in config:
                    self.rhythm_prototypes.update(config["rhythm_prototypes"])

                if "rhythm_adaptation_policies" in config:
                    self.rhythm_adaptation_policies.update(config["rhythm_adaptation_policies"])

                if "default_policy" in config:
                    self.default_policy = config["default_policy"]

                self.logger.info(f"Configuration chargée depuis {self.config_path}")
                return True
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement de la configuration: {e}")

        self.logger.info("Utilisation de la configuration par défaut")
        return False

    def _load_data(self) -> bool:
        """
        Charge les données existantes depuis le stockage.

        Returns:
            bool: True si le chargement a réussi, False sinon
        """
        try:
            if os.path.exists(self.storage_path):
                with open(self.storage_path, encoding="utf-8") as f:
                    data = json.load(f)

                if "user_rhythm_model" in data:
                    self.user_rhythm_model.update(data["user_rhythm_model"])

                if "jeffrey_rhythm_model" in data:
                    self.jeffrey_rhythm_model.update(data["jeffrey_rhythm_model"])

                if "stats" in data:
                    self.stats.update(data["stats"])

                if "interaction_history" in data:
                    # Convertir la liste en deque avec limite
                    self.interaction_history = deque(data["interaction_history"], maxlen=100)

                self.logger.info("Données de rythme interpersonnel chargées avec succès")
                return True
        except Exception as e:
            self.logger.error(f"Erreur lors du chargement des données de rythme: {e}")

        return False

    def _save_data(self) -> bool:
        """
        Sauvegarde les données dans le stockage.

        Returns:
            bool: True si la sauvegarde a réussi, False sinon
        """
        try:
            # Créer le répertoire parent si nécessaire
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)

            data = {
                "user_rhythm_model": self.user_rhythm_model,
                "jeffrey_rhythm_model": self.jeffrey_rhythm_model,
                "stats": self.stats,
                "interaction_history": list(self.interaction_history),
                "last_updated": datetime.now().isoformat(),
            }

            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            self.logger.info("Données de rythme interpersonnel sauvegardées avec succès")
            return True
        except Exception as e:
            self.logger.error(f"Erreur lors de la sauvegarde des données de rythme: {e}")
        return False

    def register_interaction(self, interaction_data: dict[str, Any], update_model: bool = True) -> dict[str, Any]:
        """
        Enregistre une nouvelle interaction et met à jour le modèle de rythme.

        Args:
            interaction_data: Données d'interaction contenant:
                - timestamp: Horodatage de l'interaction (optionnel)
                - message: Message de l'utilisateur
                - response: Réponse de Jeffrey
                - response_time: Temps de réponse en secondes
                - interaction_metrics: Métriques supplémentaires (optionnel)
            update_model: Si True, met à jour le modèle de rythme

        Returns:
            Dict: Résultats de l'analyse rythmique
        """
        try:
            # Normaliser les données
            timestamp = interaction_data.get("timestamp", datetime.now().isoformat())

            # Compléter les données d'interaction
            interaction = {
                "timestamp": timestamp,
                "message": interaction_data.get("message", ""),
                "response": interaction_data.get("response", ""),
                "response_time": interaction_data.get("response_time", 0.0),
                "metrics": {},
            }

            # Extraire les métriques rythmiques
            extracted_metrics = self._extract_rhythm_metrics(interaction_data)
            interaction["metrics"] = extracted_metrics

            # Ajouter à l'historique
            self.interaction_history.append(interaction)

            # Mettre à jour le modèle si demandé
            if update_model:
                self._update_user_rhythm_model(extracted_metrics)

            # Mettre à jour les statistiques
            self.stats["interactions_analyzed"] += 1
            self.stats["last_sync_update"] = datetime.now().isoformat()

            # Sauvegarder les données
            self._save_data()

            # Déterminer le prototype de rythme le plus proche
            nearest_prototype = self._find_nearest_rhythm_prototype(self.user_rhythm_model)

            # Calculer le niveau de synchronisation
            sync_level = self._calculate_synchronization_level()
            self.stats["current_synchronization_level"] = sync_level

            return {
                "success": True,
                "rhythm_metrics": extracted_metrics,
                "nearest_rhythm_prototype": nearest_prototype,
                "synchronization_level": sync_level,
                "user_rhythm_model": self.user_rhythm_model,
            }
        except Exception as e:
            self.logger.error(f"Erreur lors de l'enregistrement de l'interaction: {e}")
            return {"success": False, "error": str(e)}

    def adjust_jeffrey_rhythm(
        self, policy: str | None = None, custom_weights: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """
        Ajuste le rythme de Jeffrey en fonction du modèle de rythme de l'utilisateur.

        Args:
            policy: Nom de la politique d'adaptation à utiliser (optionnel)
            custom_weights: Poids personnalisés pour l'adaptation (optionnel)

        Returns:
            Dict: Résultats de l'ajustement
        """
        try:
            # Utiliser la politique par défaut si non spécifiée
            policy_name = policy or self.default_policy

            # Vérifier que la politique existe
            if policy_name not in self.rhythm_adaptation_policies:
                policy_name = self.default_policy

            # Récupérer les paramètres de la politique
            policy_params = self.rhythm_adaptation_policies[policy_name]

            # Utiliser des poids personnalisés s'ils sont fournis
            weight_self = (
                custom_weights.get("weight_self", policy_params["weight_self"])
                if custom_weights
                else policy_params["weight_self"]
            )
            weight_user = (
                custom_weights.get("weight_user", policy_params["weight_user"])
                if custom_weights
                else policy_params["weight_user"]
            )
            adaptation_speed = (
                custom_weights.get("adaptation_speed", policy_params["adaptation_speed"])
                if custom_weights
                else policy_params["adaptation_speed"]
            )

            # Sauvegarder l'ancien modèle pour comparaison
            old_model = self.jeffrey_rhythm_model.copy()

            # Mettre à jour chaque dimension du rythme
            for dimension in self.user_rhythm_model:
                # Calculer la nouvelle valeur
                target_value = (
                    weight_self * self.jeffrey_rhythm_model[dimension] + weight_user * self.user_rhythm_model[dimension]
                )

                # Appliquer la vitesse d'adaptation
                delta = target_value - self.jeffrey_rhythm_model[dimension]
                self.jeffrey_rhythm_model[dimension] += delta * adaptation_speed

                # Assurer que la valeur reste dans les limites
                self.jeffrey_rhythm_model[dimension] = max(0.0, min(1.0, self.jeffrey_rhythm_model[dimension]))

            # Calculer le changement global
            change_magnitude = sum(abs(old_model[d] - self.jeffrey_rhythm_model[d]) for d in old_model) / len(old_model)

            # Mettre à jour les statistiques
            self.stats["rhythm_adjustments"] += 1
            if change_magnitude > 0.05:  # Seuil arbitraire pour un "événement" de synchronisation
                self.stats["synchronization_events"] += 1

            # Sauvegarder les données
            self._save_data()

            return {
                "success": True,
                "policy_applied": policy_name,
                "old_rhythm_model": old_model,
                "new_rhythm_model": self.jeffrey_rhythm_model,
                "change_magnitude": change_magnitude,
                "is_significant_change": change_magnitude > 0.05,
            }
        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajustement du rythme de Jeffrey: {e}")
            return {"success": False, "error": str(e)}

    def apply_rhythm_parameters(self, response_params: dict[str, Any]) -> dict[str, Any]:
        """
        Applique les paramètres du rythme actuel à une réponse.

        Args:
            response_params: Paramètres de réponse à ajuster

        Returns:
            Dict: Paramètres de réponse ajustés
        """
        try:
            result = response_params.copy()

            # Ajuster le temps de réponse
            if "response_delay" in result:
                # Inverser le tempo (plus le tempo est élevé, plus le délai est court)
                tempo_factor = 1.0 - self.jeffrey_rhythm_model["tempo"]
                base_delay = result["response_delay"]

                # Ajuster le délai en fonction du tempo
                result["response_delay"] = base_delay * (0.5 + tempo_factor)

            # Ajuster la verbosité
            if "verbosity" in result:
                result["verbosity"] = max(
                    0.1,
                    min(1.0, result["verbosity"] * (0.5 + self.jeffrey_rhythm_model["verbosity"])),
                )

            # Ajuster les pauses et le rythme de parole
            if "speech_params" in result:
                speech_params = result["speech_params"].copy()

                # Vitesse de parole
                if "speed" in speech_params:
                    speech_params["speed"] = speech_params["speed"] * (0.7 + 0.6 * self.jeffrey_rhythm_model["tempo"])

                # Pauses
                if "pause_frequency" in speech_params:
                    pause_tolerance = self.jeffrey_rhythm_model["pause_tolerance"]
                    speech_params["pause_frequency"] = speech_params["pause_frequency"] * (0.5 + pause_tolerance)

                if "pause_duration" in speech_params:
                    pause_tolerance = self.jeffrey_rhythm_model["pause_tolerance"]
                    speech_params["pause_duration"] = speech_params["pause_duration"] * (0.5 + pause_tolerance)

                result["speech_params"] = speech_params

            # Ajuster le niveau de détail
            if "detail_level" in result:
                depth_factor = self.jeffrey_rhythm_model["depth_preference"]
                result["detail_level"] = max(0.1, min(1.0, result["detail_level"] * (0.5 + depth_factor)))

            # Ajuster la proactivité
            if "proactivity" in result:
                initiative = self.jeffrey_rhythm_model["initiative_level"]
                result["proactivity"] = max(0.1, min(1.0, result["proactivity"] * (0.5 + initiative)))

            return {
                "success": True,
                "original_params": response_params,
                "adjusted_params": result,
                "applied_rhythm_model": self.jeffrey_rhythm_model,
            }
        except Exception as e:
            self.logger.error(f"Erreur lors de l'application des paramètres de rythme: {e}")
            return {
                "success": False,
                "error": str(e),
                "params": response_params,
            }  # Retourner les paramètres originaux

    def get_rhythm_recommendations(self) -> dict[str, Any]:
        """
        Fournit des recommandations basées sur le modèle de rythme actuel.

        Returns:
            Dict: Recommandations pour l'interaction
        """
        try:
            # Trouver le prototype le plus proche pour l'utilisateur
            user_prototype = self._find_nearest_rhythm_prototype(self.user_rhythm_model)

            # Calculer le niveau de synchronisation
            sync_level = self._calculate_synchronization_level()

            # Générer des recommandations spécifiques
            recommendations = {
                "response_style": self._get_response_style_recommendation(),
                "interaction_pace": self._get_interaction_pace_recommendation(),
                "conversational_depth": self._get_conversational_depth_recommendation(),
                "initiative_taking": self._get_initiative_recommendation(),
            }

            # Générer des recommandations générales de haut niveau
            general_recommendation = self._generate_general_recommendation(user_prototype, sync_level, recommendations)

            return {
                "success": True,
                "user_rhythm_profile": user_prototype,
                "synchronization_level": sync_level,
                "specific_recommendations": recommendations,
                "general_recommendation": general_recommendation,
                "user_rhythm_model": self.user_rhythm_model,
                "jeffrey_rhythm_model": self.jeffrey_rhythm_model,
            }
        except Exception as e:
            self.logger.error(f"Erreur lors de la génération des recommandations de rythme: {e}")
            return {"success": False, "error": str(e)}

    def get_rhythm_statistics(self) -> dict[str, Any]:
        """
        Récupère des statistiques sur le rythme d'interaction.

        Returns:
            Dict: Statistiques du rythme d'interaction
        """
        return {
            "success": True,
            "stats": self.stats,
            "user_rhythm_model": self.user_rhythm_model,
            "jeffrey_rhythm_model": self.jeffrey_rhythm_model,
            "rhythm_profile": self._find_nearest_rhythm_prototype(self.user_rhythm_model),
            "interaction_count": len(self.interaction_history),
            "synchronization_level": self._calculate_synchronization_level(),
        }

    def _extract_rhythm_metrics(self, interaction_data: dict[str, Any]) -> dict[str, float]:
        """
        Extrait les métriques rythmiques d'une interaction.

        Args:
            interaction_data: Données d'interaction

        Returns:
            Dict: Métriques rythmiques extraites
        """
        metrics = {}

        # Extraire le tempo à partir du temps de réponse de l'utilisateur
        response_time = interaction_data.get("response_time", 0.0)
        if response_time > 0:
            # Normaliser : plus le temps est court, plus le tempo est élevé
            metrics["tempo"] = max(0.1, min(0.9, 1.0 - (response_time / 20.0)))

        # Extraire la verbosité à partir du message
        message = interaction_data.get("message", "")
        if message:
            # Normaliser : plus le message est long, plus la verbosité est élevée
            word_count = len(message.split())
            metrics["verbosity"] = max(0.1, min(0.9, word_count / 50.0))

        # Extraire la profondeur à partir de la complexité du message
        if message:
            # Mesures simples de complexité : longueur moyenne des mots, présence de termes complexes
            avg_word_length = sum(len(word) for word in message.split()) / max(1, len(message.split()))
            complex_word_ratio = len([w for w in message.split() if len(w) > 8]) / max(1, len(message.split()))

            metrics["depth_preference"] = max(0.1, min(0.9, (avg_word_length / 10.0) * 0.5 + complex_word_ratio * 0.5))

        # Extraire l'initiative à partir des questions et propositions
        if message:
            question_marks = message.count("?")
            suggestion_phrases = sum(
                1
                for phrase in [
                    "pourrions-nous",
                    "devrions-nous",
                    "pourrait-on",
                    "je suggère",
                    "je propose",
                ]
                if phrase in message.lower()
            )

            metrics["initiative_level"] = max(
                0.1, min(0.9, (question_marks / 3.0) * 0.6 + (suggestion_phrases / 2.0) * 0.4)
            )

        # Utiliser les métriques supplémentaires si disponibles
        additional_metrics = interaction_data.get("interaction_metrics", {})
        for metric, value in additional_metrics.items():
            if metric in self.user_rhythm_model and isinstance(value, (int, float)):
                metrics[metric] = max(0.0, min(1.0, value))

        return metrics

    def _update_user_rhythm_model(self, metrics: dict[str, float]) -> None:
        """
        Met à jour le modèle de rythme de l'utilisateur avec de nouvelles métriques.

        Args:
            metrics: Nouvelles métriques extraites
        """
        # Facteur d'apprentissage (plus petit = changement plus progressif)
        learning_rate = 0.1

        # Mettre à jour chaque dimension si disponible
        for dimension, value in metrics.items():
            if dimension in self.user_rhythm_model:
                # Mise à jour progressive
                self.user_rhythm_model[dimension] = (1 - learning_rate) * self.user_rhythm_model[
                    dimension
                ] + learning_rate * value

    def _find_nearest_rhythm_prototype(self, rhythm_model: dict[str, float]) -> str:
        """
        Trouve le prototype de rythme le plus proche d'un modèle donné.

        Args:
            rhythm_model: Modèle de rythme à comparer

        Returns:
            str: Nom du prototype le plus proche
        """
        min_distance = float("inf")
        nearest_prototype = "balanced"  # Par défaut

        for prototype_name, prototype in self.rhythm_prototypes.items():
            # Calculer la distance euclidienne
            distance = 0.0
            for dimension, value in prototype.items():
                if dimension in rhythm_model:
                    distance += (value - rhythm_model[dimension]) ** 2

            distance = math.sqrt(distance)

            if distance < min_distance:
                min_distance = distance
                nearest_prototype = prototype_name

        return nearest_prototype

    def _calculate_synchronization_level(self) -> float:
        """
        Calcule le niveau actuel de synchronisation entre Jeffrey et l'utilisateur.

        Returns:
            float: Niveau de synchronisation (0-1)
        """
        # Calculer la similitude entre les modèles de rythme
        similarity = 0.0
        dimension_count = 0

        for dimension in self.user_rhythm_model:
            if dimension in self.jeffrey_rhythm_model:
                # Différence inversée (plus la différence est faible, plus la similitude est élevée)
                dimension_similarity = 1.0 - abs(
                    self.user_rhythm_model[dimension] - self.jeffrey_rhythm_model[dimension]
                )
                similarity += dimension_similarity
                dimension_count += 1

        # Calculer la moyenne
        if dimension_count > 0:
            return similarity / dimension_count
        else:
            return 0.5  # Valeur par défaut

    def _get_response_style_recommendation(self) -> dict[str, Any]:
        """
        Génère une recommandation pour le style de réponse.

        Returns:
            Dict: Recommandation de style de réponse
        """
        verbosity = self.jeffrey_rhythm_model["verbosity"]
        depth = self.jeffrey_rhythm_model["depth_preference"]

        if verbosity > 0.7 and depth > 0.7:
            style = "détaillé et expressif"
            description = "Réponses complètes et riches avec beaucoup de contexte et d'explications"
        elif verbosity > 0.7 and depth <= 0.3:
            style = "bavard mais concis"
            description = "Réponses énergiques et expressives tout en restant directes sur le fond"
        elif verbosity <= 0.3 and depth > 0.7:
            style = "sobre mais profond"
            description = "Réponses concises mais conceptuellement riches et réfléchies"
        elif verbosity <= 0.3 and depth <= 0.3:
            style = "minimaliste"
            description = "Réponses brèves, directes et sans fioritures"
        else:
            style = "équilibré"
            description = "Réponses modérément détaillées avec un bon équilibre entre concision et richesse"

        return {
            "style": style,
            "description": description,
            "verbosity_level": verbosity,
            "depth_level": depth,
        }

    def _get_interaction_pace_recommendation(self) -> dict[str, Any]:
        """
        Génère une recommandation pour le rythme d'interaction.

        Returns:
            Dict: Recommandation de rythme d'interaction
        """
        tempo = self.jeffrey_rhythm_model["tempo"]
        pause_tolerance = self.jeffrey_rhythm_model["pause_tolerance"]

        if tempo > 0.7 and pause_tolerance <= 0.3:
            pace = "rapide et continu"
            description = "Interactions rapides avec peu de pauses entre les échanges"
        elif tempo > 0.7 and pause_tolerance > 0.7:
            pace = "dynamique mais réfléchi"
            description = "Alternance entre des échanges rapides et des moments de réflexion"
        elif tempo <= 0.3 and pause_tolerance > 0.7:
            pace = "lent et contemplatif"
            description = "Échanges espacés avec beaucoup de temps pour la réflexion"
        elif tempo <= 0.3 and pause_tolerance <= 0.3:
            pace = "délibéré"
            description = "Rythme lent mais régulier, sans longues pauses"
        else:
            pace = "modéré"
            description = "Rythme équilibré avec alternance naturelle entre échanges et pauses"

        return {
            "pace": pace,
            "description": description,
            "tempo_level": tempo,
            "pause_tolerance_level": pause_tolerance,
        }

    def _get_conversational_depth_recommendation(self) -> dict[str, Any]:
        """
        Génère une recommandation pour la profondeur conversationnelle.

        Returns:
            Dict: Recommandation de profondeur conversationnelle
        """
        depth = self.jeffrey_rhythm_model["depth_preference"]
        context_switching = self.jeffrey_rhythm_model["context_switching"]

        if depth > 0.7 and context_switching <= 0.3:
            style = "approfondi et focalisé"
            description = "Explorer un sujet en profondeur avant de passer à autre chose"
        elif depth > 0.7 and context_switching > 0.7:
            style = "exploratoire et connectif"
            description = "Établir des connexions profondes entre plusieurs sujets"
        elif depth <= 0.3 and context_switching > 0.7:
            style = "survol dynamique"
            description = "Aborder de nombreux sujets sans s'attarder sur aucun"
        elif depth <= 0.3 and context_switching <= 0.3:
            style = "concis et séquentiel"
            description = "Progresser méthodiquement à travers les sujets sans approfondir"
        else:
            style = "équilibré"
            description = "Alternance entre profondeur et diversité selon le flux de conversation"

        return {
            "style": style,
            "description": description,
            "depth_level": depth,
            "context_switching_level": context_switching,
        }

    def _get_initiative_recommendation(self) -> dict[str, Any]:
        """
        Génère une recommandation pour la prise d'initiative.

        Returns:
            Dict: Recommandation de prise d'initiative
        """
        initiative = self.jeffrey_rhythm_model["initiative_level"]
        consistency = self.jeffrey_rhythm_model["consistency"]

        if initiative > 0.7 and consistency > 0.7:
            style = "leadership structuré"
            description = "Guider activement la conversation de manière cohérente et prévisible"
        elif initiative > 0.7 and consistency <= 0.3:
            style = "impulsif et directif"
            description = "Introduire spontanément de nouvelles directions dans la conversation"
        elif initiative <= 0.3 and consistency > 0.7:
            style = "réactif et fiable"
            description = "Suivre le fil de l'utilisateur avec une approche constante et prévisible"
        elif initiative <= 0.3 and consistency <= 0.3:
            style = "adaptable et réceptif"
            description = "S'ajuster aux changements de l'utilisateur sans imposer de direction"
        else:
            style = "collaboration équilibrée"
            description = "Alterner naturellement entre suivre et guider en fonction du contexte"

        return {
            "style": style,
            "description": description,
            "initiative_level": initiative,
            "consistency_level": consistency,
        }

    def _generate_general_recommendation(
        self, user_prototype: str, sync_level: float, specific_recs: dict[str, dict[str, Any]]
    ) -> str:
        """
        Génère une recommandation générale basée sur l'ensemble des facteurs.

        Args:
            user_prototype: Prototype de rythme de l'utilisateur
            sync_level: Niveau de synchronisation
            specific_recs: Recommandations spécifiques

        Returns:
            str: Recommandation générale
        """
        # Décrire le profil rythmique de l'utilisateur
        user_profile_descriptions = {
            "reserved": "un style d'interaction réservé, réfléchi et méthodique",
            "balanced": "un style d'interaction équilibré et adaptable",
            "dynamic": "un style d'interaction dynamique, énergique et prompt",
            "analytical": "un style d'interaction analytique, structuré et approfondi",
            "spontaneous": "un style d'interaction spontané, varié et enthousiaste",
        }

        user_profile_desc = user_profile_descriptions.get(user_prototype, "un style d'interaction unique")

        # Base de la recommandation
        if sync_level > 0.8:
            base_rec = f"L'utilisateur montre {user_profile_desc}. La synchronisation est excellente, maintenir ce rythme relationnel."
        elif sync_level > 0.6:
            base_rec = f"L'utilisateur montre {user_profile_desc}. La synchronisation est bonne, mais quelques ajustements mineurs seraient bénéfiques."
        elif sync_level > 0.4:
            base_rec = f"L'utilisateur montre {user_profile_desc}. La synchronisation est moyenne, des ajustements significatifs sont recommandés."
        else:
            base_rec = f"L'utilisateur montre {user_profile_desc}. La synchronisation est faible, une recalibration importante du rythme est nécessaire."

        # Ajouter des détails spécifiques
        response_style = specific_recs.get("response_style", {}).get("style", "équilibré")
        pace = specific_recs.get("interaction_pace", {}).get("pace", "modéré")
        depth_style = specific_recs.get("conversational_depth", {}).get("style", "équilibré")

        details = f" Privilégier un style de réponse {response_style}, avec un rythme {pace} et une approche conversationnelle {depth_style}."

        return base_rec + details
