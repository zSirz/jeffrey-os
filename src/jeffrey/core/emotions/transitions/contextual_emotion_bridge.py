#!/usr/bin/env python

"""
contextual_emotion_bridge.py - Pont de liaison contextuelle des émotions pour Jeffrey

Module central permettant de lier le contexte cognitif courant (discussion, situation,
souvenir actif) à l'état émotionnel de Jeffrey.

Ce module implémente les fonctionnalités suivantes:
- Analyse contextuelle des émotions en fonction des situations
- Passerelle entre la mémoire sensorielle/affective et l'état émotionnel
- Modulation de l'émotion adaptée au contexte de conversation
- Injection d'émotions tirées des souvenirs
- Synchronisation bidirectionnelle avec les rendus visuels immersifs

Fonctionnalités PACKS:
- PACK 11: Impact des relations sur les émotions
- PACK 20: Sensations corporelles et leur impact émotionnel
- PACK 21: Souvenirs émotionnels pour modulation affective
"""

from __future__ import annotations

import logging
import random
import time
from collections.abc import Callable
from datetime import datetime
from typing import Any

from jeffrey.core.emotions.dynamic_emotion_renderer import DynamicEmotionRenderer

# Imports des composants centraux
from jeffrey.core.emotions.emotional_engine import EmotionalEngine
from jeffrey.core.emotions.emotional_learning import EmotionalLearning

# Import des systèmes de mémoire et contexte
from jeffrey.core.memoire_sensorielle import MémoireSensorielle

# Import du système de profil affectif (Sprint 13)
try:
    from jeffrey.core.emotions.affective_profile import AffectiveProfile, AffectiveProfileManager

    affective_profile_available = True
except ImportError:
    affective_profile_available = False

# Import optionnel, avec fallback si non disponible
try:
    from jeffrey.core.memory.souvenir import SouvenirManager

    souvenir_manager_available = True
except ImportError:
    souvenir_manager_available = False


class ContextualEmotionBridge:
    """
    Pont de liaison entre le contexte cognitif et l'état émotionnel.

    Cette classe centralise la gestion du lien entre:
    - Le contexte actuel (conversation, situation, environnement)
    - La mémoire sensorielle et affective
    - L'état émotionnel et sa visualisation

    Elle sert d'interface unifiée pour les modules externes souhaitant
    modifier l'état émotionnel de Jeffrey en fonction du contexte.
    """

    def __init__(
        self,
        emotional_engine: EmotionalEngine | None = None,
        emotional_learning: EmotionalLearning | None = None,
        emotion_renderer: DynamicEmotionRenderer | None = None,
        memoire_sensorielle_path: str | None = None,
        affective_profile_manager=None,
        default_user_id: str = "default_user",
    ):
        """
        Initialise le pont contextuel d'émotions.

        Args:
            emotional_engine: Moteur émotionnel
            emotional_learning: Système d'apprentissage émotionnel
            emotion_renderer: Renderer dynamique d'émotions
            memoire_sensorielle_path: Chemin du fichier de mémoire sensorielle
            affective_profile_manager: Gestionnaire de profils affectifs (optionnel)
            default_user_id: ID utilisateur par défaut
        """
        self.logger = logging.getLogger("jeffrey.contextual_emotion_bridge")

        # Initialiser le moteur émotionnel si non fourni
        self.emotional_engine = emotional_engine or EmotionalEngine()

        # Initialiser l'apprentissage émotionnel si non fourni
        self.emotional_learning = emotional_learning or EmotionalLearning()

        # Initialiser la mémoire sensorielle
        self.memoire_sensorielle = MémoireSensorielle(memoire_sensorielle_path)

        # Initialiser le renderer d'émotions si non fourni
        self.emotion_renderer = emotion_renderer or DynamicEmotionRenderer(
            emotional_engine=self.emotional_engine, learning_system=self.emotional_learning
        )

        # État du contexte actuel
        self.current_context = {
            "situation": "normal",
            "relation_level": 0.5,
            "conversation_topic": None,
            "active_memory": None,
            "emotional_state": {
                "primary": "neutre",
                "secondary": None,
                "intensity": 0.5,
                "valence": 0.0,
                "arousal": 0.5,
                "last_update": datetime.now().isoformat(),
            },
            "location": None,
            "timestamp": datetime.now().isoformat(),
            "user_id": default_user_id,  # Ajout de l'ID utilisateur (Sprint 13)
            "environment": {"noise_level": "low", "brightness": "medium", "privacy": "secure"},
        }

        # Configuration
        self.update_interval = 2.0  # Intervalle minimum entre mises à jour (secondes)
        self.last_update_time = 0.0
        self.use_emotion_blending = True  # Utiliser les transitions entre émotions
        self.enable_context_logging = True  # Journaliser les changements de contexte
        self.emotion_log_path = "logs/emotion_trace.json"
        self.default_user_id = default_user_id

        # Mappings émotionnels
        self._init_emotional_mappings()

        # Hooks et callbacks
        self.context_change_callbacks = []
        self.emotion_change_callbacks = []

        # Initialisation des systèmes de mémoire et profil affectif (Sprint 13)
        self.logger.info("Pont émotionnel contextuel initialisé")

        # Mémoire émotionnelle (connectée automatiquement si disponible)
        try:
            from jeffrey.core.emotional_memory import EmotionalMemory

            self.emotional_memory = EmotionalMemory()
        except ImportError:
            self.logger.warning("EmotionalMemory introuvable. Pont mémoire non connecté.")
            self.emotional_memory = None

        # Initialiser ou connecter le gestionnaire de profils affectifs (Sprint 13)
        if affective_profile_available:
            self.affective_profile_manager = affective_profile_manager
            if self.affective_profile_manager is None:
                self.affective_profile_manager = AffectiveProfileManager()
            self.logger.info("Gestionnaire de profils affectifs connecté")
        else:
            self.affective_profile_manager = None
            self.logger.warning("AffectiveProfileManager non disponible. Système de lien affectif désactivé.")

        # Initialiser le profil affectif actuel (par défaut)
        self.current_affective_profile = None
        if self.affective_profile_manager:
            self.current_affective_profile = self.affective_profile_manager.get_profile(default_user_id)

        # Détection de l'utilisateur d'ancrage (Sprint 13)
        self.anchor_user_id = None
        if self.affective_profile_manager:
            self.anchor_user_id = self.affective_profile_manager.detect_anchor_user()

    def _init_emotional_mappings(self):
        """
        Initialise les mappings émotionnels pour les différents contextes.
        """
        # Mapping des situations aux tendances émotionnelles
        self.situation_emotion_map = {
            "danger": {
                "primary": "peur",
                "secondary": "vigilance",
                "intensity": 0.8,
                "valence": -0.7,
            },
            "réunion": {
                "primary": "concentration",
                "secondary": "calme",
                "intensity": 0.6,
                "valence": 0.2,
            },
            "conversation": {
                "primary": "intérêt",
                "secondary": "curiosité",
                "intensity": 0.6,
                "valence": 0.4,
            },
            "détente": {
                "primary": "calme",
                "secondary": "contentement",
                "intensity": 0.5,
                "valence": 0.6,
            },
            "jeu": {"primary": "joie", "secondary": "excitation", "intensity": 0.7, "valence": 0.8},
            "conflit": {
                "primary": "anxiété",
                "secondary": "tristesse",
                "intensity": 0.7,
                "valence": -0.6,
            },
            "solitude": {
                "primary": "tristesse",
                "secondary": "ennui",
                "intensity": 0.5,
                "valence": -0.4,
            },
            "retrouvailles": {
                "primary": "joie",
                "secondary": "amour",
                "intensity": 0.8,
                "valence": 0.9,
            },
        }

        # Mapping des types de souvenirs aux émotions
        self.memory_tag_emotion_map = {
            "heureux": {
                "primary": "joie",
                "secondary": "nostalgie",
                "intensity": 0.7,
                "valence": 0.8,
            },
            "triste": {
                "primary": "tristesse",
                "secondary": "mélancolie",
                "intensity": 0.6,
                "valence": -0.6,
            },
            "effrayant": {
                "primary": "peur",
                "secondary": "anxiété",
                "intensity": 0.7,
                "valence": -0.7,
            },
            "amusant": {
                "primary": "joie",
                "secondary": "amusement",
                "intensity": 0.6,
                "valence": 0.7,
            },
            "touchant": {
                "primary": "amour",
                "secondary": "gratitude",
                "intensity": 0.7,
                "valence": 0.8,
            },
            "marquant": {
                "primary": "surprise",
                "secondary": "intérêt",
                "intensity": 0.6,
                "valence": 0.3,
            },
            "embarrassant": {
                "primary": "gêne",
                "secondary": "anxiété",
                "intensity": 0.5,
                "valence": -0.4,
            },
            "fier": {"primary": "fierté", "secondary": "joie", "intensity": 0.7, "valence": 0.8},
        }

        # Mapping des sujets de conversation aux émotions
        self.topic_emotion_map = {
            "science": {
                "primary": "curiosité",
                "secondary": "intérêt",
                "intensity": 0.6,
                "valence": 0.5,
            },
            "technologie": {
                "primary": "enthousiasme",
                "secondary": "curiosité",
                "intensity": 0.7,
                "valence": 0.6,
            },
            "art": {
                "primary": "inspiration",
                "secondary": "joie",
                "intensity": 0.7,
                "valence": 0.7,
            },
            "problème": {
                "primary": "concentration",
                "secondary": "détermination",
                "intensity": 0.6,
                "valence": 0.2,
            },
            "conflit": {
                "primary": "inquiétude",
                "secondary": "nervosité",
                "intensity": 0.6,
                "valence": -0.4,
            },
            "personnel": {
                "primary": "empathie",
                "secondary": "intérêt",
                "intensity": 0.6,
                "valence": 0.3,
            },
        }

    def update_emotion_from_context(self, context_data: dict[str, Any]) -> dict[str, Any]:
        """
        Met à jour l'état émotionnel en fonction du contexte fourni.

        Args:
            context_data: Dictionnaire contenant les informations de contexte
                Clés possibles:
                - "situation": Type de situation (conversation, réunion, etc.)
                - "relation": Niveau de relation avec l'interlocuteur (0-1)
                - "topic": Sujet de conversation
                - "memory_tag": Tag de souvenir associé
                - "sentiment_analysis": Analyse de sentiment d'un message
                - "user_state": État émotionnel perçu de l'utilisateur
                - "environment": Informations sur l'environnement
                - "user_id": Identifiant de l'utilisateur (Sprint 13)

        Returns:
            Dict: État émotionnel mis à jour
        """
        # Vérifier si assez de temps s'est écoulé depuis la dernière mise à jour
        current_time = time.time()
        if current_time - self.last_update_time < self.update_interval:
            # Retourner l'état actuel sans mise à jour
            return self.current_context["emotional_state"]

        # Mettre à jour l'horodatage
        self.last_update_time = current_time

        # Sauvegarder l'ancien état pour la journalisation des changements
        old_state = self.current_context["emotional_state"].copy()

        # Mettre à jour l'ID utilisateur si fourni (Sprint 13)
        user_id = context_data.get("user_id", self.default_user_id)
        if user_id != self.current_context.get("user_id"):
            self.current_context["user_id"] = user_id
            # Mettre à jour le profil affectif actuel
            if self.affective_profile_manager:
                self.current_affective_profile = self.affective_profile_manager.get_profile(user_id)
                self.logger.info(f"Profil affectif changé pour l'utilisateur: {user_id}")

                # Vérifier si c'est un utilisateur d'ancrage
                if self.current_affective_profile.is_anchor_user:
                    self.logger.info(f"Interaction avec un utilisateur d'ancrage: {user_id}")

        # Mettre à jour le contexte actuel avec les nouvelles données
        if "situation" in context_data:
            self.current_context["situation"] = context_data["situation"]

        if "relation" in context_data:
            self.current_context["relation_level"] = max(0.0, min(1.0, context_data["relation"]))

            # Mettre à jour le niveau de relation dans le renderer si disponible
            if self.emotion_renderer:
                self.emotion_renderer.set_relationship_level(self.current_context["relation_level"], user_id)

        if "topic" in context_data:
            self.current_context["conversation_topic"] = context_data["topic"]

        if "memory_tag" in context_data:
            self.current_context["active_memory"] = context_data["memory_tag"]

        if "location" in context_data:
            self.current_context["location"] = context_data["location"]

        if "environment" in context_data:
            self.current_context["environment"].update(context_data["environment"])

        # Calculer le nouvel état émotionnel basé sur tous les facteurs
        new_state = self._compute_emotional_state_from_context()

        # Appliquer les modifications basées sur le profil affectif (Sprint 13)
        if self.current_affective_profile and affective_profile_available:
            modified_state = self._apply_affective_profile_influence(new_state)
            # Si la modification a réussi, utiliser l'état modifié
            if modified_state:
                new_state = modified_state

        # Mettre à jour l'état interne
        self.current_context["emotional_state"] = new_state
        self.current_context["timestamp"] = datetime.now().isoformat()

        # Appliquer le nouvel état émotionnel
        self._apply_emotional_state(new_state)

        # Journaliser le changement si activé
        if self.enable_context_logging:
            self._log_context_change(old_state, new_state, context_data)

        # Appeler les callbacks de changement de contexte
        for callback in self.context_change_callbacks:
            try:
                callback(self.current_context)
            except Exception as e:
                self.logger.warning(f"Erreur lors de l'appel d'un callback de contexte: {e}")

        # Injecter automatiquement dans la mémoire émotionnelle si disponible
        memory_entry = None
        if hasattr(self, "emotional_memory") and self.emotional_memory:
            try:
                memory_entry = self.emotional_memory.inject_memory(
                    emotion=new_state["primary"],
                    intensity=new_state["intensity"],
                    valence=new_state["valence"],
                    context=context_data,
                    source="context_bridge",
                )

                # Sauvegarder la mémoire émotionnelle après la mise à jour
                self.emotional_memory.save_memory_if_needed()
            except Exception as e:
                self.logger.warning(f"Erreur lors de l'injection mémoire émotionnelle: {e}")

        # Mettre à jour le profil affectif en fonction de l'émotion (Sprint 13)
        if self.current_affective_profile and affective_profile_available:
            try:
                self.current_affective_profile.update_profile(
                    emotion=new_state["primary"],
                    intensity=new_state["intensity"],
                    valence=new_state["valence"],
                    context=context_data,
                )

                # Si c'est un souvenir important (forte intensité), l'enregistrer dans le profil
                if memory_entry and new_state["intensity"] > 0.7:
                    memories_list = [memory_entry]
                    self.current_affective_profile.update_emotional_memories(memories_list)

                # Vérifier si l'utilisateur est devenu un utilisateur d'ancrage
                if self.current_affective_profile.detect_anchor_user():
                    self.anchor_user_id = user_id
                    self.logger.info(f"Nouvel utilisateur d'ancrage détecté: {user_id}")

            except Exception as e:
                self.logger.warning(f"Erreur lors de la mise à jour du profil affectif: {e}")

        # Évolution affective naturelle (Sprint 13)
        if self.current_affective_profile and affective_profile_available:
            try:
                self.current_affective_profile.tick_emotional_evolution()
            except Exception as e:
                self.logger.warning(f"Erreur lors de l'évolution naturelle du profil affectif: {e}")
        return new_state

    def get_emotion_summary(self) -> str:
        """
        Retourne un résumé textuel lisible de l'état émotionnel actuel.

        Returns:
            str: Résumé émotionnel formaté pour affichage ou logs
        """
        state = self.current_context["emotional_state"]
        primary = state["primary"]
        secondary = state["secondary"]
        intensity = state["intensity"]
        valence = state["valence"]

        # Formater l'intensité pour l'affichage
        intensity_desc = "faiblement"
        if intensity > 0.7:
            intensity_desc = "fortement"
        elif intensity > 0.4:
            intensity_desc = "modérément"

        # Formater la valence pour l'affichage
        valence_desc = ""
        if valence > 0.5:
            valence_desc = "positivement"
        elif valence < -0.5:
            valence_desc = "négativement"
        elif valence > 0.1:
            valence_desc = "légèrement positivement"
        elif valence < -0.1:
            valence_desc = "légèrement négativement"

        # Générer le résumé de base
        summary = f"Jeffrey ressent {intensity_desc} de la {primary}"

        # Ajouter l'émotion secondaire si présente
        if secondary:
            summary += f" avec une touche de {secondary}"

        # Ajouter la valence si significative
        if valence_desc:
            summary += f", {valence_desc}"

        # Ajouter le contexte si pertinent
        if self.current_context["situation"] != "normal":
            summary += f" en raison de la situation de {self.current_context['situation']}"

        if self.current_context["active_memory"]:
            summary += f", évoquée par un souvenir {self.current_context['active_memory']}"

        if self.current_context["conversation_topic"]:
            summary += f" lors d'une conversation sur {self.current_context['conversation_topic']}"

        # Ajouter un point final
        summary += "."

        return summary

    def inject_memory_emotion(self, memory_tag: str, intensity_factor: float = 1.0) -> dict[str, Any]:
        """
        Module l'émotion actuelle en fonction d'un souvenir spécifique.

        Args:
            memory_tag: Tag du souvenir à injecter (heureux, triste, effrayant, etc.)
            intensity_factor: Facteur de modulation de l'intensité (0.0 à 2.0)

        Returns:
            Dict: État émotionnel mis à jour
        """
        # Normaliser l'intensity_factor
        intensity_factor = max(0.1, min(2.0, intensity_factor))

        # Rechercher un souvenir spécifique dans la mémoire sensorielle
        souvenir = None
        if memory_tag in self.memory_tag_emotion_map:
            # Rechercher un souvenir associé à ce tag
            matching_souvenirs = []
            for s in self.memoire_sensorielle.souvenirs_emotionnels:
                # Vérifier si l'émotion du souvenir correspond au tag
                if s.emotion == self.memory_tag_emotion_map[memory_tag]["primary"]:
                    matching_souvenirs.append(s)

            # Si des souvenirs correspondants sont trouvés, en sélectionner un
            if matching_souvenirs:
                souvenir = random.choice(matching_souvenirs)

        # Utiliser également SouvenirManager si disponible
        if souvenir_manager_available:
            try:
                sm = SouvenirManager()
                if memory_tag == "heureux":
                    souvenirs_heureux = sm.get_souvenirs_heureux()
                    if souvenirs_heureux:
                        # Sélectionner un souvenir au hasard
                        choix = random.choice(souvenirs_heureux)
                        # Enregistrer ce souvenir comme actif dans le contexte
                        self.current_context["active_memory"] = choix["titre"]
            except Exception as e:
                self.logger.warning(f"Erreur lors de l'accès aux souvenirs: {e}")

        # Obtenir les informations émotionnelles liées au tag
        memory_emotion = self.memory_tag_emotion_map.get(
            memory_tag, {"primary": "neutre", "secondary": None, "intensity": 0.5, "valence": 0.0}
        )

        # Créer un contexte temporaire pour injecter l'émotion du souvenir
        memory_context = {
            "memory_tag": memory_tag,
            "immediate": True,  # Indiquer que cette mise à jour doit être immédiate
        }

        if souvenir:
            # Si un souvenir spécifique a été trouvé, utiliser son intensité
            memory_context["intensity_override"] = souvenir.intensite * intensity_factor
            self.current_context["active_memory"] = f"{memory_tag} ({souvenir.type})"

            # Réactiver le souvenir pour l'avenir
            souvenir.derniere_reactivation = time.time()
            souvenir.nb_reactivations += 1

            # Réactiver un souvenir similaire dans la mémoire émotionnelle
            if self.emotional_memory:
                self.emotional_memory.reactivate_similar_memory(
                    primary=memory_emotion["primary"], valence=memory_emotion["valence"]
                )
        else:
            # Sinon, utiliser l'intensité par défaut du mapping
            memory_context["intensity_override"] = memory_emotion["intensity"] * intensity_factor
            self.current_context["active_memory"] = memory_tag

        # Modifier le nouvel état émotionnel
        return self.update_emotion_from_context(memory_context)

    def process_message_context(self, message: str, sentiment_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Traite un message pour en extraire le contexte et mettre à jour l'état émotionnel.

        Args:
            message: Message à analyser
            sentiment_data: Données d'analyse de sentiment (optionnel)

        Returns:
            Dict: État émotionnel mis à jour
        """
        # Si aucune donnée de sentiment n'est fournie, utiliser des valeurs par défaut
        if not sentiment_data:
            sentiment_data = {"emotion": "neutre", "valence": 0.0, "confidence": 0.5}

        # Déterminer le sujet du message (simpliste pour illustration)
        topic = self._infer_topic_from_message(message)

        # Créer le contexte de la conversation
        context = {
            "situation": "conversation",
            "topic": topic,
            "sentiment_analysis": sentiment_data,
        }

        # Mettre à jour l'émotion basée sur ce contexte
        return self.update_emotion_from_context(context)

    def process_physical_contact(
        self, zone: str, type_contact: str, intensite: float = 0.5, user_id: str = "unknown"
    ) -> dict[str, Any]:
        """
        Traite un contact physique et met à jour l'état émotionnel en conséquence.

        Args:
            zone: Zone du corps concernée par le contact
            type_contact: Type de contact (caresse, toucher, etc.)
            intensite: Intensité du contact (0.0 à 1.0)
            user_id: Identifiant de l'utilisateur

        Returns:
            Dict: État émotionnel mis à jour
        """
        # Enregistrer le contact dans la mémoire sensorielle
        self.memoire_sensorielle.enregistrer_contact(
            zone=zone, type_contact=type_contact, intensité=intensite, user_id=user_id
        )

        # Calculer la valence du contact (positive ou négative)
        valence = self._compute_contact_valence(zone, type_contact, intensite)

        # Déterminer l'émotion appropriée en fonction du type de contact et de la valence
        emotion = "neutre"
        if valence > 0.7:
            emotion = "joie"
        elif valence > 0.3:
            emotion = "contentement"
        elif valence < -0.7:
            emotion = "colère"
        elif valence < -0.3:
            emotion = "inconfort"

        # Créer le contexte du contact
        context = {
            "situation": "contact_physique",
            "zone": zone,
            "type_contact": type_contact,
            "valence": valence,
            "intensity_override": intensite,
            "emotion_override": emotion,
        }

        # Appliquer le contexte au moteur émotionnel
        return self.update_emotion_from_context(context)

    def register_context_change_callback(self, callback: Callable) -> None:
        """
        Enregistre une fonction callback pour les changements de contexte.

        Args:
            callback: Fonction à appeler lors d'un changement de contexte
        """
        if callback not in self.context_change_callbacks:
            self.context_change_callbacks.append(callback)

    def register_emotion_change_callback(self, callback: Callable) -> None:
        """
        Enregistre une fonction callback pour les changements d'émotion.

        Args:
            callback: Fonction à appeler lors d'un changement d'émotion
        """
        if callback not in self.emotion_change_callbacks:
            self.emotion_change_callbacks.append(callback)

    def connect_ui_renderer(self, renderer) -> None:
        """
        Connecte un renderer d'interface utilisateur.

        Args:
            renderer: Renderer d'interface à connecter
        """
        if self.emotion_renderer:
            self.emotion_renderer.connect_ui_renderer(renderer)
            self.logger.info("UI renderer connecté au pont émotionnel")

    def set_update_interval(self, seconds: float) -> None:
        """
        Définit l'intervalle minimum entre les mises à jour émotionnelles.

        Args:
            seconds: Intervalle en secondes
        """
        self.update_interval = max(0.1, seconds)

    def enable_emotion_blending(self, enabled: bool = True) -> None:
        """
        Active ou désactive les transitions douces entre émotions.

        Args:
            enabled: True pour activer, False pour désactiver
        """
        self.use_emotion_blending = enabled

    def get_current_context(self) -> dict[str, Any]:
        """
        Retourne le contexte actuel complet.

        Returns:
            Dict: Contexte actuel
        """
        return self.current_context.copy()

    def reset_to_neutral(self) -> dict[str, Any]:
        """
        Réinitialise l'état émotionnel à un état neutre.

        Returns:
            Dict: État émotionnel neutre
        """
        neutral_state = {
            "primary": "neutre",
            "secondary": None,
            "intensity": 0.5,
            "valence": 0.0,
            "arousal": 0.5,
            "last_update": datetime.now().isoformat(),
        }

        # Mettre à jour l'état interne
        self.current_context["emotional_state"] = neutral_state

        # Appliquer l'état neutre
        self._apply_emotional_state(neutral_state)

        return neutral_state

    def get_current_emotion(self) -> str:
        """
        Retourne l'émotion principale actuelle.

        Returns:
            str: Émotion principale
        """
        return self.current_context["emotional_state"]["primary"]

    def get_emotion_intensity(self) -> float:
        """
        Retourne l'intensité de l'émotion actuelle.

        Returns:
            float: Intensité (0.0 à 1.0)
        """
        return self.current_context["emotional_state"]["intensity"]

    def _compute_emotional_state_from_context(self) -> dict[str, Any]:
        """
        Calcule un nouvel état émotionnel basé sur tous les facteurs du contexte actuel.

        Returns:
            Dict: Nouvel état émotionnel
        """
        # Initialisation avec l'état actuel comme base
        current_state = self.current_context["emotional_state"]

        # Si une émotion est explicitement fournie, l'utiliser directement
        if hasattr(self.current_context, "emotion_override"):
            new_state = current_state.copy()
            new_state["primary"] = self.current_context["emotion_override"]
            new_state["intensity"] = self.current_context.get("intensity_override", 0.7)
            new_state["last_update"] = datetime.now().isoformat()
            return new_state

        # Évaluer les facteurs d'influence
        influences = []

        # 1. Influence de la situation
        situation = self.current_context["situation"]
        if situation in self.situation_emotion_map:
            situation_influence = self.situation_emotion_map[situation].copy()
            situation_influence["weight"] = 0.7  # Forte influence de la situation
            influences.append(situation_influence)

        # 2. Influence du souvenir actif
        memory_tag = self.current_context["active_memory"]
        if memory_tag and memory_tag in self.memory_tag_emotion_map:
            memory_influence = self.memory_tag_emotion_map[memory_tag].copy()
            memory_influence["weight"] = 0.6  # Influence moyenne-forte des souvenirs
            influences.append(memory_influence)

        # 3. Influence du sujet de conversation
        topic = self.current_context["conversation_topic"]
        if topic and topic in self.topic_emotion_map:
            topic_influence = self.topic_emotion_map[topic].copy()
            topic_influence["weight"] = 0.4  # Influence moyenne des sujets
            influences.append(topic_influence)

        # 4. Influence de l'analyse de sentiment (si présente)
        if "sentiment_analysis" in self.current_context:
            sentiment = self.current_context["sentiment_analysis"]
            if sentiment and "emotion" in sentiment:
                sentiment_influence = {
                    "primary": sentiment["emotion"],
                    "intensity": sentiment.get("confidence", 0.5),
                    "valence": sentiment.get("valence", 0.0),
                    "weight": 0.5,  # Influence moyenne-forte du sentiment
                }
                influences.append(sentiment_influence)

        # 5. Influence de la relation (modifie l'intensité et la valence)
        relation_level = self.current_context["relation_level"]
        if relation_level > 0.7:
            # Relation forte : amplifie les émotions positives
            relation_influence = {
                "primary": "joie" if relation_level > 0.9 else current_state["primary"],
                "secondary": "amour" if relation_level > 0.8 else "affection",
                "intensity": min(1.0, relation_level * 1.2),
                "valence": relation_level - 0.2,  # Plus positive avec relation forte
                "weight": 0.3,  # Influence modérée de la relation
            }
            influences.append(relation_influence)

        # Si aucune influence n'est trouvée, conserver l'état actuel
        if not influences:
            new_state = current_state.copy()
            new_state["last_update"] = datetime.now().isoformat()
            return new_state

        # Calculer l'émotion résultante par pondération des influences
        return self._compute_weighted_emotion(influences, current_state)

    def _compute_weighted_emotion(
        self, influences: list[dict[str, Any]], current_state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Calcule un état émotionnel pondéré à partir des influences.

        Args:
            influences: Liste d'influences émotionnelles avec poids
            current_state: État émotionnel actuel

        Returns:
            Dict: Nouvel état émotionnel pondéré
        """
        # Si priorité immédiate (comme pour un souvenir injecté), utiliser directement
        for influence in influences:
            if self.current_context.get("immediate", False):
                new_state = current_state.copy()
                new_state["primary"] = influence["primary"]
                new_state["secondary"] = influence.get("secondary")
                new_state["intensity"] = self.current_context.get("intensity_override", influence["intensity"])
                new_state["valence"] = influence.get("valence", 0.0)
                new_state["last_update"] = datetime.now().isoformat()
                return new_state

        # Somme des poids pour normalisation
        total_weight = sum(infl.get("weight", 0.5) for infl in influences)

        # Si pas de poids significatif, conserver l'état actuel
        if total_weight < 0.01:
            new_state = current_state.copy()
            new_state["last_update"] = datetime.now().isoformat()
            return new_state

        # Pondération pour chaque propriété
        # Pour l'émotion primaire, on utilise l'influence ayant le poids le plus fort
        sorted_influences = sorted(influences, key=lambda x: x.get("weight", 0.0), reverse=True)
        primary_emotion = sorted_influences[0]["primary"]

        # Pour l'émotion secondaire, on utilise la deuxième plus forte ou celle du premier si unique
        secondary_emotion = None
        if len(sorted_influences) > 1:
            secondary_emotion = sorted_influences[1]["primary"]
        else:
            secondary_emotion = sorted_influences[0].get("secondary")

        # Pour l'intensité et la valence, on fait une moyenne pondérée
        weighted_intensity = (
            sum(infl.get("intensity", 0.5) * infl.get("weight", 0.5) for infl in influences) / total_weight
        )
        weighted_valence = sum(infl.get("valence", 0.0) * infl.get("weight", 0.5) for infl in influences) / total_weight

        # L'intensité actuelle a aussi une influence pour éviter les changements trop brusques
        if "intensity" in current_state:
            current_weight = 0.3  # Poids de l'état actuel (inertie émotionnelle)
            weighted_intensity = (weighted_intensity * total_weight + current_state["intensity"] * current_weight) / (
                total_weight + current_weight
            )

        # Créer le nouvel état émotionnel
        new_state = {
            "primary": primary_emotion,
            "secondary": secondary_emotion,
            "intensity": weighted_intensity,
            "valence": weighted_valence,
            "arousal": 0.5,  # Valeur par défaut, à améliorer dans le futur
            "last_update": datetime.now().isoformat(),
        }

        return new_state

    def _apply_emotional_state(self, state: dict[str, Any]) -> None:
        """
        Applique un état émotionnel aux systèmes connectés.

        Args:
            state: État émotionnel à appliquer
        """
        # Extraire les informations principales
        emotion = state["primary"]
        intensity = state["intensity"]
        secondary = state["secondary"]

        # Mettre à jour le moteur émotionnel
        if self.emotional_engine:
            try:
                self.emotional_engine.update_emotion(emotion=emotion, intensity=intensity, secondary_emotion=secondary)
            except Exception as e:
                self.logger.warning(f"Erreur lors de la mise à jour du moteur émotionnel: {e}")

        # Mettre à jour le système d'apprentissage
        if self.emotional_learning:
            try:
                self.emotional_learning.observe_emotion(emotion)
                # Mettre à jour le profil périodiquement
                if random.random() < 0.2:  # 20% de chance de mise à jour
                    self.emotional_learning.update_profile()
                    self.emotional_learning.export_profile()
            except Exception as e:
                self.logger.warning(f"Erreur lors de la mise à jour du système d'apprentissage: {e}")

        # Mettre à jour le renderer d'émotions
        if self.emotion_renderer:
            try:
                if self.use_emotion_blending and hasattr(self.emotion_renderer, "trigger_transition"):
                    # Obtenir l'émotion actuelle pour la transition
                    current_emotion = self.emotion_renderer.current_emotion
                    if current_emotion != emotion:
                        self.emotion_renderer.trigger_transition(
                            from_emotion=current_emotion, to_emotion=emotion, duration=0.8
                        )
                else:
                    self.emotion_renderer.render_emotion(
                        emotion=emotion,
                        intensity=intensity,
                        secondary_emotion=secondary,
                        source="contextual_bridge",
                    )
            except Exception as e:
                self.logger.warning(f"Erreur lors de la mise à jour du renderer d'émotions: {e}")

        # Appeler les callbacks de changement d'émotion
        for callback in self.emotion_change_callbacks:
            try:
                callback(emotion, intensity, secondary)
            except Exception as e:
                self.logger.warning(f"Erreur lors de l'appel d'un callback d'émotion: {e}")

    def _compute_contact_valence(self, zone: str, type_contact: str, intensite: float) -> float:
        """
        Calcule la valence émotionnelle d'un contact physique.

        Args:
            zone: Zone du corps concernée
            type_contact: Type de contact
            intensite: Intensité du contact

        Returns:
            float: Valence du contact (-1.0 à 1.0)
        """
        # Déléguer à la mémoire sensorielle qui a cette logique
        if hasattr(self.memoire_sensorielle, "_calculer_valence"):
            return self.memoire_sensorielle._calculer_valence(type_contact, intensite)

        # Sinon, implémentation de fallback simplifiée
        valences_par_défaut = {
            "caresse": 0.8,
            "toucher": 0.2,
            "appui": 0.0,
            "tape": -0.3,
            "grattement": 0.1,
            "effleurement": 0.4,
            "massage": 0.7,
            "bisou": 0.9,
            "frôlement": 0.3,
            "pincement": -0.5,
        }

        # Valence de base selon le type
        valence_base = valences_par_défaut.get(type_contact, 0.0)

        # Ajuster selon l'intensité (trop intense peut devenir négatif)
        if intensite > 0.8 and type_contact not in ["caresse", "effleurement", "bisou"]:
            # Réduire la valence pour les contacts très intenses
            valence_ajustée = valence_base - (intensite - 0.8) * 2.0
        elif intensite < 0.2 and valence_base > 0:
            # Réduire la valence pour les contacts trop légers (sauf si déjà négatif)
            valence_ajustée = valence_base * intensite * 5.0
        else:
            valence_ajustée = valence_base

        # Limiter entre -1.0 et 1.0
        return max(-1.0, min(1.0, valence_ajustée))

    def _infer_topic_from_message(self, message: str) -> str | None:
        """
        Infère le sujet d'un message (implémentation simpliste).

        Args:
            message: Message à analyser

        Returns:
            str: Sujet inféré ou None
        """
        message_lower = message.lower()

        # Recherche simple par mots-clés (à remplacer par une analyse plus sophistiquée)
        if any(word in message_lower for word in ["science", "recherche", "étude", "physique", "chimie"]):
            return "science"
        elif any(word in message_lower for word in ["technologie", "tech", "ordinateur", "logiciel", "ia"]):
            return "technologie"
        elif any(word in message_lower for word in ["art", "musique", "peinture", "dessin", "film"]):
            return "art"
        elif any(word in message_lower for word in ["problème", "bug", "erreur", "résoudre", "solution"]):
            return "problème"
        elif any(word in message_lower for word in ["conflit", "dispute", "désaccord", "débat"]):
            return "conflit"
        elif any(word in message_lower for word in ["personnel", "vie", "sentiment", "émotion", "famille"]):
            return "personnel"

        return None

    def _log_context_change(
        self, old_state: dict[str, Any], new_state: dict[str, Any], context_data: dict[str, Any]
    ) -> None:
        """
        Journalise un changement de contexte émotionnel.

        Args:
            old_state: Ancien état émotionnel
            new_state: Nouvel état émotionnel
            context_data: Données de contexte ayant déclenché le changement
        """
        import json
        import os

        # Vérifier si les états sont différents
        if old_state["primary"] == new_state["primary"] and abs(old_state["intensity"] - new_state["intensity"]) < 0.1:
            # Pas de changement significatif, ne pas journaliser
            return

        # Préparer les données de journal
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "old_state": old_state,
            "new_state": new_state,
            "context_triggers": context_data,
            "context_summary": {
                "situation": self.current_context["situation"],
                "relation_level": self.current_context["relation_level"],
                "active_memory": self.current_context["active_memory"],
                "conversation_topic": self.current_context["conversation_topic"],
                "user_id": self.current_context.get(
                    "user_id", self.default_user_id
                ),  # Ajout de l'ID utilisateur (Sprint 13)
            },
        }

        # Ajouter les informations de profil affectif si disponibles (Sprint 13)
        if self.current_affective_profile and affective_profile_available:
            try:
                # Récupérer uniquement les informations essentielles pour le log
                affective_info = {
                    "dominant_emotion": self.current_affective_profile.dominant_emotion,
                    "trust_level": round(self.current_affective_profile.trust_level, 2),
                    "warmth_level": round(self.current_affective_profile.warmth_level, 2),
                    "proximity_level": round(self.current_affective_profile.proximity_level, 2),
                    "fatigue_level": round(self.current_affective_profile.fatigue_level, 2),
                    "emotional_capacity": round(self.current_affective_profile.emotional_capacity / 100, 2),
                    "is_anchor_user": self.current_affective_profile.is_anchor_user,
                }
                log_entry["affective_profile"] = affective_info
            except Exception as e:
                self.logger.warning(f"Erreur lors de la journalisation du profil affectif: {e}")

        try:
            # Créer le répertoire de logs si nécessaire
            os.makedirs(os.path.dirname(self.emotion_log_path), exist_ok=True)

            # Charger le journal existant (s'il existe)
            existing_log = []
            if os.path.exists(self.emotion_log_path):
                with open(self.emotion_log_path, encoding="utf-8") as f:
                    existing_log = json.load(f)

            # Ajouter la nouvelle entrée
            existing_log.append(log_entry)

            # Limiter la taille du journal (garder les 1000 dernières entrées)
            if len(existing_log) > 1000:
                existing_log = existing_log[-1000:]

            # Sauvegarder le journal mis à jour
            with open(self.emotion_log_path, "w", encoding="utf-8") as f:
                json.dump(existing_log, f, ensure_ascii=False, indent=2)

        except Exception as e:
            self.logger.warning(f"Erreur lors de la journalisation du changement de contexte: {e}")

    def _apply_affective_profile_influence(self, emotional_state: dict[str, Any]) -> dict[str, Any]:
        """
        Applique l'influence du profil affectif sur l'état émotionnel.

        Cette méthode modifie l'état émotionnel en fonction du profil affectif
        de l'utilisateur actuel, prenant en compte la fatigue émotionnelle,
        la saturation et le niveau de lien.

        Args:
            emotional_state (Dict): État émotionnel à modifier

        Returns:
            Dict: État émotionnel modifié
        """
        if not self.current_affective_profile or not affective_profile_available:
            return emotional_state

        # Créer une copie de l'état pour modification
        modified_state = emotional_state.copy()

        # Obtenir le statut de capacité émotionnelle
        capacity_status = self.current_affective_profile.get_emotional_capacity_status()

        # Appliquer la fatigue émotionnelle - réduire l'intensité
        if capacity_status["fatigued"]:
            fatigue_factor = 1.0 - (capacity_status["fatigue_level"] * 0.5)  # Max 50% de réduction
            modified_state["intensity"] *= fatigue_factor

            # Ajouter une émotion secondaire de fatigue si l'intensité est réduite significativement
            if fatigue_factor < 0.7 and not modified_state.get("secondary"):
                modified_state["secondary"] = "fatigue"

            self.logger.debug(f"Fatigue émotionnelle appliquée: intensité réduite à {modified_state['intensity']:.2f}")

        # Appliquer la saturation émotionnelle - affecter la valence
        if capacity_status["saturated"]:
            # Saturation réduit la valence positive et augmente la valence négative
            saturation_factor = capacity_status["percentage"]  # Entre 0 et 1
            if modified_state["valence"] > 0:
                modified_state["valence"] *= saturation_factor
            elif modified_state["valence"] < 0:
                modified_state["valence"] /= max(0.1, saturation_factor)  # Amplification limitée

            self.logger.debug(f"Saturation émotionnelle appliquée: valence ajustée à {modified_state['valence']:.2f}")

        # Appliquer l'influence du niveau de lien affectif
        bond_level = self.current_affective_profile._calculate_global_bond()

        # Les liens forts amplifient les émotions positives et modèrent les négatives
        if bond_level > 0.6:  # Lien fort
            if modified_state["valence"] > 0:
                # Amplifier les émotions positives
                positive_boost = 1.0 + (bond_level - 0.6) * 0.5  # Max +20% à bond_level=1.0
                modified_state["intensity"] = min(1.0, modified_state["intensity"] * positive_boost)
                modified_state["valence"] = min(1.0, modified_state["valence"] * positive_boost)
            elif modified_state["valence"] < 0:
                # Modérer les émotions négatives
                negative_dampening = 1.0 - (bond_level - 0.6) * 0.3  # Max -12% à bond_level=1.0
                modified_state["intensity"] *= negative_dampening

            self.logger.debug(f"Lien affectif fort appliqué: intensité ajustée à {modified_state['intensity']:.2f}")

        # Lien faible peut rendre les émotions plus froides et distantes
        elif bond_level < 0.3:  # Lien faible
            cold_factor = 1.0 - bond_level  # Plus haut pour les liens faibles

            # Réduire l'intensité émotionnelle
            modified_state["intensity"] *= 1.0 - (cold_factor * 0.3)  # Max -30% à bond_level=0

            # Ajouter une émotion secondaire de distance pour les interactions neutres
            if abs(modified_state["valence"]) < 0.3 and not modified_state.get("secondary"):
                modified_state["secondary"] = "indifférence"

            self.logger.debug(
                f"Lien affectif faible appliqué: réaction plus froide, intensité={modified_state['intensity']:.2f}"
            )

        return modified_state

    def summarize_emotional_memory(self) -> str:
        """
        Retourne un résumé de la mémoire émotionnelle si disponible.
        """
        if self.emotional_memory:
            return self.emotional_memory.summarize_emotions()
        return "Mémoire émotionnelle non connectée."

    def get_affective_profile_summary(self, user_id: str = None) -> dict[str, Any]:
        """
        Retourne un résumé du profil affectif pour un utilisateur spécifique.

        Args:
            user_id (str): Identifiant de l'utilisateur (optionnel, utilise l'actuel par défaut)

        Returns:
            Dict: Résumé du profil affectif
        """
        if not affective_profile_available or not self.affective_profile_manager:
            return {"error": "Système de profil affectif non disponible"}

        # Utiliser l'utilisateur actuel si non spécifié
        user_id = user_id or self.current_context.get("user_id", self.default_user_id)

        # Récupérer le profil
        profile = self.affective_profile_manager.get_profile(user_id, create_if_missing=False)

        if not profile:
            return {"error": f"Profil affectif non trouvé pour l'utilisateur {user_id}"}

        # Récupérer le résumé
        return profile.get_profile_summary()

    def get_anchored_user(self) -> str | None:
        """
        Retourne l'identifiant de l'utilisateur d'ancrage principal.

        Returns:
            str or None: Identifiant de l'utilisateur d'ancrage principal
        """
        if not affective_profile_available or not self.affective_profile_manager:
            return None

        return self.anchor_user_id

    def reinforce_affective_bond(self, user_id: str = None, factor: float = 1.0) -> bool:
        """
        Renforce le lien affectif avec un utilisateur spécifique.

        Args:
            user_id (str): Identifiant de l'utilisateur (optionnel, utilise l'actuel par défaut)
            factor (float): Facteur de renforcement (1.0 = normal)

        Returns:
            bool: True si le renforcement a réussi, False sinon
        """
        if not affective_profile_available or not self.affective_profile_manager:
            return False

        # Utiliser l'utilisateur actuel si non spécifié
        user_id = user_id or self.current_context.get("user_id", self.default_user_id)

        # Récupérer le profil
        profile = self.affective_profile_manager.get_profile(user_id)

        if not profile:
            return False

        # Renforcer le lien
        profile.reinforce_bond(factor)

        # Vérifier si l'utilisateur est devenu un utilisateur d'ancrage
        if profile.is_anchor_user and user_id.lower() != self.anchor_user_id:
            self.anchor_user_id = user_id.lower()
            self.logger.info(f"Nouvel utilisateur d'ancrage détecté: {user_id}")

        return True


# Factory function for easy instantiation
def create_emotion_bridge(
    emotional_engine=None,
    emotional_learning=None,
    emotion_renderer=None,
    memoire_sensorielle_path=None,
) -> ContextualEmotionBridge:
    """
    Crée et initialise un pont émotionnel contextuel.

    Args:
        emotional_engine: Moteur émotionnel optionnel
        emotional_learning: Système d'apprentissage émotionnel optionnel
        emotion_renderer: Renderer dynamique d'émotions optionnel
        memoire_sensorielle_path: Chemin du fichier de mémoire sensorielle optionnel

    Returns:
        ContextualEmotionBridge: Pont émotionnel contextuel initialisé
    """
    try:
        # Créer les composants nécessaires s'ils ne sont pas fournis
        if not emotional_engine:
            emotional_engine = EmotionalEngine()

        if not emotional_learning:
            emotional_learning = EmotionalLearning()
            # Charger le profil existant si disponible
            emotional_learning.load_profile()

        if not emotion_renderer and emotional_engine:
            emotion_renderer = DynamicEmotionRenderer(
                emotional_engine=emotional_engine, learning_system=emotional_learning
            )

        # Créer le pont émotionnel
        bridge = ContextualEmotionBridge(
            emotional_engine=emotional_engine,
            emotional_learning=emotional_learning,
            emotion_renderer=emotion_renderer,
            memoire_sensorielle_path=memoire_sensorielle_path,
        )

        return bridge

    except Exception as e:
        logging.getLogger("jeffrey.factory").error(f"Erreur lors de la création du pont émotionnel: {e}")
        # Créer une version minimale du pont, sans les composants qui ont échoué
        return ContextualEmotionBridge()


# Exemple d'utilisation
if __name__ == "__main__":
    # Ce code s'exécute uniquement si le fichier est lancé directement
    bridge = create_emotion_bridge()

    # Exemple de mise à jour du contexte
    context = {"situation": "conversation", "relation": 0.7, "topic": "technologie"}

    bridge.update_emotion_from_context(context)
    print(bridge.get_emotion_summary())

    # Exemple d'injection d'un souvenir
    bridge.inject_memory_emotion("heureux")
    print(bridge.get_emotion_summary())
