#!/usr/bin/env python
"""
emotion_visualizer.py - Visualisation des émotions pour Jeffrey

Ce module contient la classe EmotionVisualizer qui permet d'afficher visuellement
l'état émotionnel de Jeffrey en fonction des données reçues du pont émotionnel.
Il prend en charge les transitions émotionnelles, les effets d'intensité,
et la journalisation de chaque changement.

Sprint 13: Ajout du support pour l'affichage du lien affectif et des
informations de relation entre l'utilisateur et Jeffrey.
"""

import logging
import time
from typing import Any


class EmotionVisualizer:
    def __init__(self, emotion_bridge=None):
        self.logger = logging.getLogger("EmotionVisualizer")
        self.current_emotion = None
        self.intensity = 0
        self.secondary = None
        self.last_update = 0
        self.emotion_bridge = emotion_bridge

        # Lien affectif (Sprint 13)
        self.user_id = "default_user"
        self.trust_level = 0.0
        self.warmth_level = 0.0
        self.proximity_level = 0.0
        self.is_anchor_user = False
        self.emotional_capacity = 1.0
        self.fatigue_level = 0.0
        self.affective_profile_enabled = False

        # Journaliser les changements dans un fichier
        if not self.logger.handlers:
            handler = logging.FileHandler("logs/emotion_visuals.log", mode="a", encoding="utf-8")
            formatter = logging.Formatter("%(asctime)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        if self.emotion_bridge:
            try:
                self.emotion_bridge.register_emotion_change_callback(self.render)
                self.logger.info("Visualiseur émotionnel connecté au pont émotionnel.")

                # Vérifier si le pont émotionnel supporte les profils affectifs (Sprint 13)
                if hasattr(self.emotion_bridge, "get_affective_profile_summary"):
                    self.affective_profile_enabled = True
                    self.logger.info("Support du profil affectif détecté et activé.")
            except AttributeError:
                self.logger.warning("Pont émotionnel ne supporte pas les callbacks.")

    def render(self, emotion: str, intensity: float = 0.5, secondary: str | None = None):
        """
        Affiche une représentation visuelle de l’émotion principale reçue.
        """
        self.current_emotion = emotion
        self.intensity = intensity
        self.secondary = secondary
        self.last_update = time.time()

        display = self._generate_display(emotion, intensity, secondary)
        print(f"{time.strftime('%H:%M:%S')} - {display}")
        self.logger.info(f"Émotion affichée : {emotion} (intensité {intensity:.2f}, secondaire={secondary})")

    def _generate_display(self, emotion: str, intensity: float, secondary: str | None) -> str:
        """
        Génère une représentation textuelle stylisée de l'émotion.
        """
        bar = "█" * int(intensity * 20)
        sec = f" ({secondary})" if secondary else ""
        return f"[{emotion.upper()}{sec}] {bar}"

    def summarize_last_emotion(self) -> str:
        """
        Retourne un résumé de la dernière émotion affichée.

        Returns:
            str: Résumé de la dernière émotion
        """
        # Version standard
        emotion_summary = f"{self.current_emotion} (intensité={self.intensity:.2f}, secondaire={self.secondary})"

        # Ajouter les informations de lien affectif si disponibles (Sprint 13)
        if self.affective_profile_enabled:
            bond_level = self.trust_level * 0.35 + self.warmth_level * 0.35 + self.proximity_level * 0.3
            bond_info = f", lien affectif={bond_level:.2f}"

            # Ajouter les indicateurs d'état
            if self.is_anchor_user:
                bond_info += " ⚓"
            if self.fatigue_level > 0.5:
                bond_info += " 😴"

            return emotion_summary + bond_info

        return emotion_summary

    def _update_affective_profile(self) -> None:
        """
        Met à jour les informations de profil affectif depuis le pont émotionnel.
        """
        if not self.affective_profile_enabled or not self.emotion_bridge:
            return

        try:
            # Récupérer le résumé du profil affectif
            profile = self.emotion_bridge.get_affective_profile_summary(self.user_id)

            if isinstance(profile, dict) and "error" in profile:
                self.logger.warning(f"Erreur lors de la récupération du profil affectif: {profile['error']}")
                return

            # Mettre à jour les propriétés
            self.trust_level = profile.get("affective_bond", {}).get("trust", 0.0)
            self.warmth_level = profile.get("affective_bond", {}).get("warmth", 0.0)
            self.proximity_level = profile.get("affective_bond", {}).get("proximity", 0.0)
            self.is_anchor_user = profile.get("anchor_status", {}).get("is_anchor", False)

            # État émotionnel
            self.emotional_capacity = profile.get("emotional_state", {}).get("capacity", 1.0)
            self.fatigue_level = profile.get("emotional_state", {}).get("fatigue", 0.0)

            self.logger.info(f"Profil affectif mis à jour pour l'utilisateur: {self.user_id}")

        except Exception as e:
            self.logger.warning(f"Erreur lors de la mise à jour du profil affectif: {e}")

    def _generate_affective_display(self) -> str:
        """
        Génère une représentation visuelle du lien affectif.

        Returns:
            str: Représentation textuelle du lien affectif
        """
        trust_bar = "♥" * int(self.trust_level * 10)
        warmth_bar = "☀" * int(self.warmth_level * 10)
        proximity_bar = "⌘" * int(self.proximity_level * 10)

        # Icône spéciale pour les utilisateurs d'ancrage
        anchor_icon = "⚓ " if self.is_anchor_user else ""

        # Indicateur de fatigue ou saturation
        state_icon = ""
        if self.fatigue_level > 0.5:
            state_icon = "😴 "  # Fatigué
        elif self.emotional_capacity < 0.3:
            state_icon = "🔋 "  # Batterie faible

        return (
            f"\n{anchor_icon}Lien affectif avec {self.user_id}:\n"
            f"  Confiance: [{trust_bar.ljust(10)}] {self.trust_level:.2f}\n"
            f"  Chaleur:   [{warmth_bar.ljust(10)}] {self.warmth_level:.2f}\n"
            f"  Proximité: [{proximity_bar.ljust(10)}] {self.proximity_level:.2f}\n"
            f"  État: {state_icon}{self._get_bond_description()}\n"
        )

    def _get_bond_description(self) -> str:
        """
        Retourne une description du lien affectif actuel.

        Returns:
            str: Description textuelle du lien
        """
        # Calculer le niveau global de lien
        bond_level = self.trust_level * 0.35 + self.warmth_level * 0.35 + self.proximity_level * 0.3

        if bond_level > 0.8:
            return "Lien profond et confiant"
        elif bond_level > 0.6:
            return "Forte connexion affective"
        elif bond_level > 0.4:
            return "Relation positive en développement"
        elif bond_level > 0.2:
            return "Premiers signes d'attachement"
        else:
            return "Relation neutre ou récente"

    def get_affective_bond_summary(self) -> dict[str, Any]:
        """
        Retourne un résumé du lien affectif actuel.

        Returns:
            Dict: Résumé du lien affectif
        """
        # Mise à jour des données si possible
        if self.emotion_bridge and self.affective_profile_enabled:
            self._update_affective_profile()

        # Calculer le niveau global de lien
        bond_level = self.trust_level * 0.35 + self.warmth_level * 0.35 + self.proximity_level * 0.3

        return {
            "user_id": self.user_id,
            "trust_level": self.trust_level,
            "warmth_level": self.warmth_level,
            "proximity_level": self.proximity_level,
            "global_bond": bond_level,
            "is_anchor_user": self.is_anchor_user,
            "emotional_capacity": self.emotional_capacity,
            "fatigue_level": self.fatigue_level,
            "description": self._get_bond_description(),
        }

    def display_affective_bond(self, user_id: str = None) -> None:
        """
        Affiche le lien affectif pour un utilisateur spécifique.

        Args:
            user_id: Identifiant de l'utilisateur (optionnel, utilise l'actuel par défaut)
        """
        if not self.affective_profile_enabled:
            print("Lien affectif non disponible - profil affectif désactivé.")
            return

        # Mettre à jour l'ID utilisateur si fourni
        if user_id:
            self.user_id = user_id

        # Mettre à jour le profil
        self._update_affective_profile()

        # Afficher le lien
        print(self._generate_affective_display())
