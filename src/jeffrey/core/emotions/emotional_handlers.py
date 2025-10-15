#!/usr/bin/env python3

"""
Module contenant les handlers émotionnels de Jeffrey.
Ces méthodes gèrent les détections, enregistrements et analyses des émotions.
"""

import logging
from typing import Any


class EmotionalHandlers:
    """
    Classe regroupant les handlers émotionnels de Jeffrey.
    """

    def __init__(self):
        """
        Initialise le logger pour les handlers émotionnels.
        """
        self.logger = logging.getLogger(__name__)

    # -------------------------------------------------------------------
    # Méthodes pour EmotionalTrustModeler (Sprint 213)
    # -------------------------------------------------------------------

    def evaluer_niveau_confiance(self, interaction_data: dict[str, Any]) -> dict[str, Any]:
        """
        Évalue le niveau de confiance émotionnelle actuel entre Jeffrey et l'utilisateur.
        Implémenté pour le Sprint 213.

        Args:
            interaction_data: Données d'interaction pour l'évaluation

        Returns:
            Dict: Résultats de l'évaluation
        """
        if not hasattr(self, "emotional_trust_modeler") or not self.emotional_trust_modeler:
            # Tenter d'initialiser le modélisateur si non disponible
            if not self.initialiser_emotional_trust_modeler():
                return {"success": False, "reason": "EmotionalTrustModeler non initialisé"}

        try:
            # Enregistrer l'interaction et obtenir le niveau de confiance
            result = self.emotional_trust_modeler.register_interaction(interaction_data)

            if hasattr(self, "journal_emotionnel") and self.journal_emotionnel is not None:
                # Ajouter une entrée au journal émotionnel si le niveau de confiance change significativement
                if result.get("trust_delta", 0) > 0.05:
                    niveau = result.get("new_trust_score", 0.5)
                    niveau_texte = "faible" if niveau < 0.4 else "modéré" if niveau < 0.7 else "élevé"
                    self.journal_emotionnel.ajouter_entree(
                        pensee=f"La confiance émotionnelle avec l'utilisateur évolue positivement (niveau {niveau_texte}: {niveau:.2f})",
                        emotion="confiance",
                        intensite=niveau,
                    )
                elif result.get("trust_delta", 0) < -0.05:
                    niveau = result.get("new_trust_score", 0.5)
                    niveau_texte = "faible" if niveau < 0.4 else "modéré" if niveau < 0.7 else "élevé"
                    self.journal_emotionnel.ajouter_entree(
                        pensee=f"La confiance émotionnelle avec l'utilisateur diminue (niveau {niveau_texte}: {niveau:.2f})",
                        emotion="inquiétude",
                        intensite=0.5,
                    )

            return {"success": True, **result}

        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation du niveau de confiance: {e}")
            return {"success": False, "reason": str(e)}

    def enregistrer_revelation_personnelle(self, revelation_data: dict[str, Any]) -> dict[str, Any]:
        """
        Enregistre une révélation personnelle de l'utilisateur.
        Implémenté pour le Sprint 213.

        Args:
            revelation_data: Données de la révélation

        Returns:
            Dict: Résultats de l'enregistrement
        """
        if not hasattr(self, "emotional_trust_modeler") or not self.emotional_trust_modeler:
            # Tenter d'initialiser le modélisateur si non disponible
            if not self.initialiser_emotional_trust_modeler():
                return {"success": False, "reason": "EmotionalTrustModeler non initialisé"}

        try:
            # Enregistrer la révélation
            result = self.emotional_trust_modeler.register_personal_revelation(revelation_data)

            # Journaliser la révélation personnelle
            if hasattr(self, "journal_emotionnel") and self.journal_emotionnel is not None:
                intimite = revelation_data.get("intimacy_level", 0.5)
                categorie = revelation_data.get("category", "general")
                self.journal_emotionnel.ajouter_entree(
                    pensee=f"L'utilisateur a partagé une révélation personnelle ({categorie}, niveau d'intimité: {intimite:.2f})",
                    emotion="empathie",
                    intensite=intimite * 0.8,  # Proportionnel au niveau d'intimité
                )

            return {"success": True, **result}

        except Exception as e:
            self.logger.error(f"Erreur lors de l'enregistrement de la révélation personnelle: {e}")
            return {"success": False, "reason": str(e)}

    # -------------------------------------------------------------------
    # Méthodes pour ReactiveCompassionEngine (Sprint 214)
    # -------------------------------------------------------------------

    def detecter_detresse(
        self, message: str, vocal_features: dict[str, Any] | None = None, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Détecte les signes de détresse dans un message utilisateur.
        Implémenté pour le Sprint 214.

        Args:
            message: Texte du message de l'utilisateur
            vocal_features: Caractéristiques vocales extraites (optionnel)
            context: Contexte de la conversation (optionnel)

        Returns:
            Dict: Résultats de la détection
        """
        if not hasattr(self, "reactive_compassion_engine") or not self.reactive_compassion_engine:
            # Tenter d'initialiser le moteur si non disponible
            if not self.initialiser_reactive_compassion_engine():
                return {"success": False, "reason": "ReactiveCompassionEngine non initialisé"}

        try:
            # Détecter la détresse
            result = self.reactive_compassion_engine.detect_distress(message, vocal_features, context)

            # Journaliser la détection de détresse
            if (
                result.get("distress_detected", False)
                and hasattr(self, "journal_emotionnel")
                and self.journal_emotionnel is not None
            ):
                distress_level = result.get("distress_level", "légère")
                primary_emotion = result.get("primary_emotion", "unknown")

                self.journal_emotionnel.ajouter_entree(
                    pensee=f"J'ai détecté une détresse {distress_level} chez l'utilisateur (émotion: {primary_emotion})",
                    emotion="compassion",
                    intensite=0.7 if distress_level == "élevée" else 0.5 if distress_level == "modérée" else 0.3,
                )

            return {"success": True, **result}

        except Exception as e:
            self.logger.error(f"Erreur lors de la détection de détresse: {e}")
            return {"success": False, "reason": str(e)}

    # -------------------------------------------------------------------
    # Méthodes pour SharedEmotionReinforcer (Sprint 216)
    # -------------------------------------------------------------------

    def detecter_emotion_partagee(
        self,
        message: str,
        user_emotional_state: dict[str, Any] | None = None,
        vocal_features: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Détecte une émotion forte partagée dans un message utilisateur.
        Implémenté pour le Sprint 216.

        Args:
            message: Texte du message de l'utilisateur
            user_emotional_state: État émotionnel détecté précédemment (optionnel)
            vocal_features: Caractéristiques vocales extraites (optionnel)

        Returns:
            Dict: Résultats de la détection
        """
        if not hasattr(self, "shared_emotion_reinforcer") or not self.shared_emotion_reinforcer:
            # Tenter d'initialiser le renforçateur si non disponible
            if not self.initialiser_shared_emotion_reinforcer():
                return {"success": False, "reason": "SharedEmotionReinforcer non initialisé"}

        try:
            # Détecter l'émotion partagée
            result = self.shared_emotion_reinforcer.detect_shared_emotion(message, user_emotional_state, vocal_features)

            # Journaliser la détection d'émotion forte
            if (
                result.get("strong_emotion_shared", False)
                and hasattr(self, "journal_emotionnel")
                and self.journal_emotionnel is not None
            ):
                emotion = result.get("primary_emotion", "unknown")
                score = result.get("emotion_score", 0.5)

                self.journal_emotionnel.ajouter_entree(
                    pensee=f"L'utilisateur partage une émotion forte: {emotion} (intensité: {score:.2f})",
                    emotion="résonance",
                    intensite=min(0.9, score),
                )

            return {"success": True, **result}

        except Exception as e:
            self.logger.error(f"Erreur lors de la détection de l'émotion partagée: {e}")
            return {"success": False, "reason": str(e)}

    # -------------------------------------------------------------------
    # Méthodes pour InterpersonalRhythmSynchronizer (Sprint 217)
    # -------------------------------------------------------------------

    def enregistrer_interaction_rythmique(
        self, interaction_data: dict[str, Any], update_model: bool = True
    ) -> dict[str, Any]:
        """
        Enregistre une interaction et met à jour le modèle de rythme interpersonnel.
        Implémenté pour le Sprint 217.

        Args:
            interaction_data: Données d'interaction
            update_model: Si True, met à jour le modèle de rythme

        Returns:
            Dict: Résultats de l'analyse rythmique
        """
        if not hasattr(self, "interpersonal_rhythm_synchronizer") or not self.interpersonal_rhythm_synchronizer:
            # Tenter d'initialiser le synchronisateur si non disponible
            if not self.initialiser_interpersonal_rhythm_synchronizer():
                return {"success": False, "reason": "InterpersonalRhythmSynchronizer non initialisé"}

        try:
            # Enregistrer l'interaction rythmique
            result = self.interpersonal_rhythm_synchronizer.register_interaction(interaction_data, update_model)

            # Journaliser les changements significatifs de synchronisation
            if (
                result.get("success", False)
                and hasattr(self, "journal_emotionnel")
                and self.journal_emotionnel is not None
            ):
                sync_level = result.get("synchronization_level", 0.5)
                if sync_level > 0.8:
                    self.journal_emotionnel.ajouter_entree(
                        pensee="Notre synchronisation rythmique est excellente, l'interaction est très fluide",
                        emotion="harmonie",
                        intensite=0.8,
                    )
                elif sync_level < 0.3:
                    self.journal_emotionnel.ajouter_entree(
                        pensee="Notre synchronisation rythmique est faible, je devrais ajuster mon rythme",
                        emotion="ajustement",
                        intensite=0.6,
                    )

            return {"success": True, **result}

        except Exception as e:
            self.logger.error(f"Erreur lors de l'enregistrement de l'interaction rythmique: {e}")
            return {"success": False, "reason": str(e)}

    def ajuster_rythme_jeffrey(
        self, policy: str | None = None, custom_weights: dict[str, float] | None = None
    ) -> dict[str, Any]:
        """
        Ajuste le rythme de Jeffrey en fonction du modèle de rythme de l'utilisateur.
        Implémenté pour le Sprint 217.

        Args:
            policy: Nom de la politique d'adaptation à utiliser (optionnel)
            custom_weights: Poids personnalisés pour l'adaptation (optionnel)

        Returns:
            Dict: Résultats de l'ajustement
        """
        if not hasattr(self, "interpersonal_rhythm_synchronizer") or not self.interpersonal_rhythm_synchronizer:
            # Tenter d'initialiser le synchronisateur si non disponible
            if not self.initialiser_interpersonal_rhythm_synchronizer():
                return {"success": False, "reason": "InterpersonalRhythmSynchronizer non initialisé"}

        try:
            # Ajuster le rythme
            result = self.interpersonal_rhythm_synchronizer.adjust_jeffrey_rhythm(policy, custom_weights)

            # Journaliser les ajustements significatifs
            if (
                result.get("success", False)
                and result.get("is_significant_change", False)
                and hasattr(self, "journal_emotionnel")
                and self.journal_emotionnel is not None
            ):
                policy_applied = result.get("policy_applied", "unknown")

                self.journal_emotionnel.ajouter_entree(
                    pensee=f"J'ai ajusté mon rythme interpersonnel selon la politique '{policy_applied}' pour mieux m'aligner avec l'utilisateur",
                    emotion="adaptation",
                    intensite=0.7,
                )

            return {"success": True, **result}

        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajustement du rythme de Jeffrey: {e}")
            return {"success": False, "reason": str(e)}

    def obtenir_recommandations_rythme(self) -> dict[str, Any]:
        """
        Fournit des recommandations basées sur le modèle de rythme actuel.
        Implémenté pour le Sprint 217.

        Returns:
            Dict: Recommandations pour l'interaction
        """
        if not hasattr(self, "interpersonal_rhythm_synchronizer") or not self.interpersonal_rhythm_synchronizer:
            # Tenter d'initialiser le synchronisateur si non disponible
            if not self.initialiser_interpersonal_rhythm_synchronizer():
                return {"success": False, "reason": "InterpersonalRhythmSynchronizer non initialisé"}

        try:
            # Obtenir les recommandations de rythme
            result = self.interpersonal_rhythm_synchronizer.get_rhythm_recommendations()

            return {"success": True, **result}

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des recommandations de rythme: {e}")
            return {"success": False, "reason": str(e)}

    # -------------------------------------------------------------------
    # Méthodes pour EmotionalMetaMonitor (Sprint 219)
    # -------------------------------------------------------------------

    def analyser_patterns_emotionnels(self) -> dict[str, Any]:
        """
        Analyse les schémas émotionnels actuels pour détecter les tendances problématiques.
        Implémenté pour le Sprint 219.

        Returns:
            Dict: Résultats de l'analyse des schémas émotionnels
        """
        if not hasattr(self, "emotional_meta_monitor") or not self.emotional_meta_monitor:
            # Tenter d'initialiser le moniteur si non disponible
            if not self.initialiser_emotional_meta_monitor():
                return {"success": False, "reason": "EmotionalMetaMonitor non initialisé"}

        try:
            # Analyser les patterns émotionnels
            alerts = self.emotional_meta_monitor.analyze_patterns()

            # Journaliser les alertes critiques
            if alerts and hasattr(self, "journal_emotionnel") and self.journal_emotionnel is not None:
                for alert in alerts:
                    if alert.get("type") == "critical":
                        pattern = alert.get("pattern", "inconnu")
                        intensity = alert.get("intensity", 0.5)

                        self.journal_emotionnel.ajouter_entree(
                            pensee=f"J'ai détecté un schéma émotionnel problématique: {pattern} (intensité: {intensity:.2f})",
                            emotion="vigilance",
                            intensite=intensity,
                        )

            return {
                "success": True,
                "alerts": alerts,
                "current_intensities": self.emotional_meta_monitor.get_current_pattern_intensities(),
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse des patterns émotionnels: {e}")
            return {"success": False, "reason": str(e)}

    def ajouter_etat_emotionnel_pour_analyse(self, emotional_state: Any) -> dict[str, Any]:
        """
        Ajoute un état émotionnel à l'historique pour analyse.
        Implémenté pour le Sprint 219.

        Args:
            emotional_state: État émotionnel à ajouter

        Returns:
            Dict: Résultat de l'ajout
        """
        if not hasattr(self, "emotional_meta_monitor") or not self.emotional_meta_monitor:
            # Tenter d'initialiser le moniteur si non disponible
            if not self.initialiser_emotional_meta_monitor():
                return {"success": False, "reason": "EmotionalMetaMonitor non initialisé"}

        try:
            # Ajouter l'état émotionnel
            self.emotional_meta_monitor.add_emotional_state(emotional_state)

            return {"success": True}

        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajout de l'état émotionnel pour analyse: {e}")
            return {"success": False, "reason": str(e)}

    def obtenir_statistiques_patterns(self) -> dict[str, Any]:
        """
        Récupère des statistiques sur les schémas émotionnels.
        Implémenté pour le Sprint 219.

        Returns:
            Dict: Statistiques des schémas émotionnels
        """
        if not hasattr(self, "emotional_meta_monitor") or not self.emotional_meta_monitor:
            return {"success": False, "reason": "EmotionalMetaMonitor non initialisé"}

        try:
            # Récupérer les statistiques
            stats = self.emotional_meta_monitor.get_pattern_statistics()

            return {"success": True, "statistics": stats}

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des statistiques de patterns: {e}")
            return {"success": False, "reason": str(e)}

    def ajuster_seuils_detection_patterns(self, pattern_type: str, threshold: float) -> dict[str, Any]:
        """
        Ajuste le seuil de détection pour un type de schéma spécifique.
        Implémenté pour le Sprint 219.

        Args:
            pattern_type: Type de schéma à ajuster
            threshold: Nouveau seuil (0-1)

        Returns:
            Dict: Résultat de l'ajustement
        """
        if not hasattr(self, "emotional_meta_monitor") or not self.emotional_meta_monitor:
            return {"success": False, "reason": "EmotionalMetaMonitor non initialisé"}

        try:
            # Convertir le pattern_type en EmotionalPattern
            from core.emotions.emotional_meta_monitor import EmotionalPattern

            # Trouver le pattern correspondant
            pattern = None
            for p in EmotionalPattern:
                if p.value == pattern_type:
                    pattern = p
                    break

            if not pattern:
                return {"success": False, "reason": f"Pattern type '{pattern_type}' non reconnu"}

            # Ajuster le seuil
            self.emotional_meta_monitor.adjust_thresholds(pattern, threshold)

            return {"success": True, "pattern": pattern_type, "new_threshold": threshold}

        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajustement des seuils de détection: {e}")
            return {"success": False, "reason": str(e)}

    # -------------------------------------------------------------------
    # Méthodes pour obtenir les états actuels
    # -------------------------------------------------------------------

    def obtenir_role_actuel(self) -> dict[str, Any]:
        """
        Récupère des informations sur le rôle émotionnel actuel.
        Implémenté pour le Sprint 220.

        Returns:
            Dict: Information sur le rôle actuel
        """
        if not hasattr(self, "emotion_role_selector") or not self.emotion_role_selector:
            return {"success": False, "reason": "EmotionRoleSelector non initialisé", "has_active_role": False}

        try:
            # Récupérer le rôle actuel
            current_role = self.emotion_role_selector.get_current_role()

            if not current_role:
                return {"success": True, "has_active_role": False}

            # Récupérer les modificateurs
            modifiers = self.emotion_role_selector.get_role_modifiers()

            return {
                "success": True,
                "has_active_role": True,
                "role_id": current_role.role_id,
                "role_name": current_role.name,
                "description": current_role.description,
                "modifiers": modifiers,
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération du rôle actuel: {e}")
            return {"success": False, "reason": str(e), "has_active_role": False}

    def obtenir_statistiques_roles(self) -> dict[str, Any]:
        """
        Récupère des statistiques sur l'utilisation des rôles émotionnels.
        Implémenté pour le Sprint 220.

        Returns:
            Dict: Statistiques d'utilisation des rôles
        """
        if not hasattr(self, "emotion_role_selector") or not self.emotion_role_selector:
            return {"success": False, "reason": "EmotionRoleSelector non initialisé"}

        try:
            # Récupérer les statistiques
            stats = self.emotion_role_selector.get_role_usage_statistics()

            return {"success": True, "statistics": stats}

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération des statistiques de rôles: {e}")
            return {"success": False, "reason": str(e)}

    def obtenir_etat_verrouillage(self) -> dict[str, Any]:
        """
        Récupère l'état actuel du verrouillage émotionnel.
        Implémenté pour le Sprint 218.

        Returns:
            Dict: État actuel du verrouillage
        """
        if not hasattr(self, "emotional_stability_lock") or not self.emotional_stability_lock:
            return {"success": False, "reason": "EmotionalStabilityLock non initialisé", "is_locked": False}

        try:
            # Vérifier si l'état est verrouillé
            is_locked = self.emotional_stability_lock.is_locked()

            # Récupérer les détails du verrouillage
            locked_state = self.emotional_stability_lock.get_locked_state()

            return {"success": True, "is_locked": is_locked, "locked_state": locked_state}

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de l'état de verrouillage: {e}")
            return {"success": False, "reason": str(e), "is_locked": False}

    def get_aura_state(self) -> dict[str, Any]:
        """
        Récupère l'état actuel de l'aura émotionnelle.
        Implémenté pour le Sprint 223.

        Returns:
            Dict: État actuel de l'aura
        """
        if not hasattr(self, "aura_intensity_modulator") or not self.aura_intensity_modulator:
            return {"success": False, "reason": "AuraIntensityModulator non initialisé"}

        try:
            # Obtenir l'état actuel de l'aura
            aura_state = self.aura_intensity_modulator.get_current_aura_state()

            return {"success": True, "aura_state": aura_state}

        except Exception as e:
            self.logger.error(f"Erreur lors de la récupération de l'état de l'aura: {e}")
            return {"success": False, "reason": str(e)}
