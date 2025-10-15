#!/usr/bin/env python3

"""
Module contenant les interfaces émotionnelles de Jeffrey.
Ces méthodes sont les points d'entrée publiques exposées aux autres composants.
"""

import logging
from typing import Any


class EmotionalInterfaces:
    """
    Classe regroupant les interfaces émotionnelles de Jeffrey.
    """

    def __init__(self):
        """
        Initialise le logger pour les interfaces émotionnelles.
        """
        self.logger = logging.getLogger(__name__)

    # -------------------------------------------------------------------
    # Méthodes pour EmotionalTrustModeler (Sprint 213)
    # -------------------------------------------------------------------

    def ajuster_intimite_reponse(
        self, response: str, original_intimacy_level: float, context: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        """
        Ajuste le niveau d'intimité d'une réponse en fonction du niveau de confiance.
        Implémenté pour le Sprint 213.

        Args:
            response: Réponse originale
            original_intimacy_level: Niveau d'intimité original (0-1)
            context: Contexte de la conversation (optionnel)

        Returns:
            Tuple[str, Dict]: Réponse ajustée et métadonnées
        """
        if not hasattr(self, "emotional_trust_modeler") or not self.emotional_trust_modeler:
            # Tenter d'initialiser le modélisateur si non disponible
            if not self.initialiser_emotional_trust_modeler():
                return response, {"success": False, "reason": "EmotionalTrustModeler non initialisé"}

        try:
            # Ajuster le niveau d'intimité
            adjusted_response, result = self.emotional_trust_modeler.adjust_response_intimacy(
                response, original_intimacy_level, context
            )

            # Journaliser l'ajustement si significatif
            if (
                result.get("intimacy_adjusted", False)
                and hasattr(self, "journal_emotionnel")
                and self.journal_emotionnel is not None
            ):
                adjustment_ratio = result.get("adjustment_ratio", 1.0)
                if adjustment_ratio < 0.7:  # Ajustement significatif
                    self.journal_emotionnel.ajouter_entree(
                        pensee="J'ai ajusté le niveau d'intimité de ma réponse pour respecter le niveau de confiance actuel",
                        emotion="prudence",
                        intensite=0.6,
                    )

            return adjusted_response, {"success": True, **result}

        except Exception as e:
            self.logger.error(f"Erreur lors de l'ajustement de l'intimité de la réponse: {e}")
            return response, {"success": False, "reason": str(e)}

    # -------------------------------------------------------------------
    # Méthodes pour ReactiveCompassionEngine (Sprint 214)
    # -------------------------------------------------------------------

    def generer_reponse_compassionnelle(
        self, original_response: str, distress_info: dict[str, Any], context: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        """
        Génère une réponse avec compassion adaptée à la détresse détectée.
        Implémenté pour le Sprint 214.

        Args:
            original_response: Réponse originale à modifier
            distress_info: Informations sur la détresse détectée
            context: Contexte de la conversation (optionnel)

        Returns:
            Tuple[str, Dict]: Réponse modifiée et métadonnées
        """
        if not hasattr(self, "reactive_compassion_engine") or not self.reactive_compassion_engine:
            # Tenter d'initialiser le moteur si non disponible
            if not self.initialiser_reactive_compassion_engine():
                return original_response, {"success": False, "reason": "ReactiveCompassionEngine non initialisé"}

        try:
            # Générer la réponse compassionnelle
            compassionate_response, result = self.reactive_compassion_engine.generate_compassionate_response(
                original_response, distress_info, context
            )

            # Journaliser la génération de réponse compassionnelle
            if (
                result.get("modified", False)
                and hasattr(self, "journal_emotionnel")
                and self.journal_emotionnel is not None
            ):
                approach = result.get("approach", "unknown")
                self.journal_emotionnel.ajouter_entree(
                    pensee=f"J'ai adapté ma réponse avec une approche de '{approach}' pour apporter du soutien émotionnel",
                    emotion="soutien",
                    intensite=0.7,
                )

            return compassionate_response, {"success": True, **result}

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de la réponse compassionnelle: {e}")
            return original_response, {"success": False, "reason": str(e)}

    # -------------------------------------------------------------------
    # Méthodes pour SilentAnchorEmbedder (Sprint 215)
    # -------------------------------------------------------------------

    def creer_ancrage_silencieux(
        self,
        context: dict[str, Any],
        anchor_type: str | None = None,
        emotional_tone: str | None = None,
        intensity: float = 0.7,
        duration: float = 1.5,
    ) -> dict[str, Any]:
        """
        Crée un nouvel ancrage émotionnel silencieux.
        Implémenté pour le Sprint 215.

        Args:
            context: Contexte de l'interaction
            anchor_type: Type d'ancrage (si None, sélection automatique)
            emotional_tone: Tonalité émotionnelle (si None, sélection automatique)
            intensity: Intensité de l'ancrage (0-1)
            duration: Durée de l'ancrage en secondes

        Returns:
            Dict: Ancrage silencieux créé
        """
        if not hasattr(self, "silent_anchor_embedder") or not self.silent_anchor_embedder:
            # Tenter d'initialiser l'embedder si non disponible
            if not self.initialiser_silent_anchor_embedder():
                return {"success": False, "reason": "SilentAnchorEmbedder non initialisé"}

        try:
            # Créer l'ancrage silencieux
            result = self.silent_anchor_embedder.create_anchor(
                context, anchor_type, emotional_tone, intensity, duration
            )

            # Journaliser la création d'ancrage
            if (
                result.get("success", False)
                and hasattr(self, "journal_emotionnel")
                and self.journal_emotionnel is not None
            ):
                anchor = result.get("anchor", {})
                anchor_type = anchor.get("type", "unknown")
                emotional_tone = anchor.get("emotional_tone", "unknown")

                self.journal_emotionnel.ajouter_entree(
                    pensee=f"J'ai créé un ancrage silencieux de type '{anchor_type}' avec une tonalité '{emotional_tone}'",
                    emotion=emotional_tone
                    if emotional_tone in ["warmth", "connection", "presence", "contemplation"]
                    else "calme",
                    intensite=intensity,
                )

            return {"success": True, **result}

        except Exception as e:
            self.logger.error(f"Erreur lors de la création de l'ancrage silencieux: {e}")
            return {"success": False, "reason": str(e)}

    def suggerer_ancrage_silencieux(
        self, context: dict[str, Any], user_emotional_state: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Suggère un ancrage silencieux adapté au contexte actuel.
        Implémenté pour le Sprint 215.

        Args:
            context: Contexte de l'interaction
            user_emotional_state: État émotionnel de l'utilisateur (optionnel)

        Returns:
            Dict: Suggestion d'ancrage
        """
        if not hasattr(self, "silent_anchor_embedder") or not self.silent_anchor_embedder:
            # Tenter d'initialiser l'embedder si non disponible
            if not self.initialiser_silent_anchor_embedder():
                return {"success": False, "reason": "SilentAnchorEmbedder non initialisé"}

        try:
            # Suggérer un ancrage silencieux
            result = self.silent_anchor_embedder.suggest_anchor(context, user_emotional_state)

            return {"success": True, **result}

        except Exception as e:
            self.logger.error(f"Erreur lors de la suggestion d'ancrage silencieux: {e}")
            return {"success": False, "reason": str(e)}

    # -------------------------------------------------------------------
    # Méthodes pour SharedEmotionReinforcer (Sprint 216)
    # -------------------------------------------------------------------

    def generer_reponse_renforcement(
        self, original_response: str, emotion_info: dict[str, Any], context: dict[str, Any] | None = None
    ) -> tuple[str, dict[str, Any]]:
        """
        Génère une réponse qui renforce l'émotion partagée.
        Implémenté pour le Sprint 216.

        Args:
            original_response: Réponse originale à modifier
            emotion_info: Informations sur l'émotion partagée
            context: Contexte de la conversation (optionnel)

        Returns:
            Tuple[str, Dict]: Réponse modifiée et métadonnées
        """
        if not hasattr(self, "shared_emotion_reinforcer") or not self.shared_emotion_reinforcer:
            # Tenter d'initialiser le renforçateur si non disponible
            if not self.initialiser_shared_emotion_reinforcer():
                return original_response, {"success": False, "reason": "SharedEmotionReinforcer non initialisé"}

        try:
            # Générer la réponse de renforcement
            reinforced_response, result = self.shared_emotion_reinforcer.generate_reinforcement_response(
                original_response, emotion_info, context
            )

            # Journaliser la génération de réponse de renforcement
            if (
                result.get("modified", False)
                and hasattr(self, "journal_emotionnel")
                and self.journal_emotionnel is not None
            ):
                strategy = result.get("strategy", "unknown")
                emotion = result.get("primary_emotion", "unknown")

                self.journal_emotionnel.ajouter_entree(
                    pensee=f"J'ai renforcé l'émotion '{emotion}' partagée avec une stratégie de '{strategy}'",
                    emotion="connexion",
                    intensite=0.7,
                )

            return reinforced_response, {"success": True, **result}

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de la réponse de renforcement: {e}")
            return original_response, {"success": False, "reason": str(e)}

    # -------------------------------------------------------------------
    # Méthodes pour InterpersonalRhythmSynchronizer (Sprint 217)
    # -------------------------------------------------------------------

    def appliquer_parametres_rythme(self, response_params: dict[str, Any]) -> dict[str, Any]:
        """
        Applique les paramètres du rythme actuel à une réponse.
        Implémenté pour le Sprint 217.

        Args:
            response_params: Paramètres de réponse à ajuster

        Returns:
            Dict: Paramètres de réponse ajustés
        """
        if not hasattr(self, "interpersonal_rhythm_synchronizer") or not self.interpersonal_rhythm_synchronizer:
            # Tenter d'initialiser le synchronisateur si non disponible
            if not self.initialiser_interpersonal_rhythm_synchronizer():
                return {
                    "success": False,
                    "reason": "InterpersonalRhythmSynchronizer non initialisé",
                    "params": response_params,
                }

        try:
            # Appliquer les paramètres de rythme
            result = self.interpersonal_rhythm_synchronizer.apply_rhythm_parameters(response_params)

            return {"success": True, **result}

        except Exception as e:
            self.logger.error(f"Erreur lors de l'application des paramètres de rythme: {e}")
            return {"success": False, "reason": str(e), "params": response_params}

    # -------------------------------------------------------------------
    # Méthodes pour EmotionalStabilityLock (Sprint 218)
    # -------------------------------------------------------------------

    def verrouiller_etat_emotionnel(
        self,
        emotion: str | None = None,
        intensite: float | None = None,
        duree: int | None = None,
        modules_a_suspendre: list[str] | None = None,
        raison: str = "manuel",
    ) -> dict[str, Any]:
        """
        Verrouille l'état émotionnel de Jeffrey pour assurer la stabilité.
        Implémenté pour le Sprint 218.

        Args:
            emotion: L'émotion à verrouiller (si None, utilise l'émotion actuelle)
            intensite: L'intensité de l'émotion verrouillée (si None, réduit l'intensité actuelle)
            duree: Durée du verrouillage en secondes
            modules_a_suspendre: Liste des modules à suspendre pendant le verrouillage
            raison: Raison du verrouillage ("manuel", "auto", "dérive", "surcharge")

        Returns:
            Dict: Résultat du verrouillage
        """
        if not hasattr(self, "emotional_stability_lock") or not self.emotional_stability_lock:
            # Tenter d'initialiser le verrou si non disponible
            if not self.initialiser_emotional_stability_lock():
                return {"success": False, "reason": "EmotionalStabilityLock non initialisé"}

        try:
            # Verrouiller l'état émotionnel
            result = self.emotional_stability_lock.lock_emotion(
                emotion=emotion,
                intensity=intensite,
                duration=duree,
                modules_to_suspend=modules_a_suspendre,
                reason=raison,
            )

            # Journaliser le verrouillage
            if result and hasattr(self, "journal_emotionnel") and self.journal_emotionnel is not None:
                emotion_str = emotion or "actuelle"
                self.journal_emotionnel.ajouter_entree(
                    pensee=f"J'ai verrouillé mon état émotionnel sur '{emotion_str}' pour assurer la stabilité",
                    emotion="stabilité",
                    intensite=0.6,
                )

            return {
                "success": result,
                "emotion": emotion,
                "intensite": intensite,
                "duree": duree,
                "raison": raison,
                "modules_suspendus": modules_a_suspendre or [],
            }

        except Exception as e:
            self.logger.error(f"Erreur lors du verrouillage de l'état émotionnel: {e}")
            return {"success": False, "reason": str(e)}

    def deverrouiller_etat_emotionnel(self, force: bool = False) -> dict[str, Any]:
        """
        Déverrouille l'état émotionnel de Jeffrey.
        Implémenté pour le Sprint 218.

        Args:
            force: Si True, force le déverrouillage même si la durée n'est pas écoulée

        Returns:
            Dict: Résultat du déverrouillage
        """
        if not hasattr(self, "emotional_stability_lock") or not self.emotional_stability_lock:
            return {"success": False, "reason": "EmotionalStabilityLock non initialisé"}

        try:
            # Déverrouiller l'état émotionnel
            result = self.emotional_stability_lock.unlock(force=force)

            # Journaliser le déverrouillage
            if result and hasattr(self, "journal_emotionnel") and self.journal_emotionnel is not None:
                self.journal_emotionnel.ajouter_entree(
                    pensee="J'ai déverrouillé mon état émotionnel, permettant à nouveau une expression naturelle",
                    emotion="libération",
                    intensite=0.5,
                )

            return {"success": result, "force": force}

        except Exception as e:
            self.logger.error(f"Erreur lors du déverrouillage de l'état émotionnel: {e}")
            return {"success": False, "reason": str(e)}

    def appliquer_etat_verrouille(self, response: str) -> tuple[str, dict[str, Any]]:
        """
        Applique l'état verrouillé à une réponse si nécessaire.
        Implémenté pour le Sprint 218.

        Args:
            response: Réponse originale

        Returns:
            Tuple[str, Dict]: Réponse modifiée et informations de verrouillage
        """
        if not hasattr(self, "emotional_stability_lock") or not self.emotional_stability_lock:
            return response, {"success": False, "reason": "EmotionalStabilityLock non initialisé", "modified": False}

        try:
            # Vérifier si l'état est verrouillé
            if not self.emotional_stability_lock.is_locked():
                return response, {"success": True, "modified": False, "is_locked": False}

            # Récupérer l'état verrouillé
            locked_state = self.emotional_stability_lock.get_locked_state()

            # Si l'état est verrouillé, appliquer des modifications subtiles à la réponse
            # (par exemple, ajuster le ton, ajouter un marqueur, etc.)
            if locked_state:
                emotion = locked_state.get("emotion", "neutre")
                intensity = locked_state.get("intensity", 0.5)

                # Exemples de modifications selon l'émotion verrouillée
                if emotion in ["calme", "neutre", "serein"]:
                    modified_response = response.replace("!", ".")  # Réduire l'excitation
                    modified_response = modified_response.replace("?!", "?")
                else:
                    # Pour d'autres émotions, conserver la réponse mais ajouter un marqueur subtil
                    modified_response = response

                return modified_response, {
                    "success": True,
                    "modified": True,
                    "is_locked": True,
                    "emotion": emotion,
                    "intensity": intensity,
                    "remaining_seconds": locked_state.get("remaining_seconds", 0),
                }

            return response, {
                "success": True,
                "modified": False,
                "is_locked": True,  # Verrouillé mais sans état spécifique
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de l'application de l'état verrouillé: {e}")
            return response, {"success": False, "reason": str(e), "modified": False}

    # -------------------------------------------------------------------
    # Méthodes pour EmotionRoleSelector (Sprint 220)
    # -------------------------------------------------------------------

    def selectionner_role_emotionnel(
        self, role_id: str | None = None, user_id: str | None = None, context: str | None = None
    ) -> dict[str, Any]:
        """
        Sélectionne un rôle émotionnel pour Jeffrey.
        Implémenté pour le Sprint 220.

        Args:
            role_id: Identifiant du rôle à sélectionner (si None, sélection automatique)
            user_id: Identifiant de l'utilisateur (pour les préférences)
            context: Contexte de conversation (pour la sélection automatique)

        Returns:
            Dict: Résultat de la sélection de rôle
        """
        if not hasattr(self, "emotion_role_selector") or not self.emotion_role_selector:
            # Tenter d'initialiser le sélecteur si non disponible
            if not self.initialiser_emotion_role_selector():
                return {"success": False, "reason": "EmotionRoleSelector non initialisé"}

        try:
            if role_id:
                # Sélection manuelle d'un rôle
                result = self.emotion_role_selector.select_role_manually(role_id)

                if result:
                    self.current_emotional_role = self.emotion_role_selector.get_current_role()

                    # Journaliser le changement de rôle
                    if hasattr(self, "journal_emotionnel") and self.journal_emotionnel is not None:
                        self.journal_emotionnel.ajouter_entree(
                            pensee=f"J'ai adopté le rôle émotionnel '{self.current_emotional_role.name}'",
                            emotion="adaptation",
                            intensite=0.6,
                        )

                    return {
                        "success": True,
                        "role_id": role_id,
                        "role_name": self.current_emotional_role.name,
                        "selection_type": "manual",
                    }
                else:
                    return {
                        "success": False,
                        "reason": f"Impossible de sélectionner le rôle '{role_id}'",
                        "selection_type": "manual",
                    }
            else:
                # Sélection automatique d'un rôle
                emotional_state = None
                if hasattr(self, "emotional_engine") and hasattr(self.emotional_engine, "get_current_emotional_state"):
                    emotional_state = self.emotional_engine.get_current_emotional_state()

                selected_role_id = self.emotion_role_selector.select_role_automatically(
                    user_id=user_id, emotional_state=emotional_state, context=context
                )

                self.current_emotional_role = self.emotion_role_selector.get_current_role()

                # Journaliser le changement de rôle automatique
                if hasattr(self, "journal_emotionnel") and self.journal_emotionnel is not None:
                    self.journal_emotionnel.ajouter_entree(
                        pensee=f"J'ai automatiquement adopté le rôle émotionnel '{self.current_emotional_role.name}' en fonction du contexte",
                        emotion="intuition",
                        intensite=0.5,
                    )

                return {
                    "success": True,
                    "role_id": selected_role_id,
                    "role_name": self.current_emotional_role.name,
                    "selection_type": "automatic",
                }

        except Exception as e:
            self.logger.error(f"Erreur lors de la sélection du rôle émotionnel: {e}")
            return {"success": False, "reason": str(e)}

    def appliquer_role_a_reponse(self, response: str, strength: float = 1.0) -> tuple[str, dict[str, Any]]:
        """
        Adapte une réponse selon le rôle émotionnel actuel.
        Implémenté pour le Sprint 220.

        Args:
            response: Réponse originale
            strength: Intensité de l'adaptation (0-1)

        Returns:
            Tuple[str, Dict]: Réponse adaptée et métadonnées
        """
        if not hasattr(self, "emotion_role_selector") or not self.emotion_role_selector:
            return response, {"success": False, "reason": "EmotionRoleSelector non initialisé", "modified": False}

        try:
            # Récupérer le rôle actuel
            current_role = self.emotion_role_selector.get_current_role()

            if not current_role:
                return response, {"success": True, "modified": False, "reason": "Aucun rôle actif"}

            # Appliquer le rôle à la réponse
            adapted_response = self.emotion_role_selector.apply_role_to_response(response, strength)

            return adapted_response, {
                "success": True,
                "modified": adapted_response != response,
                "role_id": current_role.role_id,
                "role_name": current_role.name,
                "strength": strength,
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de l'application du rôle à la réponse: {e}")
            return response, {"success": False, "reason": str(e), "modified": False}

    # -------------------------------------------------------------------
    # Méthodes pour InternalStateNarrator (Sprint 221)
    # -------------------------------------------------------------------

    def activer_narration_etat_interne(self, active: bool = True) -> dict[str, Any]:
        """
        Active ou désactive la narration des états internes.
        Implémenté pour le Sprint 221.

        Args:
            active: True pour activer, False pour désactiver

        Returns:
            Dict: Résultat de l'activation/désactivation
        """
        if not hasattr(self, "internal_state_narrator") or not self.internal_state_narrator:
            # Tenter d'initialiser le narrateur si non disponible
            if not self.initialiser_internal_state_narrator():
                return {"success": False, "reason": "InternalStateNarrator non initialisé"}

        try:
            # Activer ou désactiver
            if active:
                self.internal_state_narrator.enable()
            else:
                self.internal_state_narrator.disable()

            return {"success": True, "enabled": active}

        except Exception as e:
            self.logger.error(f"Erreur lors de l'activation/désactivation de la narration d'état interne: {e}")
            return {"success": False, "reason": str(e)}

    def generer_verbalisation_etat(
        self, emotional_state: Any = None, context: str | None = None, user_id: str | None = None
    ) -> dict[str, Any]:
        """
        Génère une verbalisation de l'état interne actuel de Jeffrey.
        Implémenté pour le Sprint 221.

        Args:
            emotional_state: État émotionnel à verbaliser (si None, utilise l'état actuel)
            context: Contexte de la conversation
            user_id: Identifiant de l'utilisateur

        Returns:
            Dict: Verbalisation générée
        """
        if not hasattr(self, "internal_state_narrator") or not self.internal_state_narrator:
            return {"success": False, "reason": "InternalStateNarrator non initialisé"}

        try:
            # Si aucun état émotionnel n'est fourni, utiliser l'état actuel
            if (
                emotional_state is None
                and hasattr(self, "emotional_engine")
                and hasattr(self.emotional_engine, "get_current_emotional_state")
            ):
                emotional_state = self.emotional_engine.get_current_emotional_state()

            # Générer la verbalisation
            verbalization = self.internal_state_narrator.generate_verbalization(
                emotional_state=emotional_state, context=context, user_id=user_id
            )

            return {"success": True, "verbalization": verbalization, "generated": verbalization is not None}

        except Exception as e:
            self.logger.error(f"Erreur lors de la génération de la verbalisation d'état: {e}")
            return {"success": False, "reason": str(e)}

    def integrer_verbalisation_dans_reponse(
        self, response: str, emotional_state: Any = None, context: str | None = None, user_id: str | None = None
    ) -> tuple[str, dict[str, Any]]:
        """
        Intègre une verbalisation d'état interne dans une réponse existante.
        Implémenté pour le Sprint 221.

        Args:
            response: Réponse originale
            emotional_state: État émotionnel à verbaliser (si None, utilise l'état actuel)
            context: Contexte de la conversation
            user_id: Identifiant de l'utilisateur

        Returns:
            Tuple[str, Dict]: Réponse avec verbalisation intégrée et métadonnées
        """
        if not hasattr(self, "internal_state_narrator") or not self.internal_state_narrator:
            return response, {"success": False, "reason": "InternalStateNarrator non initialisé", "modified": False}

        try:
            # Si aucun état émotionnel n'est fourni, utiliser l'état actuel
            if (
                emotional_state is None
                and hasattr(self, "emotional_engine")
                and hasattr(self.emotional_engine, "get_current_emotional_state")
            ):
                emotional_state = self.emotional_engine.get_current_emotional_state()

            # Intégrer la verbalisation
            integrated_response = self.internal_state_narrator.integrate_verbalization(
                response=response, emotional_state=emotional_state, context=context, user_id=user_id
            )

            # Déterminer si une modification a été apportée
            modified = integrated_response != response

            return integrated_response, {"success": True, "modified": modified}

        except Exception as e:
            self.logger.error(f"Erreur lors de l'intégration de la verbalisation: {e}")
            return response, {"success": False, "reason": str(e), "modified": False}

    def definir_preference_utilisateur_narration(
        self, user_id: str, enable: bool = True, frequency: float | None = None
    ) -> dict[str, Any]:
        """
        Définit les préférences de narration pour un utilisateur.
        Implémenté pour le Sprint 221.

        Args:
            user_id: Identifiant de l'utilisateur
            enable: Activer/désactiver la narration pour cet utilisateur
            frequency: Fréquence relative des verbalisations (0 à 1)

        Returns:
            Dict: Résultat de la définition des préférences
        """
        if not hasattr(self, "internal_state_narrator") or not self.internal_state_narrator:
            return {"success": False, "reason": "InternalStateNarrator non initialisé"}

        try:
            # Importer le type ExpressionStyle

            # Définir les préférences
            self.internal_state_narrator.set_user_preference(user_id=user_id, enable=enable, frequency=frequency)

            return {"success": True, "user_id": user_id, "enabled": enable, "frequency": frequency}

        except Exception as e:
            self.logger.error(f"Erreur lors de la définition des préférences utilisateur pour la narration: {e}")
            return {"success": False, "reason": str(e)}

    # -------------------------------------------------------------------
    # Méthodes d'aide à la mémoire émotionnelle (pour l'intégration)
    # -------------------------------------------------------------------

    def add_emotional_memory(
        self, memory_entry: dict[str, Any], intensity: float = 0.5, category: str = "general"
    ) -> bool:
        """
        Ajoute une entrée à la mémoire émotionnelle.
        Cette méthode est utilisée par les modules qui ont besoin d'accéder à la mémoire.

        Args:
            memory_entry: Entrée à ajouter à la mémoire
            intensity: Intensité émotionnelle de l'entrée (0-1)
            category: Catégorie de l'entrée

        Returns:
            bool: True si l'ajout a réussi, False sinon
        """
        # Si nous avons un gestionnaire de mémoire avec cette méthode, utiliser celui-là
        if (
            hasattr(self, "memory_manager")
            and self.memory_manager
            and hasattr(self.memory_manager, "add_emotional_memory")
        ):
            return self.memory_manager.add_emotional_memory(memory_entry, intensity, category)

        # Sinon, si nous avons une méthode locale, l'utiliser
        elif hasattr(self, "ajouter_souvenir_emotionnel"):
            try:
                return self.ajouter_souvenir_emotionnel(memory_entry, intensity=intensity, categorie=category)
            except Exception as e:
                self.logger.error(f"Erreur lors de l'ajout d'un souvenir émotionnel: {e}")
                return False

        # Si aucune méthode n'est disponible, journaliser et retourner False
        else:
            self.logger.warning("Aucune méthode disponible pour ajouter un souvenir émotionnel")
            return False
