#!/usr/bin/env python3

"""
Module contenant la logique émotionnelle interne de Jeffrey.
Ce module gère les aspects de verrouillage, recalibrage, et transitions
des états émotionnels.
"""

import logging
from typing import Any


class EmotionalLogic:
    """
    Classe regroupant la logique émotionnelle interne de Jeffrey.
    """

    def __init__(self):
        """
        Initialise le logger pour la logique émotionnelle.
        """
        self.logger = logging.getLogger(__name__)

    # -------------------------------------------------------------------
    # Méthodes d'analyse émotionnelle
    # -------------------------------------------------------------------

    def analyser_tendance_emotionnelle(self, history: list[dict[str, Any]], num_points: int = 10) -> dict[str, Any]:
        """
        Analyse la tendance émotionnelle sur une période donnée.

        Args:
            history: Historique des états émotionnels
            num_points: Nombre de points d'historique à considérer

        Returns:
            Dict: Résultats de l'analyse de tendance
        """
        if not history:
            return {"success": False, "reason": "Aucun historique émotionnel disponible"}

        try:
            # Limiter l'historique aux derniers num_points points
            recent_history = history[-num_points:] if len(history) > num_points else history

            # Exemple simple d'analyse de tendance
            # Dans une implémentation réelle, ce serait plus complexe
            emotions = {}
            intensities = []

            for point in recent_history:
                emotion = point.get("emotion", "unknown")
                intensity = point.get("intensity", 0.5)

                if emotion in emotions:
                    emotions[emotion] += 1
                else:
                    emotions[emotion] = 1

                intensities.append(intensity)

            # Calculer l'émotion dominante
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0] if emotions else "unknown"

            # Calculer la tendance d'intensité (croissante, décroissante, stable)
            intensity_trend = "stable"
            if len(intensities) > 1:
                first_half = sum(intensities[: len(intensities) // 2]) / (len(intensities) // 2)
                second_half = sum(intensities[len(intensities) // 2 :]) / (len(intensities) - len(intensities) // 2)

                if second_half - first_half > 0.1:
                    intensity_trend = "croissante"
                elif first_half - second_half > 0.1:
                    intensity_trend = "décroissante"

            # Calculer la variation émotionnelle
            emotional_variation = len(emotions.keys()) / len(recent_history) if recent_history else 0

            return {
                "success": True,
                "dominant_emotion": dominant_emotion,
                "emotional_counts": emotions,
                "intensity_trend": intensity_trend,
                "avg_intensity": sum(intensities) / len(intensities) if intensities else 0,
                "emotional_variation": emotional_variation,
                "points_analyzed": len(recent_history),
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de l'analyse de la tendance émotionnelle: {e}")
            return {"success": False, "reason": str(e)}

    def evaluer_cyclicite_emotionnelle(self, history: list[dict[str, Any]], window_size: int = 24) -> dict[str, Any]:
        """
        Évalue la présence de cycles émotionnels récurrents.

        Args:
            history: Historique des états émotionnels
            window_size: Taille de la fenêtre d'analyse pour les cycles

        Returns:
            Dict: Résultats de l'analyse des cycles
        """
        if len(history) < window_size * 2:
            return {
                "success": False,
                "reason": f"Historique insuffisant pour l'analyse des cycles (minimum {window_size * 2} points nécessaires)",
            }

        try:
            # Regrouper les émotions par segments temporels
            segments = []
            for i in range(0, len(history), window_size):
                segment = history[i : i + window_size]
                if len(segment) == window_size:  # Seulement les segments complets
                    emotion_counts = {}
                    for point in segment:
                        emotion = point.get("emotion", "unknown")
                        if emotion in emotion_counts:
                            emotion_counts[emotion] += 1
                        else:
                            emotion_counts[emotion] = 1

                    segments.append(
                        {
                            "dominant_emotion": max(emotion_counts.items(), key=lambda x: x[1])[0],
                            "counts": emotion_counts,
                        }
                    )

            # Recherche de patterns répétitifs
            cycles_detected = False
            cycle_length = 0

            if len(segments) >= 4:  # Minimum de segments pour détecter un cycle
                # Recherche des cycles de différentes longueurs
                for cycle_len in range(1, len(segments) // 2):
                    matches = 0
                    total_comparisons = 0

                    for i in range(len(segments) - cycle_len):
                        if i + cycle_len < len(segments):
                            if segments[i]["dominant_emotion"] == segments[i + cycle_len]["dominant_emotion"]:
                                matches += 1
                            total_comparisons += 1

                    match_ratio = matches / total_comparisons if total_comparisons > 0 else 0

                    if match_ratio > 0.7:  # Seuil arbitraire pour considérer un cycle
                        cycles_detected = True
                        cycle_length = cycle_len
                        break

            return {
                "success": True,
                "cycles_detected": cycles_detected,
                "cycle_length": cycle_length * window_size if cycles_detected else 0,
                "segments_analyzed": len(segments),
                "window_size": window_size,
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation de la cyclicité émotionnelle: {e}")
            return {"success": False, "reason": str(e)}

    def detecter_disparite_emotionnelle(
        self, expressed_state: dict[str, Any], internal_state: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Détecte les disparités entre l'état émotionnel exprimé et l'état interne.

        Args:
            expressed_state: État émotionnel exprimé
            internal_state: État émotionnel interne

        Returns:
            Dict: Résultats de l'analyse de disparité
        """
        if not expressed_state or not internal_state:
            return {"success": False, "reason": "États émotionnels incomplets"}

        try:
            # Extraire les émotions principales
            expressed_emotion = expressed_state.get("emotion", "unknown")
            internal_emotion = internal_state.get("emotion", "unknown")

            # Extraire les intensités
            expressed_intensity = expressed_state.get("intensity", 0.5)
            internal_intensity = internal_state.get("intensity", 0.5)

            # Calculer la disparité émotionnelle
            emotion_mismatch = expressed_emotion != internal_emotion

            # Calculer la disparité d'intensité
            intensity_diff = abs(expressed_intensity - internal_intensity)

            # Évaluer le niveau global de disparité
            disparity_level = "faible"
            if emotion_mismatch and intensity_diff > 0.3:
                disparity_level = "élevée"
            elif emotion_mismatch or intensity_diff > 0.3:
                disparity_level = "modérée"

            return {
                "success": True,
                "emotion_mismatch": emotion_mismatch,
                "expressed_emotion": expressed_emotion,
                "internal_emotion": internal_emotion,
                "intensity_difference": intensity_diff,
                "disparity_level": disparity_level,
                "potential_conflict": disparity_level == "élevée",
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de la détection de disparité émotionnelle: {e}")
            return {"success": False, "reason": str(e)}

    # -------------------------------------------------------------------
    # Méthodes de transition et d'adaptation émotionnelle
    # -------------------------------------------------------------------

    def planifier_transition_emotionnelle(
        self, from_emotion: str, to_emotion: str, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Planifie une transition émotionnelle naturelle.

        Args:
            from_emotion: Émotion de départ
            to_emotion: Émotion d'arrivée
            context: Contexte de la transition (optionnel)

        Returns:
            Dict: Plan de transition émotionnelle
        """
        try:
            # Définir des émotions intermédiaires naturelles entre certaines paires d'émotions
            # En pratique, cela pourrait être défini dans une base de connaissances
            transition_map = {
                ("joie", "tristesse"): ["surprise", "calme", "nostalgie"],
                ("colère", "calme"): ["frustration", "réflexion"],
                ("peur", "confiance"): ["vigilance", "soulagement"],
                ("dégoût", "intérêt"): ["surprise", "curiosité"],
                ("ennui", "enthousiasme"): ["intérêt", "curiosité"],
            }

            # Chercher les transitions directes
            transition_steps = []
            key = (from_emotion, to_emotion)
            reversed_key = (to_emotion, from_emotion)

            if key in transition_map:
                transition_steps = transition_map[key]
            elif reversed_key in transition_map:
                transition_steps = list(reversed(transition_map[reversed_key]))

            # Si aucune transition spécifique n'est définie, utiliser une transition générique
            if not transition_steps:
                if from_emotion != to_emotion:
                    transition_steps = ["neutre"]  # Passer par un état neutre

            # Estimer la durée de chaque étape (en secondes)
            step_durations = [1.5] * len(transition_steps)

            # Ajuster les durées en fonction du contexte si disponible
            if context and "urgency" in context:
                urgency = context["urgency"]
                if urgency == "high":
                    step_durations = [d * 0.6 for d in step_durations]  # Transition plus rapide
                elif urgency == "low":
                    step_durations = [d * 1.5 for d in step_durations]  # Transition plus lente

            return {
                "success": True,
                "from_emotion": from_emotion,
                "to_emotion": to_emotion,
                "transition_steps": transition_steps,
                "step_durations": step_durations,
                "total_duration": sum(step_durations),
                "direct_transition": len(transition_steps) == 0,
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de la planification de la transition émotionnelle: {e}")
            return {"success": False, "reason": str(e)}

    def recalibrer_intensite_emotionnelle(
        self, current_intensity: float, target_intensity: float, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Recalibre l'intensité émotionnelle de manière adaptative.

        Args:
            current_intensity: Intensité émotionnelle actuelle (0-1)
            target_intensity: Intensité émotionnelle cible (0-1)
            context: Contexte du recalibrage (optionnel)

        Returns:
            Dict: Plan de recalibrage d'intensité
        """
        try:
            # Valider les intensités
            current_intensity = max(0.0, min(1.0, current_intensity))
            target_intensity = max(0.0, min(1.0, target_intensity))

            # Calculer la différence d'intensité
            intensity_diff = target_intensity - current_intensity
            abs_diff = abs(intensity_diff)

            # Déterminer le nombre d'étapes en fonction de la différence
            if abs_diff <= 0.1:
                num_steps = 1  # Petit changement, faire directement
            elif abs_diff <= 0.3:
                num_steps = 2  # Changement modéré, faire en deux étapes
            else:
                num_steps = 3  # Grand changement, faire en trois étapes

            # Créer les étapes intermédiaires
            intensity_steps = []
            for i in range(1, num_steps + 1):
                step_intensity = current_intensity + (intensity_diff * i / num_steps)
                intensity_steps.append(round(step_intensity, 2))

            # Estimer la durée de chaque étape (en secondes)
            step_durations = [abs_diff * 2] * num_steps

            # Ajuster en fonction du contexte
            if context:
                if context.get("emotional_state") == "unstable":
                    # En état instable, ralentir les transitions
                    step_durations = [d * 1.5 for d in step_durations]
                elif context.get("urgency") == "high":
                    # En urgence, accélérer les transitions
                    step_durations = [d * 0.7 for d in step_durations]

            return {
                "success": True,
                "current_intensity": current_intensity,
                "target_intensity": target_intensity,
                "intensity_difference": intensity_diff,
                "num_steps": num_steps,
                "intensity_steps": intensity_steps,
                "step_durations": step_durations,
                "total_duration": sum(step_durations),
            }

        except Exception as e:
            self.logger.error(f"Erreur lors du recalibrage de l'intensité émotionnelle: {e}")
            return {"success": False, "reason": str(e)}

    def harmoniser_emotions_conflictuelles(self, emotions: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Harmonise des émotions conflictuelles pour créer un état cohérent.

        Args:
            emotions: Liste des émotions actives avec leur intensité

        Returns:
            Dict: État émotionnel harmonisé
        """
        if not emotions:
            return {"success": False, "reason": "Aucune émotion fournie"}

        try:
            # Grouper les émotions par catégorie
            emotion_categories = {
                "joie": ["joie", "bonheur", "enthousiasme", "amusement"],
                "colère": ["colère", "irritation", "frustration"],
                "peur": ["peur", "anxiété", "appréhension"],
                "tristesse": ["tristesse", "mélancolie", "déception"],
                "surprise": ["surprise", "étonnement"],
                "dégoût": ["dégoût", "aversion"],
                "confiance": ["confiance", "sécurité", "sérénité"],
                "anticipation": ["anticipation", "intérêt", "curiosité"],
            }

            # Mapper les émotions à leurs catégories
            categorized_emotions = {}
            for emotion in emotions:
                emotion_name = emotion.get("emotion", "unknown")
                intensity = emotion.get("intensity", 0.5)

                category = "autre"
                for cat, emotions_list in emotion_categories.items():
                    if emotion_name in emotions_list:
                        category = cat
                        break

                if category in categorized_emotions:
                    categorized_emotions[category].append((emotion_name, intensity))
                else:
                    categorized_emotions[category] = [(emotion_name, intensity)]

            # Identifier les catégories conflictuelles
            conflicting_pairs = [("joie", "tristesse"), ("colère", "sérénité"), ("peur", "confiance")]

            # Résoudre les conflits
            resolved_emotions = {}
            for category, emotions_list in categorized_emotions.items():
                # Pour chaque catégorie, calculer l'intensité moyenne
                if emotions_list:
                    avg_intensity = sum(intensity for _, intensity in emotions_list) / len(emotions_list)
                    # Prendre l'émotion la plus intense de la catégorie
                    dominant_emotion = max(emotions_list, key=lambda x: x[1])[0]
                    resolved_emotions[category] = {"emotion": dominant_emotion, "intensity": avg_intensity}

            # Traiter les conflits
            for cat1, cat2 in conflicting_pairs:
                if cat1 in resolved_emotions and cat2 in resolved_emotions:
                    # Conflit détecté, garder la plus forte et réduire l'autre
                    if resolved_emotions[cat1]["intensity"] > resolved_emotions[cat2]["intensity"]:
                        resolved_emotions[cat2]["intensity"] *= 0.5
                    else:
                        resolved_emotions[cat1]["intensity"] *= 0.5

            # Déterminer l'émotion dominante finale
            final_emotions = sorted(
                [(cat, data["emotion"], data["intensity"]) for cat, data in resolved_emotions.items()],
                key=lambda x: x[2],
                reverse=True,
            )

            # Émotion principale et secondaire
            primary_emotion = {"emotion": "neutre", "intensity": 0.5}
            secondary_emotion = {"emotion": "neutre", "intensity": 0.3}

            if final_emotions:
                primary_emotion = {
                    "emotion": final_emotions[0][1],
                    "intensity": final_emotions[0][2],
                    "category": final_emotions[0][0],
                }

                if len(final_emotions) > 1:
                    secondary_emotion = {
                        "emotion": final_emotions[1][1],
                        "intensity": final_emotions[1][2],
                        "category": final_emotions[1][0],
                    }

            return {
                "success": True,
                "primary_emotion": primary_emotion,
                "secondary_emotion": secondary_emotion,
                "has_conflicts": any(
                    cat1 in resolved_emotions and cat2 in resolved_emotions for cat1, cat2 in conflicting_pairs
                ),
                "num_emotions_processed": len(emotions),
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de l'harmonisation des émotions conflictuelles: {e}")
            return {"success": False, "reason": str(e)}

    # -------------------------------------------------------------------
    # Méthodes de stabilité émotionnelle
    # -------------------------------------------------------------------

    def evaluer_stabilite_emotionnelle(self, history: list[dict[str, Any]], window_size: int = 10) -> dict[str, Any]:
        """
        Évalue la stabilité émotionnelle sur une période récente.

        Args:
            history: Historique des états émotionnels
            window_size: Taille de la fenêtre d'analyse

        Returns:
            Dict: Résultats de l'évaluation de stabilité
        """
        if len(history) < window_size:
            return {
                "success": False,
                "reason": f"Historique insuffisant pour l'analyse (minimum {window_size} points nécessaires)",
            }

        try:
            # Analyser les derniers états émotionnels
            recent_history = history[-window_size:]

            # Calculer les métriques de stabilité
            # 1. Variabilité des émotions
            emotions = [state.get("emotion", "unknown") for state in recent_history]
            unique_emotions = set(emotions)
            emotion_variability = len(unique_emotions) / window_size

            # 2. Variabilité d'intensité
            intensities = [state.get("intensity", 0.5) for state in recent_history]
            intensity_std_dev = (
                sum((i - sum(intensities) / len(intensities)) ** 2 for i in intensities) / len(intensities)
            ) ** 0.5

            # 3. Fréquence des transitions
            transitions = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i - 1])
            transition_rate = transitions / (window_size - 1) if window_size > 1 else 0

            # Calcul du score de stabilité (0-1, où 1 est très stable)
            stability_score = 1.0 - (
                (emotion_variability * 0.4) + (min(intensity_std_dev * 2, 1.0) * 0.3) + (transition_rate * 0.3)
            )

            # Interprétation du score
            stability_level = "élevée"
            if stability_score < 0.4:
                stability_level = "faible"
            elif stability_score < 0.7:
                stability_level = "modérée"

            return {
                "success": True,
                "stability_score": stability_score,
                "stability_level": stability_level,
                "emotion_variability": emotion_variability,
                "intensity_std_dev": intensity_std_dev,
                "transition_rate": transition_rate,
                "unique_emotions": list(unique_emotions),
                "window_size": window_size,
                "needs_stabilization": stability_score < 0.5,
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de l'évaluation de la stabilité émotionnelle: {e}")
            return {"success": False, "reason": str(e)}

    def definir_seuil_verouillage_emotionnel(
        self, base_threshold: float = 0.4, context: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """
        Définit dynamiquement le seuil de verrouillage émotionnel en fonction du contexte.

        Args:
            base_threshold: Seuil de base (0-1)
            context: Contexte pour l'ajustement du seuil

        Returns:
            Dict: Seuil ajusté et facteurs d'ajustement
        """
        try:
            # Valider le seuil de base
            base_threshold = max(0.1, min(0.9, base_threshold))

            # Facteurs d'ajustement par défaut
            adjustment_factors = {
                "user_fatigue": 0.0,  # Ajustement pour la fatigue de l'utilisateur
                "topic_sensitivity": 0.0,  # Ajustement pour les sujets sensibles
                "conversation_pace": 0.0,  # Ajustement pour le rythme de conversation
                "interaction_length": 0.0,  # Ajustement pour la durée d'interaction
            }

            # Appliquer les ajustements en fonction du contexte
            if context:
                # Fatigue de l'utilisateur
                if "user_fatigue" in context:
                    fatigue_level = context["user_fatigue"]
                    if fatigue_level == "high":
                        adjustment_factors["user_fatigue"] = -0.1  # Plus facile à verrouiller
                    elif fatigue_level == "low":
                        adjustment_factors["user_fatigue"] = 0.05

                # Sensibilité du sujet
                if "topic_sensitivity" in context:
                    sensitivity = context["topic_sensitivity"]
                    if sensitivity == "high":
                        adjustment_factors["topic_sensitivity"] = -0.15  # Plus facile à verrouiller
                    elif sensitivity == "medium":
                        adjustment_factors["topic_sensitivity"] = -0.05

                # Rythme de conversation
                if "conversation_pace" in context:
                    pace = context["conversation_pace"]
                    if pace == "fast":
                        adjustment_factors["conversation_pace"] = 0.05  # Plus difficile à verrouiller
                    elif pace == "slow":
                        adjustment_factors["conversation_pace"] = -0.05

                # Durée d'interaction
                if "interaction_duration" in context:
                    duration = context["interaction_duration"]
                    if duration > 30:  # Minutes
                        adjustment_factors["interaction_length"] = -0.1  # Plus facile à verrouiller
                    elif duration > 15:
                        adjustment_factors["interaction_length"] = -0.05

            # Calculer le seuil ajusté
            total_adjustment = sum(adjustment_factors.values())
            adjusted_threshold = base_threshold + total_adjustment

            # Limiter au range 0.1-0.9
            adjusted_threshold = max(0.1, min(0.9, adjusted_threshold))

            return {
                "success": True,
                "base_threshold": base_threshold,
                "adjusted_threshold": adjusted_threshold,
                "adjustment_factors": adjustment_factors,
                "total_adjustment": total_adjustment,
            }

        except Exception as e:
            self.logger.error(f"Erreur lors de la définition du seuil de verrouillage émotionnel: {e}")
            return {"success": False, "reason": str(e)}
