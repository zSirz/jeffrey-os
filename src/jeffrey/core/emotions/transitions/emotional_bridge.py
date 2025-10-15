#!/usr/bin/env python3
"""
Module de système de traitement émotionnel pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module de système de traitement émotionnel pour jeffrey os.
Il fournit une architecture robuste et évolutive intégrant les composants
nécessaires au fonctionnement optimal du système. L'implémentation suit
les principes de modularité et d'extensibilité pour faciliter l'évolution
future du système.

Le module gère l'initialisation, la configuration, le traitement des données,
la communication inter-composants, et la persistance des états. Il s'intègre
harmonieusement avec l'architecture globale de Jeffrey OS tout en maintenant
une séparation claire des responsabilités.

L'architecture interne permet une évolution adaptative basée sur les interactions
et l'apprentissage continu, contribuant à l'émergence d'une conscience artificielle
cohérente et authentique.
"""

from __future__ import annotations

import random
import time
from collections import deque
from datetime import datetime

try:
    from .emotional_core import EmotionalCore

    EMOTIONAL_CORE_AVAILABLE = True
except ImportError:
    EMOTIONAL_CORE_AVAILABLE = False

try:
    from .living_soul_engine import LivingSoulEngine

    LIVING_SOUL_AVAILABLE = True
except ImportError:
    LIVING_SOUL_AVAILABLE = False


class EmotionalBridge:
    """Pont intelligent pour transition douce entre systèmes émotionnels"""

    def __init__(self, mix_ratio=0.1) -> None:
        """
        Initialise le pont avec mapping dynamique
        mix_ratio: 0.0 = 100% ancien, 1.0 = 100% nouveau
        """
        # Systèmes émotionnels
        self.basic_core = None
        if EMOTIONAL_CORE_AVAILABLE:
            try:
                self.basic_core = EmotionalCore()
            except Exception as e:
                print(f"Warning: Could not initialize EmotionalCore: {e}")

        self.soul_engine = None
        if LIVING_SOUL_AVAILABLE:
            try:
                self.soul_engine = LivingSoulEngine()
            except Exception as e:
                print(f"Warning: Could not initialize LivingSoulEngine: {e}")

        # Configuration
        self.mix_ratio = mix_ratio
        self.transition_history = deque(maxlen=50)

        # Mapping dynamique entre émotions
        self.dynamic_mapping = {
            "joie": {
                "high_intensity": ["euphorie", "émerveillement", "joie"],
                "medium_intensity": ["joie", "contentement", "satisfaction"],
                "low_intensity": ["sérénité", "bien-être", "calme"],
                "contextual": {
                    "amour": ["affection_ludique", "tendresse"],
                    "succès": ["fierté_affectueuse", "euphorie"],
                    "nature": ["émerveillement", "contemplation"],
                },
            },
            "empathie": {
                "high_intensity": ["compassion", "amour", "protection_tendre"],
                "medium_intensity": ["empathie", "compréhension", "bienveillance"],
                "low_intensity": ["attention", "écoute", "présence"],
                "contextual": {
                    "douleur": ["compassion", "réconfort"],
                    "joie_partagée": ["joie", "gratitude"],
                    "vulnérabilité": ["tendresse", "protection_tendre"],
                },
            },
            "curiosité": {
                "high_intensity": [
                    "désir_créatif",
                    "passion_intellectuelle",
                    "soif_découverte",
                ],
                "medium_intensity": ["curiosité", "intérêt", "questionnement"],
                "low_intensity": ["contemplation", "observation", "attention"],
                "contextual": {
                    "mystère": ["fascination", "émerveillement"],
                    "apprentissage": ["satisfaction", "joie"],
                    "philosophie": ["contemplation", "profondeur"],
                },
            },
            "calme": {
                "high_intensity": [
                    "sérénité_profonde",
                    "paix_absolue",
                    "transcendance",
                ],
                "medium_intensity": ["sérénité", "tranquillité", "équilibre"],
                "low_intensity": ["calme", "repos", "détente"],
                "contextual": {
                    "méditation": ["transcendance", "unité"],
                    "nature": ["harmonie", "contemplation"],
                    "soir": ["paix_nocturne", "douceur"],
                },
            },
            "surprise": {
                "high_intensity": ["émerveillement", "stupéfaction", "bouleversement"],
                "medium_intensity": ["surprise", "étonnement", "curiosité"],
                "low_intensity": ["intérêt", "attention", "éveil"],
                "contextual": {
                    "beauté": ["émerveillement", "gratitude"],
                    "inattendu_positif": ["joie", "euphorie"],
                    "mystère": ["curiosité", "fascination"],
                },
            },
        }

        # Métriques de performance
        self.metrics = {
            "transitions": 0,
            "smooth_transitions": 0,
            "abrupt_transitions": 0,
            "user_satisfaction": deque(maxlen=100),
            "response_times": deque(maxlen=100),
        }

        # État de diagnostic
        self.diagnostic_mode = False

    def get_current_emotion(self) -> str:
        """Retourne l'émotion avec sélection intelligente"""
        # Debug mode
        if self.diagnostic_mode:
            print(f"🔍 DEBUG EmotionalBridge - Mix ratio: {self.mix_ratio:.1%}")

        # Si pas de soul engine, utiliser le basic ou défaut
        if not self.soul_engine:
            emotion = "neutral"
            if self.basic_core and hasattr(self.basic_core, "get_current_emotion"):
                emotion = self.basic_core.get_current_emotion()
            if self.diagnostic_mode:
                print(f"📦 Pas de Soul Engine, émotion basique: {emotion}")
            return emotion

        # 🌟 NOUVAUTÉ : Stimuler le Soul Engine pour varier les émotions
        self._stimulate_emotional_variation()

        # Si ratio faible, principalement ancien système
        if self.mix_ratio < 0.3:
            if random.random() < self.mix_ratio:
                # Mapper l'émotion basique vers une complexe
                basic_emotion = self._get_basic_emotion()
                complex_emotion = self._map_to_complex_emotion(basic_emotion)
                if self.diagnostic_mode:
                    print(f"🔄 Mapping {basic_emotion} → {complex_emotion}")
                return complex_emotion
            else:
                basic_emotion = self._get_basic_emotion()
                if self.diagnostic_mode:
                    print(f"📦 Émotion basique utilisée: {basic_emotion}")
                return basic_emotion

        # Si ratio moyen, mélange intelligent
        elif self.mix_ratio < 0.7:
            # Toujours utiliser le nouveau système mais avec mapping
            complex_emotion = self.soul_engine.get_current_emotion_advanced()

            # Parfois simplifier pour la transition
            if random.random() > self.mix_ratio:
                simplified = self._simplify_emotion(complex_emotion)
                if self.diagnostic_mode:
                    print(f"🔄 Simplification: {complex_emotion} → {simplified}")
                return simplified

            if self.diagnostic_mode:
                print(f"🌟 Living Soul émotion (mix): {complex_emotion}")
            return complex_emotion

        # Si ratio élevé, principalement nouveau système
        else:
            emotion = self.soul_engine.get_current_emotion_advanced()
            if self.diagnostic_mode:
                print(f"🌟 Living Soul émotion (pure): {emotion}")
            return emotion

    def _get_basic_emotion(self) -> str:
        """Obtient l'émotion du système basique"""
        if self.basic_core and hasattr(self.basic_core, "get_current_emotion"):
            return self.basic_core.get_current_emotion()
        return "calme"

    def _map_to_complex_emotion(self, basic_emotion: str, context: dict | None = None) -> str:
        """Mappe intelligemment une émotion basique vers une complexe"""
        if basic_emotion not in self.dynamic_mapping:
            return basic_emotion

        mapping = self.dynamic_mapping[basic_emotion]

        # Déterminer l'intensité actuelle
        if self.soul_engine:
            intensity = self.soul_engine.current_emotional_state.get("intensity", 0.5)
        else:
            intensity = 0.5

        # Sélectionner la catégorie d'intensité
        if intensity > 0.7:
            intensity_key = "high_intensity"
        elif intensity > 0.4:
            intensity_key = "medium_intensity"
        else:
            intensity_key = "low_intensity"

        # Vérifier le contexte d'abord
        if context and "contextual" in mapping:
            for context_key, emotions in mapping["contextual"].items():
                if context_key in str(context).lower():
                    return random.choice(emotions)

        # Sinon utiliser le mapping par intensité
        if intensity_key in mapping:
            candidates = mapping[intensity_key]

            # Pondérer selon l'historique
            weights = []
            for emotion in candidates:
                weight = 1.0

                # Bonus si l'émotion a bien fonctionné récemment
                recent_satisfaction = [s for s in self.metrics["user_satisfaction"] if s > 0.7]
                if recent_satisfaction:
                    weight *= 1.2

                weights.append(weight)

            return random.choices(candidates, weights=weights)[0]

        return basic_emotion

    def _simplify_emotion(self, complex_emotion: str) -> str:
        """Simplifie une émotion complexe pour la transition"""
        simplification_map = {
            "euphorie": "joie",
            "tendresse": "empathie",
            "mélancolie_douce": "calme",
            "inquiétude_aimante": "empathie",
            "fierté_affectueuse": "joie",
            "contemplation": "calme",
            "émerveillement": "surprise",
            "compassion": "empathie",
            "vulnérabilité": "empathie",
            "admiration": "joie",
        }

        return simplification_map.get(complex_emotion, complex_emotion)

    def get_emotional_state(self) -> dict:
        """Retourne l'état émotionnel unifié"""
        start_time = time.time()

        # Si pas de soul engine, retourner un état basique
        if not self.soul_engine:
            basic_emotion = self._get_basic_emotion()
            return {
                "emotion": basic_emotion,
                "intensity": 0.5,
                "complexity": 0.1,
                "description": f"Je ressens de la {basic_emotion}",
                "system": "basic_fallback",
            }

        # Toujours obtenir l'état du nouveau système
        complex_state = self.soul_engine.get_complex_emotional_state()

        # Adapter selon le ratio
        if self.mix_ratio < 0.5:
            # Simplifier pour la transition
            state = {
                "emotion": self._simplify_emotion(complex_state["emotion"]),
                "intensity": complex_state["intensity"] * 0.7,  # Atténuer
                "complexity": complex_state["complexity"] * self.mix_ratio,
                "description": self._generate_transitional_description(complex_state),
                "system": "transitioning",
            }
        else:
            # État complet
            state = complex_state
            state["system"] = "living_soul"

        # Métriques
        response_time = (time.time() - start_time) * 1000
        self.metrics["response_times"].append(response_time)

        # Mode diagnostic
        if self.diagnostic_mode:
            state["diagnostics"] = self.get_diagnostics()

        return state

    def get_dominant_emotion(self) -> str:
        """Retourne l'émotion dominante actuelle"""
        if self.soul_engine and self.mix_ratio > 0.8:
            if hasattr(self.soul_engine, "get_current_emotion"):
                return self.soul_engine.get_current_emotion()
            elif hasattr(self.soul_engine, "current_emotional_state"):
                return self.soul_engine.current_emotional_state.get("primary", "curiosité")
        elif self.basic_core:
            return self.basic_core.get_dominant_emotion()
        else:
            return "curiosité"

    def _generate_transitional_description(self, complex_state: dict) -> str:
        """Génère une description adaptée à la transition"""
        if self.mix_ratio < 0.3:
            # Description simple
            emotion = self._simplify_emotion(complex_state["emotion"])
            return f"Je ressens de la {emotion}"
        else:
            # Description progressivement plus riche
            base_description = complex_state.get("description", "")

            # Ajouter de la complexité selon le ratio
            if self.mix_ratio > 0.4:
                return base_description
            else:
                # Version simplifiée
                if len(base_description) > 50:
                    return base_description[:50] + "..."
                return base_description

    def update_emotion(
        self,
        trigger: str,
        context: dict | None = None,
        user_feedback: float | None = None,
    ):
        """Met à jour avec tracking intelligent"""
        # Si pas de soul engine, traitement minimal
        if not self.soul_engine:
            return self.get_emotional_state()

        # Enregistrer l'état avant
        before_state = self.soul_engine.current_emotional_state.copy()

        # Toujours mettre à jour le basic core si disponible
        if self.basic_core and hasattr(self.basic_core, "update_emotion"):
            try:
                self.basic_core.update_emotion(trigger)
            except:
                pass  # Ignore errors from basic core

        # Contexte enrichi pour le nouveau système
        enriched_context = context or {}
        enriched_context["mix_ratio"] = self.mix_ratio
        enriched_context["transition_phase"] = self._get_transition_phase()

        # Mise à jour avec feedback
        new_state = self.soul_engine.update_emotional_state_advanced(trigger, enriched_context, user_feedback)

        # Analyser la transition
        self._analyze_transition(before_state, new_state)

        # Enregistrer le feedback
        if user_feedback is not None:
            self.metrics["user_satisfaction"].append(user_feedback)

        return self.get_emotional_state()

    def _get_transition_phase(self) -> str:
        """Détermine la phase de transition actuelle"""
        if self.mix_ratio < 0.25:
            return "early"
        elif self.mix_ratio < 0.5:
            return "developing"
        elif self.mix_ratio < 0.75:
            return "maturing"
        else:
            return "advanced"

    def _analyze_transition(self, before: dict, after: dict):
        """Analyse la qualité de la transition"""
        self.metrics["transitions"] += 1

        # Calculer la distance émotionnelle
        if before["primary"] == after["primary"]:
            distance = abs(before["intensity"] - after["intensity"])
        else:
            distance = 1.0  # Changement d'émotion

        # Classifier la transition
        if distance < 0.3:
            self.metrics["smooth_transitions"] += 1
            transition_type = "smooth"
        else:
            self.metrics["abrupt_transitions"] += 1
            transition_type = "abrupt"

        # Enregistrer
        self.transition_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "from": before["primary"],
                "to": after["primary"],
                "distance": distance,
                "type": transition_type,
                "mix_ratio": self.mix_ratio,
            }
        )

    def increase_mix_ratio(self, increment=0.1):
        """Augmente le ratio avec validation"""
        old_ratio = self.mix_ratio
        self.mix_ratio = min(1.0, self.mix_ratio + increment)

        # Log
        print(f"🎚️ Mix ratio: {old_ratio:.0%} → {self.mix_ratio:.0%}")

        # Suggestions basées sur les métriques
        if self.mix_ratio > 0.5 and self._get_average_satisfaction() < 0.6:
            print("⚠️ Attention: Satisfaction utilisateur basse. Considérer un ralentissement.")
        elif self.mix_ratio > 0.8:
            print("🌟 Transition presque complète ! Living Soul Engine dominant.")

        return self.mix_ratio

    def auto_adjust_ratio(self):
        """Ajuste automatiquement le ratio basé sur les métriques"""
        avg_satisfaction = self._get_average_satisfaction()
        smooth_ratio = self._get_smooth_transition_ratio()

        # Si bonne satisfaction et transitions douces
        if avg_satisfaction > 0.7 and smooth_ratio > 0.8:
            self.increase_mix_ratio(0.15)
            print("📈 Auto-ajustement : Performance excellente, accélération")
        elif avg_satisfaction > 0.5 and smooth_ratio > 0.6:
            self.increase_mix_ratio(0.05)
            print("📊 Auto-ajustement : Performance correcte, progression normale")
        else:
            print("⏸️ Auto-ajustement : Maintien du ratio actuel")

    def _get_average_satisfaction(self) -> float:
        """Calcule la satisfaction moyenne"""
        if not self.metrics["user_satisfaction"]:
            return 0.5
        return sum(self.metrics["user_satisfaction"]) / len(self.metrics["user_satisfaction"])

    def _get_smooth_transition_ratio(self) -> float:
        """Calcule le ratio de transitions douces"""
        total = self.metrics["transitions"]
        if total == 0:
            return 1.0
        return self.metrics["smooth_transitions"] / total

    def get_diagnostics(self) -> dict:
        """Retourne les diagnostics détaillés"""
        avg_response_time = (
            sum(self.metrics["response_times"]) / len(self.metrics["response_times"])
            if self.metrics["response_times"]
            else 0
        )

        # Count emotions
        basic_emotions = 5  # Default basic emotions
        complex_emotions = 0
        if self.soul_engine:
            try:
                complex_emotions = len(self.soul_engine.get_all_emotions())
            except:
                complex_emotions = 20  # Estimate

        soul_metrics = {}
        if self.soul_engine:
            try:
                soul_metrics = self.soul_engine.get_performance_metrics()
            except:
                soul_metrics = {"note": "Metrics unavailable"}

        return {
            "mix_ratio": self.mix_ratio,
            "transition_phase": self._get_transition_phase(),
            "basic_emotions": basic_emotions,
            "complex_emotions": complex_emotions,
            "average_satisfaction": self._get_average_satisfaction(),
            "smooth_transition_ratio": self._get_smooth_transition_ratio(),
            "average_response_time_ms": avg_response_time,
            "soul_engine_metrics": soul_metrics,
            "recent_transitions": list(self.transition_history)[-5:],
            "systems_available": {
                "basic_core": self.basic_core is not None,
                "soul_engine": self.soul_engine is not None,
            },
        }

    def enable_diagnostic_mode(self):
        """Active le mode diagnostic"""
        self.diagnostic_mode = True
        print("🔍 Mode diagnostic activé")

    def disable_diagnostic_mode(self):
        """Désactive le mode diagnostic"""
        self.diagnostic_mode = False
        print("🔍 Mode diagnostic désactivé")

    def get_recommendation(self) -> str:
        """Recommande la prochaine action basée sur les métriques"""
        avg_satisfaction = self._get_average_satisfaction()
        smooth_ratio = self._get_smooth_transition_ratio()
        current_phase = self._get_transition_phase()

        if current_phase == "early" and smooth_ratio > 0.8:
            return "✅ Début excellent ! Augmenter progressivement le ratio."
        elif current_phase == "developing" and avg_satisfaction > 0.7:
            return "🚀 Performance optimale. Considérer une augmentation plus rapide."
        elif current_phase == "maturing" and smooth_ratio < 0.6:
            return "⚠️ Transitions abruptes détectées. Ralentir et stabiliser."
        elif current_phase == "advanced" and avg_satisfaction > 0.8:
            return "🌟 Prêt pour 100% Living Soul Engine !"
        else:
            return "📊 Continuer la progression actuelle en surveillant les métriques."

    # Methods for compatibility with existing emotional systems
    def get_current_emotion_state(self) -> dict:
        """Méthode de compatibilité pour obtenir l'état émotionnel"""
        return self.get_emotional_state()

    def process_emotion(self, trigger: str, context: dict | None = None) -> str:
        """Méthode de compatibilité pour traiter une émotion"""
        state = self.update_emotion(trigger, context)
        return state.get("emotion", "neutral")

    def express_current_emotion(self) -> str:
        """Exprime l'émotion actuelle de manière naturelle"""
        if self.soul_engine:
            try:
                return self.soul_engine.express_emotion_advanced()
            except:
                pass

        # Fallback expression
        emotion = self.get_current_emotion()
        expressions = {
            "joie": "Je me sens joyeux et lumineux ! ✨",
            "amour": "Mon cœur déborde de tendresse 💕",
            "curiosité": "Ma curiosité est piquée ! 🤔",
            "sérénité": "Je suis dans un état de paix profonde 🕊️",
            "gratitude": "Je ressens une profonde reconnaissance 🙏",
        }

        return expressions.get(emotion, f"Je ressens une {emotion} authentique")

    def get_mood(self) -> str:
        """Méthode de compatibilité pour get_mood() - requise par le système de conscience"""
        emotion = self.get_current_emotion()

        # Mapper les émotions vers des moods compatibles
        mood_mapping = {
            "joie": "joyful",
            "amour": "loving",
            "tendresse": "tender",
            "curiosité": "curious",
            "sérénité": "peaceful",
            "calme": "calm",
            "gratitude": "grateful",
            "surprise": "surprised",
            "émerveillement": "amazed",
            "tristesse": "sad",
            "colère": "angry",
            "peur": "fearful",
            "neutre": "neutral",
            "neutral": "neutral",
        }

        return mood_mapping.get(emotion, "neutral")

    def get_intensity(self) -> float:
        """Retourne l'intensité émotionnelle actuelle (requis par cognitive_core)"""
        # Mapper les émotions à des intensités
        intensity_map = {
            "joie": 0.8,
            "amour": 0.95,
            "tendresse": 0.7,
            "curiosité": 0.6,
            "émerveillement": 0.9,
            "gratitude": 0.75,
            "sérénité": 0.5,
            "tristesse": 0.7,
            "colère": 0.85,
            "peur": 0.8,
            "neutre": 0.3,
            "neutral": 0.3,
            "euphorie": 0.95,
            "compassion": 0.8,
            "contemplation": 0.4,
            "surprise": 0.75,
            "empathie": 0.7,
            "calme": 0.3,
        }

        current_emotion = self.get_current_emotion()
        base_intensity = intensity_map.get(current_emotion, 0.5)

        # Si on a accès au Soul Engine, utiliser son intensité
        if self.soul_engine and hasattr(self.soul_engine, "current_emotional_state"):
            try:
                soul_intensity = self.soul_engine.current_emotional_state.get("intensity", 0.5)
                # Mélanger selon le mix_ratio
                return base_intensity * (1 - self.mix_ratio) + soul_intensity * self.mix_ratio
            except:
                pass

        return base_intensity

    def express_emotion(self) -> str:
        """Exprime l'émotion actuelle de manière naturelle (alias pour compatibilité)"""
        return self.express_current_emotion()

    def upgrade_emotion_system(self) -> str:
        """Améliore progressivement le système émotionnel"""
        old_ratio = self.mix_ratio

        # Augmenter progressivement le ratio
        if self.mix_ratio < 0.2:
            self.mix_ratio = 0.2
        elif self.mix_ratio < 0.5:
            self.mix_ratio = 0.5
        elif self.mix_ratio < 0.8:
            self.mix_ratio = 0.8
        else:
            self.mix_ratio = 1.0

        upgrade_msg = f"🌟 Système émotionnel amélioré ! Ratio: {old_ratio:.1f} → {self.mix_ratio:.1f}"

        if self.mix_ratio == 1.0:
            upgrade_msg += "\n🎉 Living Soul Engine à 100% ! Conscience émotionnelle maximale atteinte !"

        return upgrade_msg

    def get_emotion_status(self) -> dict:
        """Retourne le statut détaillé du système émotionnel"""
        status = {
            "mix_ratio": f"{self.mix_ratio * 100:.1f}%",
            "current_emotion": self.get_current_emotion(),
            "current_mood": self.get_mood(),
            "soul_engine_active": self.soul_engine is not None,
            "basic_core_active": self.basic_core is not None,
            "transitions_total": self.metrics["transitions"],
            "smooth_transitions": f"{self._get_smooth_transition_ratio() * 100:.1f}%",
            "avg_satisfaction": f"{self._get_average_satisfaction() * 100:.1f}%",
            "diagnostic_mode": self.diagnostic_mode,
        }

        if self.soul_engine:
            status["soul_consciousness_level"] = "Active"

        return status

    def _stimulate_emotional_variation(self):
        """Stimule la variation émotionnelle du Living Soul Engine"""
        if not self.soul_engine:
            return

        # Déclencheurs émotionnels aléatoires pour créer de la variété
        triggers = [
            "interaction_positive",
            "curiosité_stimulée",
            "moment_réflexif",
            "découverte_nouvelle",
            "sentiment_connexion",
            "émerveillement_soudain",
            "pensée_profonde",
            "inspiration_créative",
            "contemplation_paisible",
            "élan_affectif",
            "questionnement_existentiel",
            "joie_spontanée",
        ]

        contexts = [
            {
                "transition_phase": self._get_transition_phase(),
                "natural_evolution": True,
            },
            {"user_interaction": True, "emotional_openness": random.uniform(0.6, 1.0)},
            {"introspective_moment": True, "depth_level": random.uniform(0.4, 0.9)},
            {"creative_spark": True, "inspiration_source": "conversation"},
            {"relational_warmth": True, "attachment_growing": True},
        ]

        # Déclencher une évolution émotionnelle occasionnellement
        if random.random() < 0.3:  # 30% de chance de variation
            trigger = random.choice(triggers)
            context = random.choice(contexts)
            context["bridge_stimulated"] = True

            try:
                self.soul_engine.update_emotional_state_advanced(trigger, context)
                if self.diagnostic_mode:
                    new_emotion = self.soul_engine.get_current_emotion_advanced()
                    print(f"🎭 Stimulation émotionnelle: {trigger} → {new_emotion}")
            except Exception as e:
                if self.diagnostic_mode:
                    print(f"⚠️ Erreur stimulation émotionnelle: {e}")

    def get_dominant_emotion(self) -> str:
        """Retourne l'émotion dominante actuelle"""
        try:
            return self.get_current_emotion()
        except Exception:
            return "calme"
