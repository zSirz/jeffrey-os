"""
Emotion Orchestrator V2 - Production-ready avec fusion ML
Gère 16+ modules d'émotions avec fusion intelligente
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

import numpy as np
import yaml

from ..interfaces.protocols import EmotionModule, trimmed_mean
from ..loaders.secure_module_loader import SecureModuleLoader
from ..utils.async_helpers import LatencyBudget, asyncify

logger = logging.getLogger(__name__)


class EmotionCategory:
    """Catégorie de modules d'émotions avec config"""

    def __init__(self, name: str, config: dict):
        self.name = name
        self.priority = config.get("priority", 5)
        self.weight = config.get("weight", 1.0)
        self.timeout_ms = config.get("timeout_ms", 100)
        self.max_concurrency = config.get("max_concurrency", 2)
        self.enabled = config.get("enabled", True)
        self.lazy = config.get("lazy", False)
        self.modules = {}
        self.initialized = False
        self.semaphore = asyncio.Semaphore(self.max_concurrency)


class EmotionOrchestratorV2:
    """
    Orchestrateur d'émotions production-ready
    Avec fusion ML, budget, privacy
    """

    def __init__(self, loader: SecureModuleLoader, config_path: str = "config/federation.yaml"):
        self.loader = loader
        self.config = self._load_config(config_path)
        self.categories = {}
        self.initialized = False
        self.bus = None

        # État émotionnel global avec historique
        self.global_state = {
            "valence": 0.0,
            "arousal": 0.0,
            "dominance": 0.5,
            "mood": "neutral",
            "confidence": 0.5,
            "history": [],
        }

        # Fusion ML weights (appris)
        self.fusion_weights = defaultdict(lambda: 1.0)
        self.fusion_history = []

        # Stats
        self.stats = defaultdict(lambda: {"analyses": 0, "errors": 0, "timeouts": 0, "latency_ms_avg": 0})

    def _load_config(self, config_path: str) -> dict:
        """Charge la configuration"""
        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)
            return config.get("emotion_orchestrator", {})
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {"enabled": True, "budget_ms": 350}

    def _define_categories(self):
        """Définit les catégories depuis la config"""
        category_configs = self.config.get("categories", {})

        # Modules par catégorie
        category_modules = {
            "core": {
                "emotion_engine": "src.jeffrey.core.emotions.core.emotion_engine:EmotionEngine",
                "emotional_core": "src.jeffrey.core.emotions.core.jeffrey_emotional_core:JeffreyEmotionalCore",
                "ml_enhancer": "src.jeffrey.core.emotions.core.emotion_ml_enhancer:EmotionMLEnhancer",
                "prompt_detector": "src.jeffrey.core.emotions.core.emotion_prompt_detector:EmotionPromptDetector",
                "consciousness": "src.jeffrey.core.emotions.core.emotional_consciousness:EmotionalConsciousness",
            },
            "bridges": {
                "emotional_bridge": "src.jeffrey.core.emotions.emotional_bridge:EmotionalBridge",
                "contextual_bridge": "src.jeffrey.core.emotions.transitions.contextual_emotion_bridge:ContextualEmotionBridge",
                "transitions_bridge": "src.jeffrey.core.emotions.transitions.emotional_bridge:EmotionalBridge",
            },
            "profiles": {
                "profile_manager": "src.jeffrey.core.emotions.profiles.emotional_profile_manager:EmotionalProfileManager",
            },
            "temporal": {
                "seasons": "src.jeffrey.core.emotions.seasons.emotional_seasons:EmotionalSeasons",
                "surprises": "src.jeffrey.core.emotions.surprises.surprises_emotionnelles:SurprisesEmotionnelles",
            },
            "visual": {
                "aura_manager": "src.jeffrey.core.emotions.visuals.emotion_aura_manager:EmotionAuraManager",
                "emotional_display": "src.jeffrey.core.emotions.visuals.jeffrey_emotional_display:JeffreyEmotionalDisplay",
                "visual_emotions": "src.jeffrey.core.emotions.visuals.jeffrey_visual_emotions:JeffreyVisualEmotions",
            },
            "voice": {
                "voice_adapter": "src.jeffrey.services.voice.adapters.voice_emotion_adapter:VoiceEmotionAdapter",
                "voice_renderer": "src.jeffrey.services.voice.adapters.voice_emotion_renderer:VoiceEmotionRenderer",
            },
        }

        # Créer les catégories
        for cat_name, modules in category_modules.items():
            cat_config = category_configs.get(cat_name, {})

            if not cat_config.get("enabled", True):
                continue

            category = EmotionCategory(cat_name, cat_config)
            category.modules = modules  # Import paths
            self.categories[cat_name] = category

    async def initialize(self, bus=None, trace_id: str | None = None):
        """Initialise l'orchestrateur"""
        self.bus = bus

        logger.info(f"🎭 Initializing Emotion Orchestrator V2 (trace: {trace_id})")

        # Définir les catégories
        self._define_categories()

        # Charger les catégories non-lazy
        sorted_categories = sorted(self.categories.items(), key=lambda x: x[1].priority)

        for cat_name, category in sorted_categories:
            if not category.lazy:
                await self._load_category(cat_name, category)

        # Setup bus
        if self.bus:
            await self._setup_bus_subscriptions()

        self.initialized = True

        total_loaded = sum(len(c.modules) for c in self.categories.values() if c.initialized)
        logger.info(f"✅ Emotion Orchestrator initialized: {total_loaded} modules")

    async def _load_category(self, cat_name: str, category: EmotionCategory):
        """Charge une catégorie avec concurrence contrôlée"""
        if not category.enabled:
            return

        logger.info(f"Loading emotion category: {cat_name}")

        loaded = {}
        tasks = []

        for module_name, import_path in category.modules.items():
            task = self._load_module_safe(module_name, import_path, category.timeout_ms / 1000.0, category.semaphore)
            tasks.append((module_name, task))

        for module_name, task in tasks:
            try:
                instance = await task
                if instance:
                    loaded[module_name] = self._wrap_module(instance, module_name)
                    logger.info(f"  ✅ Loaded: {module_name}")
            except Exception as e:
                logger.warning(f"  ⚠️ Failed: {module_name}: {e}")
                self.stats[cat_name]["errors"] += 1

        category.modules = loaded
        category.initialized = len(loaded) > 0

    async def _load_module_safe(self, module_name: str, import_path: str, timeout: float, semaphore: asyncio.Semaphore):
        """Charge un module de manière safe"""
        async with semaphore:
            try:
                from ..loaders.secure_module_loader import ModuleSpec

                spec = ModuleSpec(name=module_name, import_path=import_path, enabled=True, critical=False)
                return await asyncify(self.loader._safe_import, spec, timeout=timeout)
            except Exception as e:
                logger.error(f"Error loading {module_name}: {e}")
                return None

    def _wrap_module(self, instance: Any, module_name: str) -> EmotionModule:
        """Wrap pour conformité interface"""
        if isinstance(instance, EmotionModule):
            return instance

        from .emotion_adapter import EmotionAdapter

        return EmotionAdapter(instance, module_name)

    async def _setup_bus_subscriptions(self):
        """Setup subscriptions"""
        if self.bus:
            await self.bus.subscribe("emotion.analyze", self._handle_analyze)
            await self.bus.subscribe("emotion.update_state", self._handle_update_state)

    # === ANALYSE HIÉRARCHIQUE ===

    async def analyze_fast(self, text: str) -> dict[str, Any]:
        """
        Analyse rapide (core uniquement)
        Budget: 80ms
        """
        budget = LatencyBudget(80)

        analyses = await self._analyze_categories(text, ["core"], budget)

        return self._fuse_analyses(analyses)

    async def analyze_deep(self, text: str) -> dict[str, Any]:
        """
        Analyse profonde (toutes catégories)
        Budget: 350ms
        """
        budget = LatencyBudget(self.config.get("budget_ms", 350))

        analyses = await self._analyze_categories(text, list(self.categories.keys()), budget)

        return self._fuse_analyses_ml(analyses)

    async def _analyze_categories(self, text: str, category_names: list[str], budget: LatencyBudget) -> list[dict]:
        """Analyse avec budget"""
        analyses = []

        for cat_name in category_names:
            if not budget.has_budget(20):
                break

            category = self.categories.get(cat_name)
            if not category or not category.initialized:
                continue

            cat_analyses = await self._analyze_category(
                category, text, min(category.timeout_ms, budget.remaining_ms()) / 1000.0
            )

            for analysis in cat_analyses:
                analysis["_category"] = cat_name
                analysis["_weight"] = category.weight
                analyses.append(analysis)

        return analyses

    async def _analyze_category(self, category: EmotionCategory, text: str, timeout: float) -> list[dict]:
        """Analyse une catégorie"""
        results = []

        tasks = []
        for module_name, instance in category.modules.items():
            if not instance:
                continue

            task = asyncify(instance.analyze, text, timeout=timeout)
            tasks.append((module_name, task))

        for module_name, task in tasks:
            try:
                result = await task
                if result:
                    if not isinstance(result, dict):
                        result = {"value": result}
                    result["_module"] = module_name
                    results.append(result)
            except Exception as e:
                logger.debug(f"Analysis failed in {module_name}: {e}")
                self.stats[category.name]["errors"] += 1

        return results

    def _fuse_analyses(self, analyses: list[dict]) -> dict[str, Any]:
        """Fusion simple pondérée"""
        if not analyses:
            return self.global_state.copy()

        # Calculer moyennes pondérées avec trimmed mean
        valences = []
        arousals = []
        dominances = []
        weights = []

        for analysis in analyses:
            weight = analysis.get("_weight", 1.0)
            weights.append(weight)

            if "valence" in analysis:
                valences.append(analysis["valence"] * weight)
            if "arousal" in analysis:
                arousals.append(analysis["arousal"] * weight)
            if "dominance" in analysis:
                dominances.append(analysis["dominance"] * weight)

        total_weight = sum(weights) if weights else 1.0

        result = {
            "valence": trimmed_mean(valences) / total_weight if valences else 0,
            "arousal": trimmed_mean(arousals) / total_weight if arousals else 0,
            "dominance": trimmed_mean(dominances) / total_weight if dominances else 0.5,
        }

        # Déterminer l'émotion
        result["mood"] = self._determine_emotion(result)
        result["confidence"] = min(len(analyses) / 5.0, 1.0)  # Plus d'analyses = plus de confiance
        result["sources"] = len(analyses)

        return result

    def _fuse_analyses_ml(self, analyses: list[dict]) -> dict[str, Any]:
        """
        Fusion ML dynamique avec apprentissage
        """
        if not analyses:
            return self.global_state.copy()

        # Utiliser les poids appris
        weighted_results = []

        for analysis in analyses:
            module = analysis.get("_module", "unknown")
            category = analysis.get("_category", "unknown")

            # Poids appris pour ce module
            learned_weight = self.fusion_weights[f"{category}.{module}"]
            base_weight = analysis.get("_weight", 1.0)

            final_weight = learned_weight * base_weight

            weighted_results.append({**analysis, "_final_weight": final_weight})

        # Fusion avec poids finaux
        result = self._fuse_weighted(weighted_results)

        # Apprendre des résultats (simpliste pour l'instant)
        self._update_fusion_weights(analyses, result)

        return result

    def _fuse_weighted(self, analyses: list[dict]) -> dict[str, Any]:
        """Fusion avec poids finaux"""
        if not analyses:
            return self.global_state.copy()

        total_weight = sum(a.get("_final_weight", 1.0) for a in analyses)

        result = {"valence": 0, "arousal": 0, "dominance": 0.5}

        for analysis in analyses:
            weight = analysis.get("_final_weight", 1.0) / total_weight

            result["valence"] += analysis.get("valence", 0) * weight
            result["arousal"] += analysis.get("arousal", 0) * weight
            result["dominance"] += analysis.get("dominance", 0.5) * weight

        result["mood"] = self._determine_emotion(result)
        result["confidence"] = min(total_weight, 1.0)
        result["sources"] = len(analyses)

        return result

    def _update_fusion_weights(self, analyses: list[dict], result: dict):
        """Met à jour les poids de fusion basé sur l'historique"""
        # Stocker pour apprentissage futur
        self.fusion_history.append({"timestamp": datetime.now().isoformat(), "analyses": analyses, "result": result})

        # Limiter l'historique
        if len(self.fusion_history) > 100:
            self.fusion_history = self.fusion_history[-100:]

        # Simple heuristique : booster les modules qui étaient proches du résultat final
        for analysis in analyses:
            module = analysis.get("_module", "unknown")
            category = analysis.get("_category", "unknown")
            key = f"{category}.{module}"

            # Distance au résultat final
            distance = abs(analysis.get("valence", 0) - result["valence"])

            # Ajuster le poids (plus proche = plus de poids)
            if distance < 0.2:
                self.fusion_weights[key] = min(self.fusion_weights[key] * 1.05, 3.0)
            else:
                self.fusion_weights[key] = max(self.fusion_weights[key] * 0.95, 0.3)

    def _determine_emotion(self, state: dict) -> str:
        """Détermine l'émotion selon PAD"""
        v = state.get("valence", 0)
        a = state.get("arousal", 0)
        d = state.get("dominance", 0.5)

        # Mapping PAD -> émotions
        if v > 0.3:
            if a > 0.3:
                return "excited" if d > 0.5 else "happy"
            else:
                return "content" if d > 0.5 else "relaxed"
        elif v < -0.3:
            if a > 0.3:
                return "angry" if d > 0.5 else "anxious"
            else:
                return "sad" if d < 0.5 else "bored"
        else:
            if a > 0.3:
                return "alert"
            else:
                return "neutral"

    async def update_global_state(self, new_state: dict):
        """Met à jour l'état global avec lissage"""
        alpha = 0.3  # Lissage

        # Mise à jour PAD
        for dim in ["valence", "arousal", "dominance"]:
            if dim in new_state:
                old = self.global_state[dim]
                new = new_state[dim]
                self.global_state[dim] = alpha * new + (1 - alpha) * old

        # Mood
        self.global_state["mood"] = self._determine_emotion(self.global_state)
        self.global_state["confidence"] = new_state.get("confidence", 0.5)

        # Historique
        self.global_state["history"].append(
            {
                "timestamp": datetime.now().isoformat(),
                **{k: v for k, v in self.global_state.items() if k != "history"},
            }
        )

        # Limiter
        if len(self.global_state["history"]) > 100:
            self.global_state["history"] = self.global_state["history"][-100:]

        # Publier si bus
        if self.bus:
            await self.bus.publish({"type": "emotion.state.changed", "data": self.global_state})

    def get_alignment_score(self, user_state: dict) -> float:
        """
        Calcule le score d'alignement avec l'utilisateur
        Pour mesurer la symbiose
        """
        if not user_state:
            return 0.5

        # Cosine similarity simplifié sur PAD
        system = np.array(
            [
                self.global_state["valence"],
                self.global_state["arousal"],
                self.global_state["dominance"],
            ]
        )

        user = np.array(
            [
                user_state.get("valence", 0),
                user_state.get("arousal", 0),
                user_state.get("dominance", 0.5),
            ]
        )

        # Cosine similarity
        dot_product = np.dot(system, user)
        norm_system = np.linalg.norm(system)
        norm_user = np.linalg.norm(user)

        if norm_system == 0 or norm_user == 0:
            return 0.5

        similarity = dot_product / (norm_system * norm_user)

        # Normaliser entre 0 et 1
        return (similarity + 1) / 2

    async def _handle_analyze(self, envelope):
        """Handle analyze request"""
        text = envelope.get("data", {}).get("text", "")
        result = await self.analyze_deep(text)
        return result

    async def _handle_update_state(self, envelope):
        """Handle state update request"""
        new_state = envelope.get("data", {})
        await self.update_global_state(new_state)
        return {"status": "updated"}

    def get_stats(self) -> dict[str, Any]:
        """Retourne les stats détaillées"""
        cat_stats = {}

        for name, category in self.categories.items():
            cat_stats[name] = {
                "initialized": category.initialized,
                "modules": len(category.modules),
                "priority": category.priority,
                "weight": category.weight,
                **self.stats.get(name, {}),
            }

        return {
            "categories": cat_stats,
            "current_state": self.global_state,
            "fusion_method": self.config.get("fusion_method", "trimmed_mean"),
            "alignment_score": 0.5,  # Par défaut
            "initialized": self.initialized,
        }
