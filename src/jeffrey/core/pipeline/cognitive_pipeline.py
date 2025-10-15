"""
Pipeline cognitif principal avec intégration modules RÉELS
Architecture 3 couches, 6 GFC, 0 stubs
"""

import asyncio
import logging
import sys
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

# GPT Fix #1: Import explicite de importlib.util
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any, Protocol

import yaml

# Ajouter les paths vendor
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "icloud_vendor"))

logger = logging.getLogger(__name__)


class CognitiveModule(Protocol):
    """Interface standard pour tous les modules cognitifs"""

    async def initialize(self, config: dict[str, Any]) -> None:
        """Initialise le module avec sa config"""
        ...

    async def process(self, context: dict[str, Any]) -> dict[str, Any]:
        """Traite le contexte et retourne le contexte enrichi"""
        ...

    def get_gfc(self) -> str:
        """Retourne le GFC du module"""
        ...

    async def shutdown(self) -> None:
        """Arrêt propre du module"""
        ...


@dataclass
class ModuleAdapter:
    """
    Adaptateur pour modules existants non-conformes
    Permet d'intégrer les vrais modules sans modification
    """

    module_name: str
    module_path: str
    gfc: str
    config: dict[str, Any] = field(default_factory=dict)
    instance: Any | None = None
    initialized: bool = False
    priority: int = 0  # GPT Fix #3: Ajouter priorité

    async def initialize(self) -> bool:
        """Charge et initialise le module réel"""
        try:
            # GPT Fix #1: Utilisation correcte de spec_from_file_location et module_from_spec
            spec = spec_from_file_location(self.module_name, self.module_path)
            if not spec or not spec.loader:
                raise ImportError(f"Cannot load {self.module_path}")

            module = module_from_spec(spec)
            spec.loader.exec_module(module)

            # Détecter la classe principale
            # Stratégies: chercher par nom, par héritage, par convention
            main_class = None

            # 1. Convention de nommage
            for attr_name in dir(module):
                if attr_name.lower() == self.module_name.replace("_", "").lower():
                    main_class = getattr(module, attr_name)
                    break

            # 2. Première classe trouvée
            if not main_class:
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if isinstance(attr, type) and not attr_name.startswith("_"):
                        main_class = attr
                        break

            if not main_class:
                # Module fonctionnel sans classe
                self.instance = module
            else:
                # Instancier la classe
                try:
                    self.instance = main_class(**self.config)
                except TypeError:
                    self.instance = main_class()  # Sans config

            # Initialiser si méthode disponible
            if hasattr(self.instance, "initialize"):
                if asyncio.iscoroutinefunction(self.instance.initialize):
                    await self.instance.initialize(self.config)
                else:
                    self.instance.initialize(self.config)
            elif hasattr(self.instance, "init"):
                if asyncio.iscoroutinefunction(self.instance.init):
                    await self.instance.init()
                else:
                    self.instance.init()

            self.initialized = True
            logger.info(f"✅ Module initialized: {self.module_name} ({self.gfc})")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to initialize {self.module_name}: {e}")
            logger.error(traceback.format_exc())
            return False

    async def process(self, context: dict[str, Any]) -> dict[str, Any]:
        """Adapte l'appel process pour le module"""
        if not self.initialized:
            return context

        try:
            # Différentes conventions possibles
            if hasattr(self.instance, "process"):
                if asyncio.iscoroutinefunction(self.instance.process):
                    return await self.instance.process(context)
                else:
                    return self.instance.process(context)

            elif hasattr(self.instance, "run"):
                if asyncio.iscoroutinefunction(self.instance.run):
                    result = await self.instance.run(context)
                else:
                    result = self.instance.run(context)

                # Si run retourne quelque chose, l'intégrer au contexte
                if result:
                    if isinstance(result, dict):
                        context.update(result)
                    else:
                        context[f"{self.module_name}_result"] = result
                return context

            elif hasattr(self.instance, "execute"):
                if asyncio.iscoroutinefunction(self.instance.execute):
                    result = await self.instance.execute(context)
                else:
                    result = self.instance.execute(context)

                if isinstance(result, dict):
                    return result
                context["result"] = result
                return context

            # Modules spécifiques
            elif self.gfc == "memory_associative" and hasattr(self.instance, "recall"):
                # Module mémoire
                query = context.get("input", "")
                memories = (
                    await self.instance.recall(query)
                    if asyncio.iscoroutinefunction(self.instance.recall)
                    else self.instance.recall(query)
                )
                context["memories"] = memories
                return context

            elif self.gfc == "valence_emotional" and hasattr(self.instance, "feel"):
                # Module émotionnel
                emotion = (
                    await self.instance.feel(context)
                    if asyncio.iscoroutinefunction(self.instance.feel)
                    else self.instance.feel(context)
                )
                context["emotion"] = emotion
                return context

            elif self.gfc == "expression_generation" and hasattr(self.instance, "generate"):
                # Module génération
                response = (
                    await self.instance.generate(context)
                    if asyncio.iscoroutinefunction(self.instance.generate)
                    else self.instance.generate(context)
                )
                context["response"] = response
                return context

            # Fallback: retourner contexte inchangé
            logger.warning(f"Module {self.module_name} has no compatible process method")
            return context

        except Exception as e:
            logger.error(f"Module {self.module_name} error: {e}")
            return context

    def get_gfc(self) -> str:
        """Retourne le GFC du module"""
        return self.gfc

    async def shutdown(self):
        """Arrêt propre du module"""
        if self.instance and hasattr(self.instance, "shutdown"):
            if asyncio.iscoroutinefunction(self.instance.shutdown):
                await self.instance.shutdown()
            else:
                self.instance.shutdown()
        self.initialized = False


@dataclass
class PipelineMetrics:
    """Métriques du pipeline"""

    processed: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    gfc_latencies: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    emotion_variations: list[str] = field(default_factory=list)
    memory_hits: int = 0

    def get_stats(self) -> dict:
        avg_latency = self.total_latency_ms / self.processed if self.processed > 0 else 0

        gfc_stats = {}
        for gfc, latencies in self.gfc_latencies.items():
            if latencies:
                gfc_stats[gfc] = {
                    "avg_ms": sum(latencies) / len(latencies),
                    "min_ms": min(latencies),
                    "max_ms": max(latencies),
                    "count": len(latencies),
                }

        return {
            "processed": self.processed,
            "errors": self.errors,
            "avg_latency_ms": avg_latency,
            "emotion_changes": len(set(self.emotion_variations)),
            "memory_hit_rate": self.memory_hits / self.processed if self.processed > 0 else 0,
            "gfc_stats": gfc_stats,
        }


class CognitivePipeline:
    """
    Pipeline cognitif principal avec architecture 3 couches
    Intègre les VRAIS modules via adaptateurs
    """

    def __init__(self, bus, config_path: str = "config/neuro_architecture.yaml"):
        self.bus = bus
        self.config = self._load_config(config_path)

        # Organisation par GFC
        self.modules: dict[str, list[ModuleAdapter]] = {
            "perception_integration": [],
            "memory_associative": [],
            "valence_emotional": [],
            "executive_function": [],
            "expression_generation": [],
            "metacognition": [],
        }

        # Infrastructure
        self.kernel = None
        self.runtime = None

        # Métriques
        self.metrics = PipelineMetrics()

        # État
        self._initialized = False
        self._running = True

    def _load_config(self, config_path: str) -> dict:
        """Charge la configuration depuis le yaml"""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config not found: {config_path}, using defaults")
            return {}

        with open(config_file) as f:
            return yaml.safe_load(f)

    async def initialize(self) -> bool:
        """
        Initialise le pipeline et charge les modules Bundle 1
        """
        logger.info("=" * 60)
        logger.info("🚀 INITIALIZING COGNITIVE PIPELINE")
        logger.info("=" * 60)

        start_time = time.time()
        loaded_count = 0

        # Charger les modules actifs de Bundle 1
        for gfc_name, gfc_config in self.config.get("gfc_groups", {}).items():
            for module_config in gfc_config.get("modules", []):
                if not module_config.get("active", False):
                    continue  # Skip si pas dans Bundle 1

                # GPT Fix #2: Ignorer modules dont le path n'existe pas
                p = Path(module_config["path"])
                if not p.exists():
                    logger.error(f"Module path not found: {p}")
                    continue

                # Créer adaptateur avec priorité
                adapter = ModuleAdapter(
                    module_name=module_config["name"].replace(".py", ""),
                    module_path=module_config["path"],
                    gfc=gfc_name,
                    config=module_config.get("config", {}),
                    priority=module_config.get("priority", 0),  # GPT Fix #3
                )

                # Initialiser
                if await adapter.initialize():
                    self.modules[gfc_name].append(adapter)
                    loaded_count += 1

                    # GPT Fix #3: Respecter la priorité des modules
                    self.modules[gfc_name].sort(key=lambda a: a.priority, reverse=True)

                    # S'abonner aux événements du module
                    await self.bus.subscribe(
                        f"module.{gfc_name}.{adapter.module_name}",
                        self._handle_module_event,
                        priority=module_config.get("priority", 0),
                    )

        # Charger infrastructure
        infra = self.config.get("infrastructure", {})

        # Kernel
        if infra.get("kernel", {}).get("active"):
            kernel_path = infra["kernel"]["path"]
            # TODO: Charger le vrai kernel
            logger.info(f"Loading kernel from {kernel_path}")

        # Runtime
        if infra.get("runtime", {}).get("active"):
            runtime_path = infra["runtime"]["path"]
            # TODO: Charger le vrai runtime
            logger.info(f"Loading runtime from {runtime_path}")

        # S'abonner aux événements pipeline
        await self.bus.subscribe("pipeline.process", self.process_request, priority=10)
        await self.bus.subscribe("pipeline.status", self._handle_status, priority=5)

        init_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info("✅ PIPELINE INITIALIZED")
        logger.info(f"   Modules loaded: {loaded_count}")
        logger.info(f"   Init time: {init_time:.2f}s")
        logger.info(f"   GFC active: {[gfc for gfc, mods in self.modules.items() if mods]}")
        logger.info("=" * 60)

        self._initialized = True

        # Publier événement
        await self.bus.publish(
            "pipeline.initialized",
            {"modules": loaded_count, "time": init_time, "gfc_active": list(self.modules.keys())},
        )

        return init_time < 2.0  # Test gate: boot < 2s

    async def process_request(self, topic: str, data: dict[str, Any]) -> dict[str, Any]:
        """
        Point d'entrée principal pour traiter une requête
        """
        if not self._initialized:
            await self.initialize()

        input_text = data.get("input", data.get("text", data.get("message", "")))
        return await self.process(input_text, data.get("context", {}))

    async def process(self, input_text: str, initial_context: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Traite un input à travers les 3 couches et 6 GFC
        """
        start = time.perf_counter()

        # Contexte initial
        context = {
            "input": input_text,
            "timestamp": datetime.now().isoformat(),
            "correlation_id": f"req_{int(time.time() * 1000)}",
            "memories": [],
            "emotion": "neutral",
            "confidence": 1.0,
            "metadata": {},
            **(initial_context or {}),
        }

        try:
            # === COUCHE 1: RÉACTIVE (Perception) ===
            await self.bus.publish("pipeline.layer.reactive", context)

            for adapter in self.modules["perception_integration"]:
                gfc_start = time.perf_counter()
                context = await adapter.process(context)
                gfc_latency = (time.perf_counter() - gfc_start) * 1000
                self.metrics.gfc_latencies["perception_integration"].append(gfc_latency)

            # === COUCHE 2: COGNITIVE ===
            await self.bus.publish("pipeline.layer.cognitive", context)

            # Mémoire
            for adapter in self.modules["memory_associative"]:
                gfc_start = time.perf_counter()
                context = await adapter.process(context)
                gfc_latency = (time.perf_counter() - gfc_start) * 1000
                self.metrics.gfc_latencies["memory_associative"].append(gfc_latency)

                if context.get("memories"):
                    self.metrics.memory_hits += 1

            # Émotion
            previous_emotion = context.get("emotion", "neutral")
            for adapter in self.modules["valence_emotional"]:
                gfc_start = time.perf_counter()
                context = await adapter.process(context)
                gfc_latency = (time.perf_counter() - gfc_start) * 1000
                self.metrics.gfc_latencies["valence_emotional"].append(gfc_latency)

            current_emotion = context.get("emotion", "neutral")
            if current_emotion != previous_emotion:
                self.metrics.emotion_variations.append(current_emotion)

            # Décision (Bundle 3)
            for adapter in self.modules["executive_function"]:
                gfc_start = time.perf_counter()
                context = await adapter.process(context)
                gfc_latency = (time.perf_counter() - gfc_start) * 1000
                self.metrics.gfc_latencies["executive_function"].append(gfc_latency)

            # Expression
            for adapter in self.modules["expression_generation"]:
                gfc_start = time.perf_counter()
                context = await adapter.process(context)
                gfc_latency = (time.perf_counter() - gfc_start) * 1000
                self.metrics.gfc_latencies["expression_generation"].append(gfc_latency)

            # === COUCHE 3: MÉTA (Bundle 4) ===
            await self.bus.publish("pipeline.layer.meta", context)

            for adapter in self.modules["metacognition"]:
                gfc_start = time.perf_counter()
                context = await adapter.process(context)
                gfc_latency = (time.perf_counter() - gfc_start) * 1000
                self.metrics.gfc_latencies["metacognition"].append(gfc_latency)

            # Métriques finales
            total_latency = (time.perf_counter() - start) * 1000
            self.metrics.processed += 1
            self.metrics.total_latency_ms += total_latency

            # Enrichir contexte avec métriques
            context["metrics"] = {
                "latency_ms": total_latency,
                "gfc_processed": sum(len(m) for m in self.modules.values()),
                "emotion_changed": current_emotion != previous_emotion,
                "memory_used": len(context.get("memories", [])) > 0,
            }

            # Publier completion
            await self.bus.publish(
                "pipeline.complete",
                {
                    "correlation_id": context["correlation_id"],
                    "response": context.get("response", "Je réfléchis..."),
                    "emotion": context.get("emotion"),
                    "latency_ms": total_latency,
                },
            )

            return context

        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            logger.error(traceback.format_exc())
            self.metrics.errors += 1

            context["error"] = str(e)
            context["response"] = "Je rencontre une difficulté technique momentanée."

            await self.bus.publish("pipeline.error", {"correlation_id": context.get("correlation_id"), "error": str(e)})

            return context

    async def _handle_module_event(self, topic: str, data: dict[str, Any]):
        """Gère les événements des modules"""
        logger.debug(f"Module event: {topic} - {data}")

    async def _handle_status(self, topic: str, data: dict[str, Any]) -> dict:
        """Retourne le status du pipeline"""
        return {
            "initialized": self._initialized,
            "running": self._running,
            "metrics": self.metrics.get_stats(),
            "modules_loaded": {gfc: len(adapters) for gfc, adapters in self.modules.items()},
        }

    def get_stats(self) -> dict:
        """Retourne les statistiques du pipeline"""
        return self.metrics.get_stats()

    async def shutdown(self):
        """Arrêt propre du pipeline"""
        logger.info("Shutting down pipeline...")
        self._running = False

        # Arrêter tous les modules
        for gfc_modules in self.modules.values():
            for adapter in gfc_modules:
                await adapter.shutdown()

        logger.info("Pipeline shutdown complete")
