#!/usr/bin/env python3
"""
BrainKernel - Noyau central avec auto-load du census
Int�gre toutes les corrections et am�liorations
"""

import asyncio
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Bridge
from ..bridge.registry import BridgeRegistry

# Core components
from .neural_bus import EventPriority, NeuralBus, NeuralConnector, NeuralEnvelope
from .service_registry import ServiceRegistry

logger = logging.getLogger(__name__)


class BrainKernel:
    """
    Noyau central du cerveau Jeffrey
    Auto-load tous les modules du census (am�lioration Grok)
    """

    def __init__(self, bridge: BridgeRegistry, config_path: str | None = None):
        """
        Initialise le cerveau avec auto-discovery
        Args:
            bridge: Registry des connexions au monde extérieur
            config_path: Chemin vers fichier config OU dict de config direct
        """
        logger.info("🧠 Initializing BrainKernel with Census Auto-Load...")

        # Config
        self._config = self._load_config(config_path)

        # Syst�me nerveux (avec Redis si disponible)
        redis_url = self._config.get("redis_url", "redis://localhost:6379")
        self.bus = NeuralBus(redis_url)

        # Registre des services
        self.registry = ServiceRegistry()

        # Bridge externe
        self.bridge = bridge

        # Components core (charg�s dynamiquement)
        self.components = {}

        # Census pour auto-load (am�lioration Grok)
        self.census = self._load_census()

        # Sessions (m�moire de travail)
        self._sessions: dict[str, dict[str, Any]] = {}

        # M�triques d�taill�es
        self._metrics = defaultdict(lambda: defaultdict(int))

        # �tat
        self._initialized = False
        self._running = False

        # Connecteurs
        self._connectors: dict[str, NeuralConnector] = {}

    def _load_config(self, config_path: str | None = None) -> dict[str, Any]:
        """Charge la configuration depuis un fichier ou les défauts"""
        # Configuration par défaut
        base_config = {
            "redis_url": None,  # None par défaut pour tests
            "bus_workers": 4,
            "memory_cache_size": 1000,
            "session_ttl": 3600,
            "enable_symbiosis": True,
            "enable_proactive": True,
            "auto_load_census": True,
            "census_path": "data/census_complete.json",
            "load_ui_modules": False,  # UI désactivée par défaut
            "use_stubs_for_missing": False,  # NOUVEAU - Utiliser stubs si modules manquants
            "load_nlp_models": False,  # NOUVEAU - Éviter chargements lourds
            "fast_test_mode": False,  # NOUVEAU - Mode test rapide
        }

        # Si config_path est un dict, l'utiliser directement
        if isinstance(config_path, dict):
            base_config.update(config_path)
            return base_config

        # Si c'est un chemin de fichier
        if config_path:
            config_file = Path(config_path)
            if config_file.exists():
                try:
                    with open(config_file) as f:
                        loaded_config = json.load(f)
                        base_config.update(loaded_config)
                except Exception as e:
                    logger.error(f"Failed to load config from {config_path}: {e}")

        return base_config

    def _load_census(self) -> dict[str, Any]:
        """
        AM�LIORATION GROK : Charge le census pour auto-discovery
        """
        census_path = Path(self._config.get("census_path", "data/census_complete.json"))

        # Essayer aussi le chemin tools/reports si data n'existe pas
        if not census_path.exists():
            census_path = Path("tools/reports/census_complete_v2.json")

        if not census_path.exists():
            logger.warning(f"Census file not found: {census_path}")
            return {}

        try:
            with open(census_path) as f:
                census = json.load(f)
                logger.info(f" Loaded census with {len(census.get('modules', []))} modules")
                return census
        except Exception as e:
            logger.error(f"Failed to load census: {e}")
            return {}

    async def _auto_load_modules(self) -> None:
        """
        AMÉLIORATION : Auto-load tous les modules depuis le census en filtrant l'UI
        """
        if not self._config.get("auto_load_census") or not self.census:
            logger.info("Auto-load disabled or no census available")
            return

        logger.info("🔄 Auto-loading modules from census...")

        # Filtrage des modules
        load_ui = self._config.get("load_ui_modules", False)
        load_nlp = self._config.get("load_nlp_models", False)
        fast_mode = self._config.get("fast_test_mode", False)

        ui_keywords = ("kivy", "ui", "screen", "gui", "frontend", "avatar", "display", "window")
        nlp_keywords = (
            "sentence_transformer",
            "huggingface",
            "transformers",
            "bert",
            "gpt",
            "embedding",
        )

        modules_by_category = defaultdict(list)
        skipped_counts = {"ui": 0, "nlp": 0, "missing": 0}

        # Grouper les modules par catégorie avec filtrage
        for module_info in self.census.get("modules", []):
            path = module_info.get("path", "").lower()
            name = module_info.get("name", "").lower()
            category = module_info.get("category", "unknown").upper()

            # Skip UI modules si désactivé
            if not load_ui:
                if any(keyword in path for keyword in ui_keywords) or category in (
                    "UI",
                    "FRONTEND",
                    "AVATAR",
                    "DISPLAY",
                    "GUI",
                ):
                    skipped_counts["ui"] += 1
                    logger.debug(f"⏭️ Skipping UI module: {name}")
                    continue

            # Skip NLP modules si désactivé ou mode test rapide
            if not load_nlp or fast_mode:
                if any(keyword in path for keyword in nlp_keywords) or category in (
                    "NLP",
                    "ML",
                    "AI",
                    "EMBEDDING",
                ):
                    skipped_counts["nlp"] += 1
                    logger.debug(f"⏭️ Skipping NLP module: {name}")
                    continue

            modules_by_category[category].append(module_info)

        # Log des modules skippés
        if skipped_counts["ui"] > 0:
            logger.info(f"⏭️ Skipped {skipped_counts['ui']} UI modules")
        if skipped_counts["nlp"] > 0:
            logger.info(f"⏭️ Skipped {skipped_counts['nlp']} NLP modules")

        # Charger par catégorie avec priorité
        priority_order = [
            "MEMORY",  # Mémoire d'abord
            "EMOTIONS",  # Puis émotions
            "CONSCIOUSNESS",  # Conscience
            "ORCHESTRATION",  # Orchestration
            "SKILLS",  # Compétences
            "SECURITY",  # Sécurité
            "BRIDGE",  # Bridge en dernier
        ]

        loaded_count = 0
        failed_count = 0

        for category in priority_order + list(set(modules_by_category.keys()) - set(priority_order)):
            modules = modules_by_category.get(category, [])

            for module_info in modules:
                path = module_info.get("path", "")
                name = module_info.get("name", "")

                # Essayer de charger le module
                try:
                    # Construire le chemin d'import
                    import_path = path.replace("/", ".").replace(".py", "")

                    # Cas spéciaux pour les modules core
                    if "memory" in import_path.lower():
                        if "memory_manager" in import_path:
                            from ..core.memory.memory_manager import MemoryManager

                            self.components["memory"] = MemoryManager()
                            loaded_count += 1

                    elif "emotion" in import_path.lower():
                        if "emotion_engine" in import_path:
                            from ..core.emotions.core.emotion_engine import EmotionEngine

                            self.components["emotions"] = EmotionEngine()
                            loaded_count += 1

                    elif "consciousness" in import_path.lower():
                        if "jeffrey_consciousness" in import_path:
                            from ..core.consciousness.jeffrey_consciousness_v3 import JeffreyConsciousnessV3

                            self.components["consciousness"] = JeffreyConsciousnessV3()
                            loaded_count += 1

                    elif "orchestrat" in import_path.lower():
                        if "ia_orchestrator_ultimate" in import_path:
                            from ..core.orchestration.ia_orchestrator_ultimate import UltimateOrchestrator

                            self.components["orchestrator"] = UltimateOrchestrator()
                            loaded_count += 1

                    elif "symbiosis" in import_path.lower():
                        from .living_memory.symbiosis_engine import SymbiosisEngine

                        self.components["symbiosis"] = SymbiosisEngine()
                        loaded_count += 1

                    # Enregistrer dans le registry
                    if name and name in self.components:
                        self.registry.register(name, self.components[name], {"category": category, "path": path})

                        # Créer un connecteur
                        self._connectors[name] = NeuralConnector(self.bus, name)

                except Exception as e:
                    logger.warning(f"Failed to load {name} from {path}: {e}")
                    failed_count += 1

        logger.info(f"✅ Auto-loaded {loaded_count} modules, {failed_count} failed")

    async def initialize(self) -> None:
        """
        Initialise tous les composants du cerveau
        """
        if self._initialized:
            logger.warning("BrainKernel already initialized")
            return

        logger.info("🚀 Starting BrainKernel initialization...")

        try:
            # 0. Injecter les stubs si configuré
            if self._config.get("use_stubs_for_missing", False):
                from ..stubs import inject_stubs_to_sys_modules

                inject_stubs_to_sys_modules(self._config)
                logger.info("⚙️ Stubs injected for missing modules")

            # 1. Démarrer le bus
            await self.bus.start(num_workers=self._config["bus_workers"])

            # 2. Auto-load modules depuis le census
            if self._config.get("auto_load_census"):
                await self._auto_load_modules()

            # 3. S'assurer que les modules critiques sont présents
            self._ensure_critical_modules()

            # 4. Initialiser les composants
            await self._initialize_components()

            # 5. Configurer les routes neuronales
            self._setup_neural_routes()

            # 6. Initialiser le Bridge
            await self.bridge.initialize_all()

            # 7. Démarrer la symbiose si activée
            if self._config.get("enable_symbiosis") and "symbiosis" in self.components:
                await self.components["symbiosis"].initialize(self.bus)

            # 8. Initialiser l'orchestrateur
            if "orchestrator" in self.components:
                orchestrator = self.components["orchestrator"]
                if hasattr(orchestrator, "initialize_with_kernel"):
                    await orchestrator.initialize_with_kernel(self)

            # 9. Démarrer les processus proactifs
            if self._config.get("enable_proactive"):
                asyncio.create_task(self._proactive_thinking_loop())

            # État final
            self._initialized = True
            self._running = True

            # Publier l'événement de démarrage
            await self.bus.publish(
                NeuralEnvelope(
                    topic="system.started",
                    payload={
                        "kernel": "BrainKernel",
                        "version": "2.0.0",
                        "modules_loaded": list(self.components.keys()),
                        "census_modules": len(self.census.get("modules", [])),
                    },
                    priority=EventPriority.HIGH,
                    source="kernel",
                )
            )

            logger.info("✅ BrainKernel initialization complete!")

        except Exception as e:
            logger.error(f"❌ BrainKernel initialization failed: {e}")
            await self.shutdown()
            raise

    def _ensure_critical_modules(self) -> None:
        """S'assure que les modules critiques sont chargés"""
        critical = ["memory", "emotions", "consciousness", "orchestrator"]

        for module_name in critical:
            if module_name not in self.components:
                logger.warning(f"Critical module {module_name} not loaded, using fallback")

                # Créer des fallbacks minimaux
                if module_name == "memory":
                    from ..core.memory.memory_manager import MemoryManager

                    self.components["memory"] = MemoryManager()

                elif module_name == "emotions":
                    from ..core.emotions.core.emotion_engine import EmotionEngine

                    self.components["emotions"] = EmotionEngine()

                elif module_name == "consciousness":
                    from ..core.consciousness.jeffrey_consciousness_v3 import JeffreyConsciousnessV3

                    self.components["consciousness"] = JeffreyConsciousnessV3()

                elif module_name == "orchestrator":
                    from ..core.orchestration.ia_orchestrator_ultimate import UltimateOrchestrator

                    self.components["orchestrator"] = UltimateOrchestrator()

                # Enregistrer le fallback
                self.registry.register(module_name, self.components[module_name])
                self._connectors[module_name] = NeuralConnector(self.bus, module_name)

    async def _initialize_components(self) -> None:
        """Initialise tous les composants chargés"""
        for name, component in self.components.items():
            try:
                # Skip symbiosis car elle a besoin du bus
                if name == "symbiosis":
                    self.registry.set_status(name, "ready")
                    logger.info("↷ Skipping generic init for symbiosis (will pass bus later)")
                    continue

                if hasattr(component, "initialize"):
                    await component.start()
                    self.registry.set_status(name, "running")
                    logger.info(f"✅ Initialized: {name}")
                else:
                    self.registry.set_status(name, "ready")

            except Exception as e:
                logger.error(f"Failed to initialize {name}: {e}")
                self.registry.set_status(name, "error")

    def _setup_neural_routes(self) -> None:
        """Configure toutes les routes neuronales"""
        logger.info("Setting up neural routes...")

        # Routes essentielles
        self.bus.register_handler("chat.in", self._handle_chat_input)
        self.bus.register_handler("memory.store", self._handle_memory_store)
        self.bus.register_handler("memory.retrieve", self._handle_memory_retrieve)
        self.bus.register_handler("emotion.analyze", self._handle_emotion_analyze)
        self.bus.register_handler("external.request", self._handle_external_request)
        self.bus.register_handler("system.command", self._handle_system_command)

        # Monitoring
        self.bus.register_handler("*", self._monitor_all_events)

        logger.info("✅ Neural routes configured")

    async def _handle_chat_input(self, envelope: NeuralEnvelope) -> dict[str, Any]:
        """
        Gère l'entrée de chat avec flux complet
        """
        try:
            session_id = envelope.meta.get("session_id", "default")
            message = envelope.payload.get("text", "")

            # Métriques
            self._metrics["chat"]["requests"] += 1

            logger.info(f"Chat input: {message[:50]}...")

            # 1. Analyse émotionnelle
            emotion_state = {}
            if "emotions" in self.components:
                emotion_state = await self.components["emotions"].analyze(message)

            # 2. Contexte de session
            session = self._get_or_create_session(session_id)

            # Créer un contexte simplifié pour consciousness
            simple_context = {
                "session_id": session_id,
                "history_count": len(session.get("history", [])),
                "speaker": session.get("context", {}).get("speaker"),
            }

            # 3. Génération de réponse
            response = "Je traite votre message..."
            if "consciousness" in self.components:
                response = await self.components["consciousness"].respond(
                    message, context=simple_context, emotion_state=emotion_state
                )

            # 4. Stockage mémoire
            if "memory" in self.components:
                # Appel correct de add_to_context (non async)
                self.components["memory"].add_to_context(
                    message=message,
                    user_id=session_id,
                    response=response,
                    metadata={"emotion": emotion_state},
                )

                # Ajouter à l'historique de session
                memory_entry = {
                    "user": message,
                    "assistant": response,
                    "emotion": emotion_state,
                    "timestamp": envelope.timestamp,
                }
                session["history"].append(memory_entry)

            # 5. Renforcer les connexions si symbiose
            if "symbiosis" in self.components:
                await self.components["symbiosis"].strengthen_connection("chat", "consciousness", 0.1)

            # Métriques
            self._metrics["chat"]["success"] += 1

            return {"response": response, "emotion": emotion_state, "processed": True}

        except Exception as e:
            logger.error(f"Chat error: {e}")
            self._metrics["chat"]["errors"] += 1
            return {"error": str(e), "processed": False}

    async def _handle_memory_store(self, envelope: NeuralEnvelope) -> dict[str, Any]:
        """Stockage mémoire"""
        try:
            if "memory" not in self.components:
                return {"error": "Memory component not available", "stored": False}

            memory_id = await self.components["memory"].store(envelope.payload)

            # Renforcer connexion si symbiose
            if "symbiosis" in self.components:
                importance = envelope.payload.get("importance", 0.5)
                await self.components["symbiosis"].strengthen_connection(
                    envelope.source or "unknown", "memory", importance * 0.2
                )

            self._metrics["memory"]["stores"] += 1

            return {"memory_id": memory_id, "stored": True}

        except Exception as e:
            logger.error(f"Memory store error: {e}")
            self._metrics["memory"]["errors"] += 1
            return {"error": str(e), "stored": False}

    async def _handle_memory_retrieve(self, envelope: NeuralEnvelope) -> dict[str, Any]:
        """Récupération mémoire"""
        try:
            if "memory" not in self.components:
                return {"error": "Memory not available", "memories": []}

            query = envelope.payload.get("query", "")
            limit = envelope.payload.get("limit", 10)

            memories = await self.components["memory"].search(query, limit=limit)

            self._metrics["memory"]["retrievals"] += 1

            return {"memories": memories, "count": len(memories)}

        except Exception as e:
            logger.error(f"Memory retrieve error: {e}")
            self._metrics["memory"]["errors"] += 1
            return {"error": str(e), "memories": []}

    async def _handle_emotion_analyze(self, envelope: NeuralEnvelope) -> dict[str, Any]:
        """Analyse émotionnelle"""
        try:
            if "emotions" not in self.components:
                return {"error": "Emotions not available", "analyzed": False}

            text = envelope.payload.get("text", "")
            analysis = await self.components["emotions"].analyze(text)

            self._metrics["emotions"]["analyses"] += 1

            return {"analysis": analysis, "analyzed": True}

        except Exception as e:
            logger.error(f"Emotion error: {e}")
            self._metrics["emotions"]["errors"] += 1
            return {"error": str(e), "analyzed": False}

    async def _handle_external_request(self, envelope: NeuralEnvelope) -> dict[str, Any]:
        """Requête externe via Bridge"""
        try:
            adapter_name = envelope.payload.get("adapter", "http")
            action = envelope.payload.get("action", "get")
            params = envelope.payload.get("params", {})

            # Vérifier permissions
            context = {"avatar": envelope.meta.get("avatar", "unknown")}
            if not self.bridge.check_permission(adapter_name, action, context):
                self._metrics["bridge"]["denied"] += 1
                return {"error": "Permission denied", "processed": False}

            # Récupérer l'adapter
            adapter = self.bridge.get(adapter_name)
            if not adapter:
                return {"error": f"Adapter {adapter_name} not found", "processed": False}

            # Exécuter l'action
            method = getattr(adapter, action, None)
            if not method:
                return {"error": f"Action {action} not supported", "processed": False}

            result = await method(**params)

            self._metrics["bridge"]["requests"] += 1

            return {"result": result, "processed": True}

        except Exception as e:
            logger.error(f"External request error: {e}")
            self._metrics["bridge"]["errors"] += 1
            return {"error": str(e), "processed": False}

    async def _handle_system_command(self, envelope: NeuralEnvelope) -> dict[str, Any]:
        """Commandes système"""
        command = envelope.payload.get("command")

        if command == "health":
            return await self.get_health_status()
        elif command == "metrics":
            return await self.get_metrics()
        elif command == "shutdown":
            asyncio.create_task(self.shutdown())
            return {"status": "Shutdown initiated"}
        else:
            return {"error": f"Unknown command: {command}"}

    async def _monitor_all_events(self, envelope: NeuralEnvelope) -> None:
        """Monitoring global"""
        # Métriques par topic
        self._metrics["events"][envelope.topic] += 1

        # Log si niveau debug
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Event: {envelope.topic} from {envelope.source}")

    async def _proactive_thinking_loop(self) -> None:
        """
        Boucle proactive améliorée (Grok suggestion)
        Utilise la symbiose pour des suggestions intelligentes
        """
        logger.info("Starting enhanced proactive thinking loop...")

        while self._running:
            try:
                await asyncio.sleep(30)  # Cycle de 30 secondes

                for session_id, session in self._sessions.items():
                    if not session.get("history"):
                        continue

                    # Analyser avec symbiose si disponible
                    suggestions = []

                    if "symbiosis" in self.components:
                        analysis = await self.components["symbiosis"].analyze_history(session_id)
                        suggestions = analysis.get("suggestions", [])

                    # Si pas de suggestions symbiose, utiliser basique
                    if not suggestions:
                        last_interaction = session["history"][-1]
                        time_since = (
                            datetime.now()
                            - datetime.fromisoformat(last_interaction.get("timestamp", datetime.now().isoformat()))
                        ).total_seconds()

                        if time_since > 120:  # Inactif depuis 2 minutes
                            suggestions = ["Puis-je vous aider avec autre chose?"]

                    # Publier les suggestions
                    for suggestion in suggestions:
                        await self.bus.publish(
                            NeuralEnvelope(
                                topic="proactive.suggestion",
                                payload={
                                    "session_id": session_id,
                                    "suggestion": suggestion,
                                    "context": session,
                                },
                                priority=EventPriority.LOW,
                                source="kernel.proactive",
                            )
                        )

                # Maintenance périodique
                await self._periodic_maintenance()

            except Exception as e:
                logger.error(f"Proactive thinking error: {e}")

    async def _periodic_maintenance(self) -> None:
        """Maintenance périodique du cerveau"""
        # Nettoyer les sessions expirées
        expired = []
        now = datetime.now()

        for sid, session in self._sessions.items():
            if "last_activity" in session:
                age = (now - session["last_activity"]).total_seconds()
                if age > self._config["session_ttl"]:
                    expired.append(sid)

        for sid in expired:
            del self._sessions[sid]
            logger.info(f"Expired session: {sid}")

        # Sauvegarder l'état de la symbiose
        if "symbiosis" in self.components:
            await self.components["symbiosis"].save_state()

        # Nettoyer les caches
        if hasattr(self.bridge, "get"):
            http_adapter = self.bridge.get("http")
            if http_adapter and hasattr(http_adapter, "clear_cache"):
                if len(http_adapter._cache) > 200:  # Si cache trop gros
                    http_adapter.clear_cache()

    def _get_or_create_session(self, session_id: str) -> dict[str, Any]:
        """Gestion des sessions (mémoire de travail)"""
        if session_id not in self._sessions:
            self._sessions[session_id] = {
                "id": session_id,
                "created": datetime.now(),
                "history": [],
                "context": {},
                "last_activity": datetime.now(),
            }

        session = self._sessions[session_id]
        session["last_activity"] = datetime.now()

        return session

    async def get_health_status(self) -> dict[str, Any]:
        """État de santé complet"""
        bridge_health = await self.bridge.health_check()

        # Santé des composants
        components_health = {}
        for name, component in self.components.items():
            if hasattr(component, "health"):
                try:
                    components_health[name] = await component.health()
                except:
                    components_health[name] = False
            else:
                components_health[name] = self.registry.get_status(name) == "running"

        return {
            "kernel": "healthy" if self._running else "stopped",
            "bus": self.bus.get_metrics(),
            "bridge": bridge_health,
            "sessions": len(self._sessions),
            "components": components_health,
            "census_loaded": len(self.census.get("modules", [])) > 0,
        }

    async def get_metrics(self) -> dict[str, Any]:
        """Métriques détaillées"""
        # Métriques symbiose
        symbiosis_metrics = {}
        if "symbiosis" in self.components:
            symbiosis_metrics = self.components["symbiosis"].get_metrics()

        return {
            "bus": self.bus.get_metrics(),
            "bridge": self.bridge.get_metrics(),
            "sessions": {
                "active": len(self._sessions),
                "total_interactions": sum(len(s.get("history", [])) for s in self._sessions.values()),
            },
            "components": {name: dict(self._metrics[name]) for name in self._metrics},
            "symbiosis": symbiosis_metrics,
            "census": {
                "total_modules": len(self.census.get("modules", [])),
                "loaded_modules": len(self.components),
            },
        }

    async def shutdown(self) -> None:
        """Arrêt propre du cerveau"""
        logger.info("🛑 Shutting down BrainKernel...")

        self._running = False

        # Sauvegarder l'état
        if "symbiosis" in self.components:
            await self.components["symbiosis"].save_state()

        # Publier l'événement d'arrêt
        await self.bus.publish(
            NeuralEnvelope(
                topic="system.stopping",
                payload={"kernel": "BrainKernel"},
                priority=EventPriority.CRITICAL,
                source="kernel",
            )
        )

        # Arrêter les composants
        await self.bus.stop()
        await self.bridge.shutdown_all()

        self._initialized = False

        logger.info("✅ BrainKernel shutdown complete")
