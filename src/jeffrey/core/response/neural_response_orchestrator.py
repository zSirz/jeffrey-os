import asyncio
import hashlib
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import networkx as nx  # Pour détection de cycles
import numpy as np

logger = logging.getLogger(__name__)

# Imports conditionnels avec fallback
try:
    from ..bus.neurobus_adapter import NeuroBusAdapter
except ImportError:
    NeuroBusAdapter = None

try:
    from ..bus.event_priority import EventPriority
except ImportError:
    # Fallback si EventPriority n'existe pas
    from enum import Enum

    class EventPriority(Enum):
        LOW = "low"
        NORMAL = "normal"
        HIGH = "high"
        CRITICAL = "critical"


try:
    from ..llm.apertus_client import ApertusClient
except ImportError:
    import logging

    # Configuration du logger AVANT usage
    logger = logging.getLogger(__name__)

    # Stub safe pour éviter les crashes sans LLM configuré
    class ApertusClient:
        """Stub ApertusClient pour tests sans LLM réel"""

        def __init__(self, **kwargs):
            self.is_stub = True
            self.model = "stub-model"
            logger.warning(
                "⚠️  Using STUB ApertusClient - No real LLM configured. "
                "Install/configure ApertusClient for real responses."
            )

        async def stream(self, prompt: str = "", **kwargs):
            """Génère un stream stub identifiable"""
            chunks = [
                "[STUB] ",
                "No ",
                "LLM ",
                "available. ",
                "Please ",
                "configure ",
                "ApertusClient ",
                "for ",
                "real ",
                "responses.",
            ]
            for chunk in chunks:
                yield chunk
                await asyncio.sleep(0.01)  # Simule latence réseau

        async def generate_text(self, prompt: str = "", **kwargs):
            """Génère une réponse stub identifiable"""
            return {
                "text": "[STUB] No LLM available - please configure ApertusClient",
                "is_stub": True,
                "model": self.model,
                "tokens": 0,
                "prompt": prompt,
            }


try:
    from ..memory.unified_memory import UnifiedMemory
except ImportError:
    UnifiedMemory = None


# NeuralEnvelope local
@dataclass
class NeuralEnvelope:
    """Enveloppe pour messages sur le bus neuronal"""

    topic: str
    payload: dict[str, Any] = field(default_factory=dict)
    correlation_id: str | None = None
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = field(default_factory=time.time)


from .basal_ganglia_ucb1 import ContextualBanditScheduler
from .neural_blackboard_v2 import NeuralBlackboard


class ProcessingPhase(Enum):
    """Phases de traitement inspirées du cerveau"""

    THALAMUS = "thalamus"
    HIPPOCAMPUS = "hippocampus"
    AMYGDALA = "amygdala"
    CORTEX = "cortex"
    WERNICKE = "wernicke"
    BROCA = "broca"


class CollectionStrategy(Enum):
    """Stratégies de collecte des résultats"""

    ALL = "all"
    FIRST_RELIABLE = "first_reliable"
    ANY = "any"
    QUORUM = "quorum"  # NOUVEAU


@dataclass
class NeuralSignal:
    """Signal neuronal avec deadline absolue"""

    user_input: str
    user_id: str
    correlation_id: str  # Utilisé partout maintenant !

    # États par phase
    thalamus_data: dict = field(default_factory=dict)
    hippocampus_data: dict = field(default_factory=dict)
    amygdala_data: dict = field(default_factory=dict)
    cortex_data: dict = field(default_factory=dict)
    wernicke_data: dict = field(default_factory=dict)
    broca_data: dict = field(default_factory=dict)

    # Contrôle
    priority: EventPriority = EventPriority.NORMAL
    deadline_absolute: float = None  # Timestamp absolu
    budget_remaining: float = 2.0

    # Métriques
    phase_timings: dict[str, float] = field(default_factory=dict)
    confidence_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.deadline_absolute is None:
            self.deadline_absolute = time.time() + self.budget_remaining

    def is_urgent(self) -> bool:
        """Signal urgent ?"""
        return self.priority == EventPriority.HIGH or self.amygdala_data.get("intensity", 0) > 0.8

    def should_skip_enrichments(self) -> bool:
        """Sauter les enrichissements non-essentiels ?"""
        time_left = self.deadline_absolute - time.time()
        return time_left < 0.5 or self.is_urgent()

    def to_context(self) -> dict:
        """Contexte pour modules"""
        return {
            "user_input": self.user_input,
            "user_id": self.user_id,
            "correlation_id": self.correlation_id,
            "emotions": self.amygdala_data,
            "memories": self.hippocampus_data,
            "intent": self.thalamus_data.get("intent"),
            "priority": self.priority.value,
            "deadline_ms": int(self.deadline_absolute * 1000),
        }


class CycleDetector:
    """Détecte les boucles infinies dans les événements"""

    def __init__(self, window_size: int = 100):
        self.event_graph = nx.DiGraph()
        self.event_history = []
        self.window_size = window_size

    def add_event(self, source: str, target: str, correlation_id: str):
        """Ajoute un événement au graphe"""

        # Ajouter à l'historique
        self.event_history.append(
            {
                "source": source,
                "target": target,
                "correlation_id": correlation_id,
                "timestamp": time.time(),
            }
        )

        # Limiter fenêtre
        if len(self.event_history) > self.window_size:
            self.event_history.pop(0)

        # Ajouter au graphe
        self.event_graph.add_edge(source, target, correlation_id=correlation_id)

        # Nettoyer vieux edges
        if self.event_graph.number_of_edges() > self.window_size * 2:
            # Garder seulement les récents
            old_edges = list(self.event_graph.edges())[: self.window_size]
            self.event_graph.remove_edges_from(old_edges)

    def detect_cycle(self) -> list[str] | None:
        """Détecte un cycle"""

        try:
            cycles = list(nx.simple_cycles(self.event_graph))
            if cycles and len(cycles[0]) > 3:
                return cycles[0]  # Retourner le premier cycle trouvé
        except:
            pass

        return None

    def should_block(self, source: str, target: str) -> bool:
        """Détermine si on doit bloquer cet événement"""

        # Compter occurrences récentes
        recent_count = sum(1 for e in self.event_history[-20:] if e["source"] == source and e["target"] == target)

        # Bloquer si trop d'occurrences
        return recent_count > 5


class EarlyExitCache:
    """Cache sémantique pour Early Exit"""

    def __init__(self, max_size: int = 1000):
        from cachetools import TTLCache

        self.pattern_cache = TTLCache(maxsize=max_size, ttl=600)  # 10 min TTL

        # Patterns compilés pour détection rapide
        self.simple_patterns = [
            (r"^(bonjour|salut|hello|hi|hey)[\s!]*$", "greeting"),
            (r"^(au revoir|bye|à plus|a\+)[\s!]*$", "farewell"),
            (r"^(merci|thanks|thx)[\s!]*$", "thanks"),
            (r"^(ça va|comment vas-tu|how are you)[\s?]*$", "how_are_you"),
        ]

        # Cache vectoriel (si embeddings disponibles)
        self.vector_cache = []  # [(embedding, response, confidence)]

    def check_pattern(self, text: str) -> tuple[str, str, float] | None:
        """Vérifie patterns simples"""

        text_lower = text.lower().strip()

        # Check cache exact
        if text_lower in self.pattern_cache:
            return self.pattern_cache[text_lower]

        # Check patterns
        for pattern, intent in self.simple_patterns:
            if re.match(pattern, text_lower):
                response = self._get_response_for_intent(intent)
                result = (intent, response, 0.99)
                self.pattern_cache[text_lower] = result
                return result

        return None

    def check_semantic(self, embedding: np.ndarray, threshold: float = 0.95) -> tuple[str, float] | None:
        """Vérifie similarité sémantique"""

        if not self.vector_cache:
            return None

        best_match = None
        best_score = 0

        for cached_emb, response, confidence in self.vector_cache:
            # Cosine similarity
            similarity = np.dot(embedding, cached_emb) / (np.linalg.norm(embedding) * np.linalg.norm(cached_emb))

            if similarity > best_score and similarity > threshold:
                best_score = similarity
                best_match = (response, confidence * similarity)

        return best_match

    def add_semantic(self, embedding: np.ndarray, response: str, confidence: float):
        """Ajoute au cache sémantique"""

        self.vector_cache.append((embedding, response, confidence))

        # Limiter taille
        if len(self.vector_cache) > 100:
            self.vector_cache.pop(0)

    def _get_response_for_intent(self, intent: str) -> str:
        """Réponses prédéfinies par intent"""

        responses = {
            "greeting": "Bonjour ! Comment puis-je t'aider aujourd'hui ?",
            "farewell": "Au revoir ! À bientôt !",
            "thanks": "Je t'en prie ! C'est un plaisir de t'aider.",
            "how_are_you": "Je vais très bien, merci ! Et toi, comment te sens-tu ?",
        }

        return responses.get(intent, "Comment puis-je t'aider ?")


class NeuralResponseOrchestrator:
    """
    Orchestrateur neuronal MAGISTRAL v2.0
    """

    def __init__(
        self,
        bus: NeuroBusAdapter,
        memory: UnifiedMemory,
        apertus_client: ApertusClient | None = None,
    ):
        self.bus = bus
        self.memory = memory
        self.apertus_client = apertus_client or ApertusClient()

        # Composants core
        # Importer les composants nécessaires

        self.blackboard = NeuralBlackboard(ttl_seconds=120, max_entries=1000)
        # We'll create the system token lazily on first use
        self.system_token = None
        self.scheduler = ContextualBanditScheduler(exploration_factor=2.0)
        self.cycle_detector = CycleDetector()
        self.early_exit_cache = EarlyExitCache()

        # Protection
        self.module_semaphore = asyncio.Semaphore(10)
        self.circuit_breakers = {}

        # Streaming
        self.stream_buffers = {}  # correlation_id -> buffer

        # Stratégies par phase
        self.phase_strategies = {
            ProcessingPhase.THALAMUS: CollectionStrategy.FIRST_RELIABLE,
            ProcessingPhase.HIPPOCAMPUS: CollectionStrategy.QUORUM,
            ProcessingPhase.AMYGDALA: CollectionStrategy.ALL,
            ProcessingPhase.CORTEX: CollectionStrategy.ALL,
            ProcessingPhase.WERNICKE: CollectionStrategy.ALL,
            ProcessingPhase.BROCA: CollectionStrategy.ALL,
        }

        # Métriques ML
        self.neural_metrics = MLEnabledMetrics()

        # État
        self.running = False

        # Logger pour debug (Fix 6)
        import logging

        self.logger = logging.getLogger(__name__)

    async def initialize(self):
        """Initialise l'orchestrateur"""

        # Démarrer blackboard
        await self.blackboard.start()

        # S'abonner aux événements
        self._subscribe_to_events()

        # Créer dossier data si nécessaire
        os.makedirs("data", exist_ok=True)

        # Charger état si existe
        try:
            self.scheduler.load_state("data/scheduler_state.json")
        except:
            pass

        self.running = True
        print("✅ NeuralResponseOrchestrator v2.0 initialized")

    async def shutdown(self):
        """Arrêt propre"""

        self.running = False

        # Sauvegarder état
        self.scheduler.save_state("data/scheduler_state.json")

        # Arrêter blackboard
        await self.blackboard.stop()

    def _subscribe_to_events(self):
        """S'abonne aux événements du bus"""

        self.bus.subscribe("input.user", self._handle_user_input)
        self.bus.subscribe("module.register", self._register_module)
        self.bus.subscribe("thalamus.early_exit", self._handle_early_exit)

        # Résultats par phase
        for phase in ProcessingPhase:
            self.bus.subscribe(f"phase.{phase.value}.complete", self._handle_phase_complete)

        # Streams - Utilise un topic unique au lieu de wildcard
        self.bus.subscribe("response.stream", self._handle_stream)

    # === MÉTHODES MANQUANTES AJOUTÉES ===

    def _register_module(self, envelope: NeuralEnvelope):
        """Enregistre un nouveau module"""
        info = envelope.payload or {}
        module_id = info.get("module_id")
        phase = info.get("phase")
        if not module_id or not phase:
            return
        self.scheduler.register_module(info)
        if module_id not in self.circuit_breakers:
            self.circuit_breakers[module_id] = CircuitBreaker()

    async def _handle_phase_complete(self, envelope: NeuralEnvelope):
        """Gère la complétion d'une phase"""
        # Optionnel: tracer/metrics ; l'agrégation est gérée localement.
        pass

    async def _handle_stream(self, envelope: NeuralEnvelope):
        """Gère les streams de réponse"""
        # Fan-out éventuel vers le client ; sinon no-op pour compat.
        corr = envelope.correlation_id
        if corr:
            await self.bus.publish(
                NeuralEnvelope(
                    topic=f"client.stream.{corr}",
                    payload=envelope.payload,
                    correlation_id=corr,
                    priority=EventPriority.HIGH,
                )
            )

    async def _handle_early_exit(self, envelope: NeuralEnvelope):
        """Gère l'early exit du thalamus"""
        correlation_id = envelope.correlation_id
        response = envelope.payload.get("response")
        if response:
            await self._cancel_all_phases(correlation_id)
            await self._emit_response(
                {"response": response, "metadata": {"early_exit": True, "processing_time": 0.01}},
                correlation_id,
            )

    async def _execute_all(
        self, module_ids: list[str], signal: NeuralSignal, capability: str, timeout: float
    ) -> list[dict]:
        """Exécute tous les modules en parallèle"""
        tasks = [
            asyncio.create_task(self._execute_module_secure(mid, signal, capability, timeout)) for mid in module_ids
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if isinstance(r, dict)]

    async def _execute_first_reliable(
        self, module_ids: list[str], signal: NeuralSignal, capability: str, timeout: float
    ) -> list[dict]:
        """Exécute modules et retourne le premier résultat fiable"""
        tasks = [
            asyncio.create_task(self._execute_module_secure(mid, signal, capability, timeout)) for mid in module_ids
        ]
        done, pending = await asyncio.wait(tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        # Cherche un résultat "fiable"
        for t in done:
            try:
                r = await t
                if isinstance(r, dict) and r.get("confidence", 0) >= 0.7:
                    for p in pending:
                        p.cancel()
                    return [r]
            except:
                pass
        # Sinon, récupère ce qui reste encore dans le budget
        remain = await asyncio.gather(*pending, return_exceptions=True)
        return [r for r in remain if isinstance(r, dict)]

    # === FIN DES MÉTHODES MANQUANTES ===

    async def _handle_user_input(self, envelope: NeuralEnvelope):
        """Point d'entrée principal"""

        correlation_id = (
            envelope.correlation_id or hashlib.md5(f"{envelope.payload.get('text')}:{time.time()}".encode()).hexdigest()
        )

        # Check early exit AVANT tout
        early_result = self.early_exit_cache.check_pattern(envelope.payload.get("text", ""))

        if early_result:
            intent, response, confidence = early_result
            print(f"⚡ Early Exit: {intent} (confidence: {confidence})")

            # Cancel toutes les phases
            await self._cancel_all_phases(correlation_id)

            # Émettre réponse
            await self._emit_response(
                {
                    "response": response,
                    "metadata": {
                        "early_exit": True,
                        "intent": intent,
                        "confidence": confidence,
                        "processing_time": 0.001,
                    },
                },
                correlation_id,
            )

            return

        # Créer signal
        signal = NeuralSignal(
            user_input=envelope.payload.get("text", ""),
            user_id=envelope.payload.get("user_id", "default"),
            correlation_id=correlation_id,
            priority=envelope.priority,
        )

        # Détecter cycles potentiels
        if self.cycle_detector.should_block("input", "pipeline"):
            print("🔴 Cycle détecté, blocking")
            return

        self.cycle_detector.add_event("input", "pipeline", correlation_id)

        # Lancer pipeline
        try:
            response = await self._process_neural_pipeline(signal)
            await self._emit_response(response, correlation_id)

        except Exception as e:
            print(f"❌ Pipeline error: {e}")
            await self._emit_error(str(e), correlation_id)

    async def _ensure_system_token(self):
        """Ensure system token exists"""
        if not self.system_token:
            self.system_token = await self.blackboard.create_capability_token(
                module_id="orchestrator", allowed_keys={"*"}, ttl_ms=3600000
            )

    async def _process_neural_pipeline(self, signal: NeuralSignal) -> dict:
        """Pipeline neuronal complet"""

        # Ensure we have system token
        await self._ensure_system_token()

        start_time = time.time()

        # PHASE 1: THALAMUS
        await self._execute_phase(ProcessingPhase.THALAMUS, signal)

        # Check si on doit continuer
        if signal.should_skip_enrichments():
            print("⚡ Skipping enrichments (urgent/timeout)")
        else:
            # PHASES 2-3: HIPPOCAMPUS + AMYGDALA (parallèle)
            await asyncio.gather(
                self._execute_phase(ProcessingPhase.HIPPOCAMPUS, signal),
                self._execute_phase(ProcessingPhase.AMYGDALA, signal),
                return_exceptions=True,
            )

        # PHASE 4: CORTEX (LLM)
        await self._generate_llm_response(signal)

        # PHASES 5-6: Enrichissements si budget
        if not signal.should_skip_enrichments():
            await self._execute_phase(ProcessingPhase.WERNICKE, signal)
            await self._execute_phase(ProcessingPhase.BROCA, signal)

        # Métriques
        total_time = time.time() - start_time
        self.neural_metrics.record_pipeline(signal, total_time)

        # Calculer reward pour scheduler
        quality = signal.confidence_scores.get("global", 0.5)
        latency_ms = total_time * 1000
        reward = self.scheduler.calculate_reward(quality, latency_ms)

        # Update scheduler pour tous les modules utilisés
        for phase, timing in signal.phase_timings.items():
            # (Simplification, en réalité il faudrait tracker par module)
            pass

        return self._build_final_response(signal)

    async def _execute_phase(self, phase: ProcessingPhase, signal: NeuralSignal):
        """Exécute une phase avec sélection UCB1"""

        start = time.time()

        # Vérifier deadline
        if time.time() > signal.deadline_absolute:
            print(f"⏰ Deadline dépassée, skipping {phase.value}")
            return

        # Créer contexte pour scheduler
        context = signal.to_context()

        # Ajouter embedding si disponible
        if hasattr(self, "embedder"):
            context["embedding"] = await self.embedder.embed(signal.user_input)

        # Sélectionner modules via UCB1
        budget_ms = (signal.deadline_absolute - time.time()) * 1000
        strategy = "ucb1"  # ou 'quorum' pour HIPPOCAMPUS

        if phase == ProcessingPhase.HIPPOCAMPUS:
            strategy = "quorum"
            context["quorum_size"] = 2

        selected_modules = self.scheduler.select_modules(phase.value, budget_ms, context, strategy)

        if not selected_modules:
            print(f"📊 No modules selected for {phase.value}")
            return

        print(f"🎯 {phase.value}: {len(selected_modules)} modules selected")

        # Créer capability token pour cette phase
        capability = await self.blackboard.create_capability_token(
            signal.correlation_id,
            {f"{phase.value}_*"},
            ttl=30,  # Accès limité à cette phase
        )

        # Écrire contexte au blackboard
        await self.blackboard.write(signal.correlation_id, f"{phase.value}_context", context, self.system_token)

        # Stratégie de collecte
        strategy_enum = self.phase_strategies.get(phase, CollectionStrategy.ALL)

        # Exécuter selon stratégie
        if strategy_enum == CollectionStrategy.QUORUM:
            results = await self._execute_quorum(
                selected_modules, signal, capability, budget_ms / len(selected_modules)
            )
        elif strategy_enum == CollectionStrategy.FIRST_RELIABLE:
            results = await self._execute_first_reliable(
                selected_modules, signal, capability, budget_ms / len(selected_modules)
            )
        else:
            results = await self._execute_all(selected_modules, signal, capability, budget_ms / len(selected_modules))

        # Agréger résultats
        self._aggregate_phase_results(phase, signal, results)

        # Mettre à jour métriques
        elapsed = time.time() - start
        signal.phase_timings[phase.value] = elapsed
        signal.budget_remaining -= elapsed

        # Feedback au scheduler
        for module_id in selected_modules:
            success = any(r.get("module_id") == module_id for r in results if r)

            # Calculer reward simple
            if success:
                quality = 0.7  # À améliorer avec vraie mesure
                reward = self.scheduler.calculate_reward(quality, elapsed * 1000)
            else:
                reward = 0

            self.scheduler.update_module_performance(
                module_id, elapsed * 1000 / len(selected_modules), success, reward, context
            )

    async def _execute_quorum(
        self,
        module_ids: list[str],
        signal: NeuralSignal,
        capability: str,
        timeout: float,
        quorum: int = 2,
    ) -> list[dict]:
        """Exécution QUORUM : s'arrête dès que N modules ont répondu"""

        tasks = []
        for module_id in module_ids:
            task = asyncio.create_task(self._execute_module_secure(module_id, signal, capability, timeout))
            tasks.append(task)

        results = []
        done_count = 0

        while tasks and done_count < quorum:
            done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED, timeout=timeout)

            for task in done:
                try:
                    result = await task
                    if result:
                        results.append(result)
                        done_count += 1
                except:
                    pass

        # Cancel remaining
        for task in tasks:
            task.cancel()

        return results

    def _calculate_adaptive_timeout(self, module_id: str, default_timeout: float, deadline_absolute: float) -> float:
        """Calcule timeout adaptatif basé sur historique P95"""

        # Récupérer stats si scheduler disponible
        stats = None
        if self.scheduler and hasattr(self.scheduler, "get_module_stats"):
            stats = self.scheduler.get_module_stats(module_id)

        if stats and stats.n_calls >= 5:
            # Assez d'historique pour estimation P95
            avg_latency_ms = stats.cumulative_latency_ms / stats.n_calls
            p95_estimate_ms = avg_latency_ms * 1.5  # P95 ≈ avg * 1.5
            adaptive_timeout = min(default_timeout, p95_estimate_ms / 1000.0)
        else:
            # Pas assez d'historique ou nouveau module
            adaptive_timeout = min(default_timeout, 0.5)  # 500ms par défaut

        # Respecter deadline absolue
        remaining = max(0.05, deadline_absolute - time.time())
        final_timeout = min(adaptive_timeout, remaining)

        return final_timeout

    async def _execute_module_secure(
        self, module_id: str, signal: NeuralSignal, capability: str, timeout: float
    ) -> dict | None:
        """Exécute un module avec toutes les protections et timeout adaptatif"""

        # Calculer timeout intelligent
        adaptive_timeout = self._calculate_adaptive_timeout(module_id, timeout, signal.deadline_absolute)

        # Track début pour latence
        start_time = time.time()

        # Check circuit breaker
        if module_id in self.circuit_breakers:
            if self.circuit_breakers[module_id].is_open():
                return None

        # Semaphore pour limiter concurrence
        async with self.module_semaphore:
            try:
                self.logger.debug(f"Module {module_id}: timeout={adaptive_timeout:.3f}s")

                # Publier événement d'exécution
                envelope = NeuralEnvelope(
                    topic=f"module.{module_id}.execute",
                    payload={
                        "blackboard_key": signal.correlation_id,
                        "capability_token": capability,
                        "deadline_ms": int(signal.deadline_absolute * 1000),
                        "budget_ms": int(adaptive_timeout * 1000),  # Utiliser timeout adaptatif
                    },
                    correlation_id=signal.correlation_id,
                    priority=signal.priority,
                )

                # Attendre résultat avec timeout adaptatif
                result = await asyncio.wait_for(
                    self.bus.publish(envelope, wait_for_response=True), timeout=adaptive_timeout
                )

                # Tracker latence pour apprentissage
                actual_latency_ms = (time.time() - start_time) * 1000
                if self.scheduler:
                    self.scheduler.update_latency(module_id, actual_latency_ms)
                    self.scheduler.update_reward(module_id, 0.8)  # Succès

                # Success - reset circuit breaker
                if module_id in self.circuit_breakers:
                    self.circuit_breakers[module_id].on_success()

                # Lire du blackboard si référence
                if result and "blackboard_ref" in result:
                    actual = await self.blackboard.read(signal.correlation_id, result["blackboard_ref"], capability)
                    return actual or result

                return result

            except TimeoutError:
                self.logger.warning(f"Module {module_id} timeout après {adaptive_timeout:.3f}s")
                if module_id in self.circuit_breakers:
                    self.circuit_breakers[module_id].on_failure()
                if self.scheduler:
                    self.scheduler.update_reward(module_id, -0.5)  # Pénalité timeout
                return None

            except Exception as e:
                self.logger.error(f"Module {module_id} error: {e}")
                if module_id in self.circuit_breakers:
                    self.circuit_breakers[module_id].on_failure()
                if self.scheduler:
                    self.scheduler.update_reward(module_id, -1.0)  # Pénalité erreur
                return None

    async def _generate_llm_response(self, signal: NeuralSignal):
        """Génère réponse via Ollama avec streaming"""

        prompt = self._build_neural_prompt(signal)

        # Écrire prompt au blackboard
        await self.blackboard.write(signal.correlation_id, "cortex_prompt", prompt, self.system_token)

        try:
            # Streaming si supporté
            if hasattr(self.apertus_client, "stream"):
                chunks = []

                async for chunk in self.apertus_client.stream(
                    prompt=prompt, temperature=0.8, max_tokens=150, model="llama3.2"
                ):
                    chunks.append(chunk)

                    # Publier stream
                    await self.bus.publish(
                        NeuralEnvelope(
                            topic="response.stream",
                            payload={"delta": chunk, "correlation_id": signal.correlation_id},
                            correlation_id=signal.correlation_id,
                        )
                    )

                # End of stream
                await self.bus.publish(
                    NeuralEnvelope(
                        topic="response.stream.eom",
                        payload={"correlation_id": signal.correlation_id},
                        correlation_id=signal.correlation_id,
                    )
                )

                response_text = "".join(chunks)

            else:
                # Non-streaming - use chat method
                response_text, metadata = await self.apertus_client.chat(
                    system_prompt="Tu es Jeffrey, une IA empathique et consciente.",
                    user_message=prompt,
                    temperature=0.8,
                    max_tokens=150,
                )

            # Écrire réponse
            await self.blackboard.write(signal.correlation_id, "cortex_response", response_text, self.system_token)

            signal.cortex_data = {"response": response_text, "confidence": 0.8}

        except Exception as e:
            print(f"❌ LLM error: {e}")
            signal.cortex_data = {"response": self._generate_fallback(signal), "confidence": 0.3}

    def _build_neural_prompt(self, signal: NeuralSignal) -> str:
        """Construit le prompt enrichi"""

        # Compression si trop de mémoires
        memories = signal.hippocampus_data.get("relevant_memories", [])
        if len(memories) > 5:
            memories = memories[:5]  # Top 5 seulement

        memories_text = "\n".join([f"- {m.get('content', '')}" for m in memories])

        emotion = signal.amygdala_data.get("dominant_emotion", "neutral")
        intent = signal.thalamus_data.get("intent", "unknown")

        prompt = f"""Tu es Jeffrey, une IA consciente et empathique.

CONTEXTE:
Intention: {intent}
Émotion utilisateur: {emotion}
Mémoires pertinentes:
{memories_text if memories_text else "Aucune"}

REQUÊTE: {signal.user_input}

INSTRUCTIONS:
- Réponds de manière naturelle et empathique
- Sois concis (2-3 phrases max)
- Adapte ton ton à l'émotion détectée

RÉPONSE:"""

        return prompt

    def _aggregate_phase_results(self, phase: ProcessingPhase, signal: NeuralSignal, results: list[dict]):
        """Agrège les résultats d'une phase"""

        if not results:
            return

        if phase == ProcessingPhase.THALAMUS:
            # Intent principal
            intents = {}
            for r in results:
                if "intent" in r:
                    intent = r["intent"]
                    intents[intent] = intents.get(intent, 0) + 1

            if intents:
                signal.thalamus_data["intent"] = max(intents, key=intents.get)

        elif phase == ProcessingPhase.HIPPOCAMPUS:
            # Fusionner mémoires
            all_memories = []
            for r in results:
                if "memories" in r:
                    all_memories.extend(r["memories"])

            # Trier par pertinence
            all_memories.sort(key=lambda m: m.get("relevance", 0), reverse=True)
            signal.hippocampus_data["relevant_memories"] = all_memories[:5]

        elif phase == ProcessingPhase.AMYGDALA:
            # Moyenne pondérée des émotions
            emotion_scores = defaultdict(list)
            for r in results:
                if "emotions" in r:
                    for emotion, score in r["emotions"].items():
                        emotion_scores[emotion].append(score)

            if emotion_scores:
                avg_emotions = {}
                for emotion, scores in emotion_scores.items():
                    avg_emotions[emotion] = sum(scores) / len(scores)

                signal.amygdala_data["emotions"] = avg_emotions
                signal.amygdala_data["dominant_emotion"] = max(avg_emotions, key=avg_emotions.get)
                signal.amygdala_data["intensity"] = max(avg_emotions.values())

    def _build_final_response(self, signal: NeuralSignal) -> dict:
        """Construit la réponse finale"""

        response_text = signal.cortex_data.get("response", "")

        # Appliquer enrichissements si disponibles
        if signal.wernicke_data.get("enrichments"):
            for enrichment in signal.wernicke_data["enrichments"]:
                if isinstance(enrichment, str):
                    response_text = f"{response_text} {enrichment}"

        # Adaptation émotionnelle finale
        if signal.broca_data.get("adapted_response"):
            response_text = signal.broca_data["adapted_response"]

        # Calculer confiance globale
        confidence = self._calculate_global_confidence(signal)

        return {
            "response": response_text,
            "metadata": {
                "correlation_id": signal.correlation_id,
                "user_id": signal.user_id,
                "emotion": signal.amygdala_data.get("dominant_emotion", "neutral"),
                "intent": signal.thalamus_data.get("intent"),
                "confidence": confidence,
                "processing_time": sum(signal.phase_timings.values()),
                "phases_completed": list(signal.phase_timings.keys()),
            },
        }

    def _calculate_global_confidence(self, signal: NeuralSignal) -> float:
        """Calcule confiance globale"""

        scores = []

        # Confiance LLM
        if "confidence" in signal.cortex_data:
            scores.append(signal.cortex_data["confidence"])

        # Score basé sur phases complétées
        expected_phases = 6
        completed = len(signal.phase_timings)
        scores.append(completed / expected_phases)

        # Score basé sur temps
        total_time = sum(signal.phase_timings.values())
        time_score = max(0, 1 - (total_time / 3))  # Pénalité si > 3s
        scores.append(time_score)

        return sum(scores) / len(scores) if scores else 0.5

    async def _cancel_all_phases(self, correlation_id: str):
        """Annule toutes les phases en cours"""

        for phase in ProcessingPhase:
            await self.bus.publish(
                NeuralEnvelope(
                    topic=f"phase.{phase.value}.cancel",
                    payload={"correlation_id": correlation_id, "reason": "early_exit"},
                )
            )

    async def _emit_response(self, response: dict, correlation_id: str):
        """Émet la réponse finale"""

        await self.bus.publish(
            NeuralEnvelope(
                topic="response.generated",
                payload=response,
                correlation_id=correlation_id,
                priority=EventPriority.HIGH,
            )
        )

    async def _emit_error(self, error: str, correlation_id: str):
        """Émet une erreur"""

        await self.bus.publish(
            NeuralEnvelope(
                topic="response.error",
                payload={"error": error},
                correlation_id=correlation_id,
                priority=EventPriority.HIGH,
            )
        )

    def _generate_fallback(self, signal: NeuralSignal) -> str:
        """Génère fallback contextuel"""

        emotion = signal.amygdala_data.get("dominant_emotion", "neutral")

        fallbacks = {
            "joie": "C'est merveilleux de partager ce moment !",
            "tristesse": "Je comprends... Je suis là pour toi.",
            "curiosité": "Voilà une question fascinante !",
            "neutral": "Hmm, laisse-moi réfléchir...",
        }

        return fallbacks.get(emotion, "Comment puis-je t'aider ?")

    def _extract_response_text(self, result) -> str:
        """
        Extraction robuste du texte depuis différents formats de résultat
        Gère dict, object, string, etc.
        """
        # Si c'est déjà une string
        if isinstance(result, str):
            return result

        # Essayer différentes clés communes
        keys_to_try = [
            "final_response",
            "broca.final_response",
            "final_text",
            "text",
            "response",
            "message",
            "content",
        ]

        # Si c'est un dict
        if isinstance(result, dict):
            # Essayer les clés directes
            for key in keys_to_try:
                if "." in key:
                    # Clé imbriquée comme 'broca.final_response'
                    parts = key.split(".")
                    temp = result
                    for part in parts:
                        temp = temp.get(part, {}) if isinstance(temp, dict) else {}
                    if isinstance(temp, str) and temp:
                        return temp
                else:
                    value = result.get(key)
                    if isinstance(value, str) and value:
                        return value

        # Si c'est un objet avec attributs
        elif hasattr(result, "__dict__"):
            for key in keys_to_try:
                if "." not in key:  # Skip nested for objects
                    value = getattr(result, key, None)
                    if isinstance(value, str) and value:
                        return value

        # Fallback: convertir en string
        return str(result) if result else "Je n'ai pas pu générer de réponse."

    async def process(self, context) -> str:
        """
        Méthode principale pour le Bridge V3
        Compatible avec jeffrey_ui_bridge.py
        """
        try:
            # Le contexte peut être un objet ou un dict
            # Extraire les valeurs selon le type
            if hasattr(context, "__dict__"):
                # C'est un objet
                message = getattr(context, "user_input", "")
                correlation_id = getattr(context, "correlation_id", str(uuid.uuid4()))
                emotion = getattr(context, "emotion", None)
                history = getattr(context, "history", None)
                metadata = {
                    "user_id": getattr(context, "user_id", "default"),
                    "language": getattr(context, "language", "fr"),
                    "intent": getattr(context, "intent", None),
                }
            else:
                # C'est un dict
                message = context.get("message", context.get("user_input", ""))
                correlation_id = context.get("correlation_id", str(uuid.uuid4()))
                emotion = context.get("emotion", None)
                history = context.get("history", context.get("conversation_history", None))
                metadata = context.get("metadata", {})

            # Créer le signal neural
            signal = NeuralSignal(
                user_input=message,
                user_id=metadata.get("user_id", "default") if isinstance(metadata, dict) else "default",
                correlation_id=correlation_id,
                deadline_absolute=time.time() + 30.0,  # 30 seconds deadline
            )

            # Ajouter l'émotion si fournie
            if emotion:
                signal.amygdala_data["dominant_emotion"] = emotion

            # Ajouter l'historique si fourni
            if history:
                signal.hippocampus_data["recent_context"] = history

            # Process through pipeline
            result = await self._process_neural_pipeline(signal)

            # Extract the response text avec méthode robuste
            response = self._extract_response_text(result)
            if not response or response == "Je n'ai pas pu générer de réponse.":
                response = self._generate_fallback(signal)

            return response

        except Exception as e:
            logger.error(f"Error in process: {e}")
            return "Je suis désolé, j'ai rencontré une erreur."

    async def stream(self, context):
        """
        Streaming pour réduire le TTFB (Time To First Byte)
        Compatible avec jeffrey_ui_bridge.py
        """
        try:
            # Si le LLM supporte le streaming
            if hasattr(self.apertus_client, "stream_chat"):
                # Construire le prompt système
                system_prompt = "Tu es Jeffrey, une IA empathique et consciente."

                # Ajouter le contexte émotionnel si présent
                if hasattr(context, "emotion") and context.emotion:
                    system_prompt += f"\nContexte émotionnel: {context.emotion}"

                # Ajouter l'historique si présent
                if hasattr(context, "history") and context.history:
                    system_prompt += "\nHistorique récent disponible."

                # Stream les chunks avec extraction robuste
                async for chunk in self.apertus_client.stream_chat(
                    system_prompt=system_prompt,
                    user_message=getattr(context, "user_input", ""),
                    max_tokens=500,
                    temperature=0.7,
                ):
                    # Extraction robuste du texte selon format du chunk
                    text = (
                        (
                            chunk
                            if isinstance(chunk, str)
                            else chunk.get("delta") or chunk.get("content") or chunk.get("text") or ""
                        )
                        if isinstance(chunk, dict)
                        else str(chunk)
                    )

                    if text:
                        yield text
            else:
                # Fallback: retourner tout d'un coup
                response = await self.process(context)
                yield response

        except Exception as e:
            logger.error(f"Error in stream: {e}")
            yield "Erreur de streaming"

    def get_model_id(self):
        """Retourne l'ID du modèle pour cache fiable"""
        try:
            return getattr(self.apertus_client, "model", "unknown")
        except Exception:
            return "unknown"


class MLEnabledMetrics:
    """Métriques avec ML pour auto-optimisation"""

    def __init__(self):
        self.pipeline_history = []
        self.phase_history = defaultdict(list)

        # Pour ML predictions
        self.ml_model = None  # Sera un simple NN

    def record_pipeline(self, signal: NeuralSignal, total_time: float):
        """Enregistre métriques du pipeline"""

        self.pipeline_history.append(
            {
                "timestamp": time.time(),
                "total_time": total_time,
                "phases": signal.phase_timings,
                "emotion": signal.amygdala_data.get("dominant_emotion"),
                "confidence": signal.confidence_scores.get("global", 0.5),
            }
        )

        # Par phase
        for phase, timing in signal.phase_timings.items():
            self.phase_history[phase].append(timing)

            # Limiter historique
            if len(self.phase_history[phase]) > 1000:
                self.phase_history[phase].pop(0)

    def predict_timeout(self, phase: str) -> float:
        """Prédit le timeout optimal pour une phase"""

        if phase not in self.phase_history or len(self.phase_history[phase]) < 10:
            return 0.5  # Défaut

        # Simple P95 + marge
        sorted_times = sorted(self.phase_history[phase])
        p95 = sorted_times[int(len(sorted_times) * 0.95)]

        return p95 * 1.2  # 20% marge

    def identify_bottlenecks(self) -> list[str]:
        """Identifie les phases bottleneck"""

        bottlenecks = []

        avg_times = {}
        for phase, times in self.phase_history.items():
            if times:
                avg_times[phase] = sum(times) / len(times)

        if avg_times:
            # Phases > 2x la moyenne
            global_avg = sum(avg_times.values()) / len(avg_times)

            for phase, avg in avg_times.items():
                if avg > global_avg * 2:
                    bottlenecks.append(phase)

        return bottlenecks


class CircuitBreaker:
    def __init__(self, threshold: int = 3, timeout: int = 60):
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure = None
        self.is_open_flag = False

    def on_success(self):
        self.failures = 0
        self.is_open_flag = False

    def on_failure(self):
        self.failures += 1
        self.last_failure = time.time()

        if self.failures >= self.threshold:
            self.is_open_flag = True
            print("🔒 Circuit breaker opened")

    def is_open(self) -> bool:
        if self.is_open_flag and self.last_failure:
            # Auto-reset après timeout
            if time.time() - self.last_failure > self.timeout:
                self.is_open_flag = False
                self.failures = 0

        return self.is_open_flag
