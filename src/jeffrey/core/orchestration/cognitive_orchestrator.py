"""
CognitiveOrchestrator - The brain's conductor
Implements Grok's auto-healing and Gemini's Blackboard pattern
"""
import asyncio
import logging
import time
import json
from typing import Dict, List, Any, Callable, Optional
from collections import defaultdict

# GPT CORRECTION 1: dépendances optionnelles
try:
    import psutil  # type: ignore
except Exception:
    psutil = None

try:
    import networkx as nx  # type: ignore
except Exception:
    nx = None

logger = logging.getLogger(__name__)

class CognitiveOrchestrator:
    """
    Master orchestrator implementing:
    - Blackboard architecture (Gemini's vision)
    - Auto-healing and adaptation (Grok's innovation)
    - Priority-based routing
    - Connection graph visualization
    - Predictive monitoring
    """

    def __init__(self, bus, config: Dict[str, Any] = None):
        self.bus = bus
        self.config = config or {}

        # Agent registry (name -> agent instance)
        self.agents = {}

        # Handler registry (topic -> [(priority, handler)])
        self.handlers = defaultdict(list)

        # Health tracking
        self.health = {}  # handler -> {last_success, errors, latency}

        # Connection graph for visualization (with fallback)
        self.graph = nx.DiGraph() if nx else None

        # Auto-healing configuration
        self.auto_heal_interval = self.config.get("auto_heal_interval", 60)
        self.error_threshold = self.config.get("error_threshold", 5)
        self.inactive_threshold = self.config.get("inactive_threshold", 300)

        # Predictive monitoring
        self.predictions = {}
        self._monitoring_task = None

        # Circadian integration
        self.circadian_state = {"phase": "day", "energy": 1.0}

        # Stats
        self.stats = {
            "agents_registered": 0,
            "handlers_registered": 0,
            "healings_performed": 0,
            "predictions_made": 0,
            "graph_nodes": 0,
            "graph_edges": 0
        }

    def register_agent(self, name: str, agent: Any) -> None:
        """Register a cognitive agent"""
        self.agents[name] = agent
        self.stats["agents_registered"] = len(self.agents)

        # Add to graph (with fallback)
        if self.graph:
            self.graph.add_node(name, type="agent", instance=agent)
        logger.info(f"🤖 Registered agent: {name}")

    def register_handler(
        self,
        topic: str,
        handler: Callable,
        priority: int = 5,
        source_agent: Optional[str] = None
    ) -> Callable:
        """
        Register event handler with priority
        Returns unsubscribe function
        """
        # Add to registry
        self.handlers[topic].append((priority, handler, source_agent))
        self.handlers[topic].sort(key=lambda x: x[0], reverse=True)

        # Initialize health tracking
        self.health[handler] = {
            "last_success": time.time(),
            "errors": 0,
            "total_calls": 0,
            "avg_latency_ms": 0
        }

        # Update graph (with fallback)
        if source_agent and self.graph:
            self.graph.add_edge(topic, source_agent, weight=priority)

        self.stats["handlers_registered"] = sum(len(h) for h in self.handlers.values())
        self._update_graph_stats()

        logger.debug(f"📌 Registered handler for {topic} with priority {priority}")

        # Return unsubscribe function
        def unsubscribe():
            self.handlers[topic] = [
                (p, h, s) for p, h, s in self.handlers[topic]
                if h != handler
            ]
            if handler in self.health:
                del self.health[handler]
            self._update_graph_stats()

        return unsubscribe

    async def dispatch(self, topic: str, event: Dict) -> None:
        """
        Dispatch event to handlers with health-aware routing
        """
        handlers = self.handlers.get(topic, [])

        if not handlers:
            logger.debug(f"No handlers for topic: {topic}")
            return

        # Filter healthy handlers based on energy level
        energy = self.circadian_state.get("energy", 1.0)
        max_handlers = max(1, int(len(handlers) * energy))  # Process fewer when tired

        healthy_handlers = []
        for priority, handler, source in handlers[:max_handlers]:
            health = self.health.get(handler, {})

            # Skip if too many errors
            if health.get("errors", 0) >= self.error_threshold:
                logger.warning(f"Skipping unhealthy handler (errors={health['errors']})")
                continue

            healthy_handlers.append((priority, handler, source))

        # Process with healthy handlers
        tasks = []
        for priority, handler, source in healthy_handlers:
            task = asyncio.create_task(
                self._execute_handler(handler, event, source)
            )
            tasks.append(task)

        # Wait for all with timeout
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_handler(
        self,
        handler: Callable,
        event: Dict,
        source: Optional[str] = None
    ) -> None:
        """Execute single handler with monitoring"""
        start_time = time.perf_counter()
        health = self.health[handler]

        try:
            health["total_calls"] += 1

            # Execute handler
            result = handler(event)
            if asyncio.iscoroutine(result):
                await result

            # Update health on success
            latency_ms = (time.perf_counter() - start_time) * 1000
            health["last_success"] = time.time()
            health["errors"] = 0  # Reset on success

            # Update average latency (simple moving average)
            health["avg_latency_ms"] = (
                health["avg_latency_ms"] * 0.9 + latency_ms * 0.1
            )

            logger.debug(f"✅ Handler executed in {latency_ms:.1f}ms")

        except Exception as e:
            # Update health on failure
            health["errors"] += 1
            logger.error(f"Handler failed: {e}")

            # Auto-disable if too many errors
            if health["errors"] >= self.error_threshold:
                logger.error(f"⚠️ Handler disabled after {health['errors']} errors")
                # Will be skipped in future dispatches

    def check_system_load(self) -> bool:
        """GPT CORRECTION: check load with psutil fallback"""
        try:
            if psutil is None:
                return False  # pas de switch auto si pas dispo
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent

            if cpu_percent > 80 or memory_percent > 85:
                logger.warning(f"High system load (CPU: {cpu_percent}%, MEM: {memory_percent}%)")
                return True
            return False
        except Exception as e:
            logger.warning(f"Load check failed: {e}")
            return False

    async def auto_heal(self) -> None:
        """
        Auto-healing loop that:
        - Revives inactive handlers
        - Adjusts priorities based on performance
        - Prunes unhealthy handlers
        """
        while True:
            await asyncio.sleep(self.auto_heal_interval)

            try:
                now = time.time()
                healings = 0

                # Check system load
                self.check_system_load()

                # Check each handler's health
                for handler, health in list(self.health.items()):
                    time_since_success = now - health.get("last_success", 0)

                    # Revive inactive handlers
                    if time_since_success > self.inactive_threshold:
                        if health["errors"] > 0:
                            # Reset errors for retry
                            health["errors"] = 0
                            health["last_success"] = now
                            healings += 1
                            logger.info(f"🔧 Revived inactive handler (was inactive for {time_since_success:.0f}s)")

                    # Adjust priorities based on performance
                    if health["avg_latency_ms"] > 100:
                        # Find and lower priority of slow handlers
                        for topic, handlers in self.handlers.items():
                            for i, (p, h, s) in enumerate(handlers):
                                if h == handler and p > 1:
                                    handlers[i] = (p - 1, h, s)
                                    logger.debug(f"Lowered priority of slow handler")

                if healings > 0:
                    self.stats["healings_performed"] += healings
                    logger.info(f"🏥 Auto-healing performed {healings} operations")

            except Exception as e:
                logger.error(f"Auto-healing failed: {e}")

    async def predict_issues(self) -> Dict[str, Any]:
        """
        Predictive monitoring using simple heuristics
        (Could be enhanced with ML)
        """
        predictions = {
            "high_error_risk": [],
            "performance_degradation": [],
            "inactive_risk": []
        }

        now = time.time()

        for handler, health in self.health.items():
            # Predict error escalation
            if health["errors"] >= self.error_threshold * 0.6:
                predictions["high_error_risk"].append({
                    "handler": str(handler),
                    "current_errors": health["errors"],
                    "threshold": self.error_threshold
                })

            # Predict performance issues
            if health["avg_latency_ms"] > 50 and health["total_calls"] > 10:
                predictions["performance_degradation"].append({
                    "handler": str(handler),
                    "avg_latency_ms": health["avg_latency_ms"]
                })

            # Predict inactivity
            time_since_success = now - health.get("last_success", now)
            if time_since_success > self.inactive_threshold * 0.7:
                predictions["inactive_risk"].append({
                    "handler": str(handler),
                    "inactive_seconds": time_since_success
                })

        self.stats["predictions_made"] += 1
        self.predictions = predictions

        return predictions

    async def handle_dlq_event(self, dlq_entry: Dict) -> None:
        """
        Handle Dead Letter Queue events
        Could implement retry strategies or alerting
        """
        logger.warning(f"📮 DLQ event received: {dlq_entry.get('error')}")

        # Could implement:
        # - Retry with different handler
        # - Alert monitoring system
        # - Store for manual inspection

    def update_circadian_state(self, state: Dict) -> None:
        """Update circadian context for adaptive behavior"""
        self.circadian_state = state
        logger.debug(f"🌙 Circadian update: phase={state.get('phase')}, energy={state.get('energy')}")

    async def get_circadian_state(self) -> Dict:
        """Get current circadian state"""
        return self.circadian_state

    def _update_graph_stats(self):
        """Update graph statistics - GPT CORRECTION for networkx fallback"""
        if not self.graph:
            self.stats["graph_nodes"] = 0
            self.stats["graph_edges"] = 0
            return
        self.stats["graph_nodes"] = self.graph.number_of_nodes()
        self.stats["graph_edges"] = self.graph.number_of_edges()

    def export_graph(self, format: str = "json") -> str:
        """Export connection graph for visualization - GPT CORRECTION for networkx fallback"""
        if not nx or not self.graph:
            return "{}"

        if format == "json":
            return json.dumps(nx.node_link_data(self.graph), indent=2)
        elif format == "dot":
            try:
                return nx.nx_agraph.to_agraph(self.graph).to_string()
            except:
                return "digraph {}"  # Fallback
        else:
            raise ValueError(f"Unsupported format: {format}")

    def get_stats(self) -> Dict:
        """Get comprehensive statistics"""
        return {
            **self.stats,
            "handlers": {
                topic: len(handlers) for topic, handlers in self.handlers.items()
            },
            "health_summary": {
                "healthy": sum(1 for h in self.health.values() if h["errors"] == 0),
                "warning": sum(1 for h in self.health.values() if 0 < h["errors"] < self.error_threshold),
                "critical": sum(1 for h in self.health.values() if h["errors"] >= self.error_threshold)
            },
            "predictions": self.predictions,
            "circadian": self.circadian_state
        }

    async def start(self):
        """Start orchestrator services"""
        # Start auto-healing
        asyncio.create_task(self.auto_heal())

        # Start predictive monitoring
        async def monitor_loop():
            while True:
                await self.predict_issues()
                await asyncio.sleep(30)

        self._monitoring_task = asyncio.create_task(monitor_loop())

        logger.info("🎼 Cognitive Orchestrator started")

    async def stop(self):
        """Stop orchestrator services"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        logger.info("🛑 Cognitive Orchestrator stopped")