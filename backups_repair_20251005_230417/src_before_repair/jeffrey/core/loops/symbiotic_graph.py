"""
Symbiotic Graph - Inter-loop dependencies with ML
Phase 2.2 implementation with fixes
"""

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any

# Try to import networkx, make it optional
try:
    import networkx as nx

    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False

logger = logging.getLogger(__name__)


@dataclass
class LoopInteraction:
    """Represents interaction between two loops"""

    source: str
    target: str
    weight: float  # Strength of interaction (-1 to 1)
    interaction_type: str  # 'synergy', 'conflict', 'neutral'
    timestamp: float
    confidence: float


class SymbioticGraph:
    """
    Manages inter-loop dependencies and predicts optimal interactions
    """

    def __init__(self, loop_manager):
        self.loop_manager = loop_manager
        self.interactions_history = deque(maxlen=1000)

        # Graph only if networkx available
        if HAS_NETWORKX:
            self.graph = nx.DiGraph()
            self._initialize_graph()
        else:
            self.graph = None
            logger.warning("networkx not available, graph features disabled")

        # Interaction rules (hardcoded initially, will learn later)
        self.interaction_rules = {
            ("awareness", "curiosity"): 0.7,  # High awareness boosts curiosity
            ("awareness", "emotional_decay"): 0.5,  # Awareness affects emotions
            ("emotional_decay", "curiosity"): -0.3,  # High emotion reduces curiosity
            ("memory_consolidation", "awareness"): 0.4,  # Memories boost awareness
            ("curiosity", "memory_consolidation"): 0.6,  # Curiosity generates memories
        }

        # Learned interactions
        self.learned_interactions = defaultdict(list)

    def _initialize_graph(self):
        """Initialize graph with all loops"""
        if self.graph is None or not self.loop_manager.loops:
            return

        for loop_name in self.loop_manager.loops.keys():
            if loop_name not in self.graph:
                self.graph.add_node(loop_name, cycles=0, health=1.0, last_update=time.time())

    async def analyze_interactions(self) -> dict[str, Any]:
        """
        Analyze current loop interactions and predict synergies
        """
        if self.graph is None:
            return {"error": "networkx not available"}

        # Check if graph has nodes
        node_count = self.graph.number_of_nodes()
        if node_count == 0:
            # Try to initialize nodes if not done
            self._initialize_graph()
            node_count = self.graph.number_of_nodes()

            if node_count == 0:
                return {
                    "error": "Graph not initialized (no loops found)",
                    "graph_metrics": {
                        "nodes": 0,
                        "edges": 0,
                        "density": 0.0,
                        "is_connected": False,
                    },
                }

        # Update node attributes with current metrics
        for loop_name, loop in self.loop_manager.loops.items():
            if loop_name in self.graph:
                metrics = loop.get_metrics() if hasattr(loop, "get_metrics") else {}
                self.graph.nodes[loop_name].update(
                    {
                        "cycles": metrics.get("cycles", 0),
                        "health": 1.0 - metrics.get("error_rate", 0),
                        "latency": metrics.get("p95_latency_ms", 0),
                        "last_update": time.time(),
                    }
                )

        # Detect interactions based on rules and history
        interactions = self._detect_interactions()

        # Classify interactions
        synergies = [i for i in interactions if i.weight > 0.5]
        conflicts = [i for i in interactions if i.weight < -0.3]

        # Check for cycles (potential deadlocks)
        cycles = []
        try:
            if node_count > 0:
                cycles = list(nx.simple_cycles(self.graph))
        except:
            pass

        # Generate recommendations
        recommendations = self._generate_recommendations(synergies, conflicts, cycles)

        # Calculate graph metrics safely
        is_connected = False
        density = 0.0

        if node_count > 0:
            density = nx.density(self.graph) if node_count > 1 else 0.0
            try:
                is_connected = nx.is_weakly_connected(self.graph) if node_count > 1 else True
            except:
                is_connected = node_count == 1

        return {
            "graph_metrics": {
                "nodes": node_count,
                "edges": self.graph.number_of_edges(),
                "density": density,
                "is_connected": is_connected,
            },
            "synergies": [self._interaction_to_dict(s) for s in synergies],
            "conflicts": [self._interaction_to_dict(c) for c in conflicts],
            "potential_deadlocks": cycles,
            "recommendations": recommendations,
        }

    def _detect_interactions(self) -> list[LoopInteraction]:
        """Detect interactions between loops"""
        interactions = []

        # Check predefined rules
        for (source, target), weight in self.interaction_rules.items():
            if source in self.graph and target in self.graph:
                interaction = LoopInteraction(
                    source=source,
                    target=target,
                    weight=weight,
                    interaction_type=self._classify_interaction(weight),
                    timestamp=time.time(),
                    confidence=0.8,
                )
                interactions.append(interaction)

                # Update graph edge
                self.graph.add_edge(source, target, weight=weight)

        # Check learned interactions from history
        for pair, weights in self.learned_interactions.items():
            if len(weights) >= 5:  # Need enough data
                avg_weight = sum(weights[-5:]) / 5
                source, target = pair

                interaction = LoopInteraction(
                    source=source,
                    target=target,
                    weight=avg_weight,
                    interaction_type=self._classify_interaction(avg_weight),
                    timestamp=time.time(),
                    confidence=min(0.9, len(weights) / 10),
                )
                interactions.append(interaction)

        return interactions

    def _classify_interaction(self, weight: float) -> str:
        """Classify interaction type based on weight"""
        if weight > 0.5:
            return "synergy"
        elif weight < -0.3:
            return "conflict"
        else:
            return "neutral"

    def _generate_recommendations(self, synergies: list, conflicts: list, cycles: list) -> list[str]:
        """Generate actionable recommendations"""
        recommendations = []

        # Synergy recommendations
        if synergies:
            top_synergy = max(synergies, key=lambda s: s.weight)
            recommendations.append(
                f"💫 Boost {top_synergy.target} when {top_synergy.source} is active (synergy: {top_synergy.weight:.2f})"
            )

        # Conflict recommendations
        if conflicts:
            top_conflict = min(conflicts, key=lambda c: c.weight)
            recommendations.append(
                f"⚠️ Throttle {top_conflict.target} if {top_conflict.source} has high load "
                f"(conflict: {top_conflict.weight:.2f})"
            )

        # Deadlock warnings
        if cycles:
            cycle_str = " → ".join(cycles[0]) if cycles[0] else ""
            recommendations.append(f"🔄 Potential deadlock detected: {cycle_str}")

        # Graph density recommendation
        if self.graph:
            density = nx.density(self.graph) if self.graph.number_of_nodes() > 1 else 0.0
            if density > 0.7:
                recommendations.append("📊 Consider reducing inter-loop dependencies (high coupling)")
            elif density < 0.2:
                recommendations.append("🔗 Loops are too isolated, consider adding synergies")

        return recommendations

    def _interaction_to_dict(self, interaction: LoopInteraction) -> dict:
        """Convert interaction to dict for serialization"""
        return {
            "source": interaction.source,
            "target": interaction.target,
            "weight": interaction.weight,
            "type": interaction.interaction_type,
            "confidence": interaction.confidence,
        }

    async def learn_from_metrics(self):
        """Learn interactions from actual metrics correlations"""
        if not self.loop_manager.symbiosis_history or len(self.loop_manager.symbiosis_history) < 2:
            return

        # Get recent history
        recent = self.loop_manager.symbiosis_history[-10:]

        # Analyze correlations between loop metrics
        for i in range(1, len(recent)):
            prev = recent[i - 1].get("loop_metrics", {})
            curr = recent[i].get("loop_metrics", {})

            # Check each pair of loops
            for loop1 in self.loop_manager.loops:
                for loop2 in self.loop_manager.loops:
                    if loop1 == loop2:
                        continue

                    # Simple correlation: if loop1 cycles increase, does loop2 perform better?
                    if loop1 in prev and loop2 in prev and loop1 in curr and loop2 in curr:
                        prev_cycles = prev[loop1].get("cycles", 0)
                        curr_cycles = curr[loop1].get("cycles", 0)
                        loop1_change = curr_cycles - prev_cycles

                        curr_error_rate = curr[loop2].get("error_rate", 0)
                        loop2_health = 1.0 - curr_error_rate

                        if loop1_change > 0:  # Loop1 was active
                            # Record correlation
                            weight = loop2_health * 2 - 1  # Convert to -1 to 1
                            self.learned_interactions[(loop1, loop2)].append(weight)

                            # Keep history limited
                            if len(self.learned_interactions[(loop1, loop2)]) > 100:
                                self.learned_interactions[(loop1, loop2)].pop(0)
