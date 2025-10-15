"""
Module système pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module système pour jeffrey os.
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

import graphlib
import logging
from dataclasses import dataclass


@dataclass
class ModuleNode:
    """Nœud avec métriques évolutives"""

    name: str
    category: str
    complexity: int = 0
    health_score: float = 1.0
    success_count: int = 0
    failure_count: int = 0

    @property
    def fitness(self) -> float:
        """Fitness basée sur succès/échecs"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total


@dataclass
class DependencyEdge:
    """Arête avec force évolutive"""

    source: str
    target: str
    strength: float = 0.5
    usage_count: int = 0


class SimpleEvolutionaryGraph:
    """Graphe évolutif avec pruning"""

    def __init__(self, pruning_threshold: float = 0.2) -> None:
        self.nodes: dict[str, ModuleNode] = {}
        self.edges: dict[tuple[str, str], DependencyEdge] = {}
        self.graph = {}
        self.pruning_threshold = pruning_threshold
        self.logger = logging.getLogger("graph.engine")

        # Métriques de symbiose
        self.symbiosis_metrics = {
            "total_connections": 0,
            "pruned_count": 0,
            "avg_strength": 0.5,
        }

    def add_module(self, node: ModuleNode):
        """Ajouter un module"""
        self.nodes[node.name] = node
        if node.name not in self.graph:
            self.graph[node.name] = set()

    def add_dependency(self, source: str, target: str, initial_strength: float = 0.5):
        """Ajouter une dépendance"""

        if source not in self.nodes or target not in self.nodes:
            return

        edge = DependencyEdge(source, target, initial_strength)
        self.edges[(source, target)] = edge

        if source not in self.graph:
            self.graph[source] = set()
        self.graph[source].add(target)

        self.symbiosis_metrics["total_connections"] += 1

    def update_edge_feedback(self, source: str, target: str, success: bool, delta: float = 0.1):
        """Mise à jour du feedback"""

        key = (source, target)
        if key not in self.edges:
            return

        edge = self.edges[key]

        if success:
            edge.strength = min(1.0, edge.strength + delta)
            self.nodes[source].success_count += 1
            self.nodes[target].success_count += 1
        else:
            edge.strength = max(0.0, edge.strength - delta)
            self.nodes[source].failure_count += 1
            self.nodes[target].failure_count += 1

        edge.usage_count += 1

        # Pruning si trop faible
        if edge.strength < self.pruning_threshold:
            self.logger.info(f"🔪 Pruning weak edge {source}->{target} (strength={edge.strength:.2f})")
            self.remove_dependency(source, target)
            self.symbiosis_metrics["pruned_count"] += 1

        # Mettre à jour métrique moyenne
        if self.edges:
            self.symbiosis_metrics["avg_strength"] = sum(e.strength for e in self.edges.values()) / len(self.edges)

    def remove_dependency(self, source: str, target: str):
        """Retirer une dépendance"""

        key = (source, target)
        if key in self.edges:
            del self.edges[key]

        if source in self.graph:
            self.graph[source].discard(target)
            self.symbiosis_metrics["total_connections"] -= 1

    def topological_sort(self) -> list[str]:
        """Tri topologique avec gestion des cycles"""

        try:
            sorter = graphlib.TopologicalSorter(self.graph)
            return list(sorter.static_order())
        except graphlib.CycleError as e:
            self.logger.error(f"❌ Cycle detected: {e}")

            if self._break_weakest_cycle_edge():
                return self.topological_sort()
            else:
                raise

    def _break_weakest_cycle_edge(self) -> bool:
        """Casser le cycle en retirant l'arête la plus faible"""

        weakest_edge = None
        min_strength = 1.0

        for (source, target), edge in self.edges.items():
            if edge.strength < min_strength:
                min_strength = edge.strength
                weakest_edge = (source, target)

        if weakest_edge:
            self.logger.info(f"Breaking cycle: removing {weakest_edge[0]}->{weakest_edge[1]}")
            self.remove_dependency(weakest_edge[0], weakest_edge[1])
            return True

        return False

    def optimize(self):
        """Optimisation du graphe"""

        edges_to_remove = []

        for key, edge in self.edges.items():
            if edge.usage_count == 0 and edge.strength < 0.5:
                edges_to_remove.append(key)

        for source, target in edges_to_remove:
            self.remove_dependency(source, target)

        if edges_to_remove:
            self.logger.info(f"🧹 Optimized: removed {len(edges_to_remove)} unused edges")

    def get_metrics(self) -> dict:
        """Métriques incluant symbiose"""

        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "avg_fitness": (sum(n.fitness for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0),
            **self.symbiosis_metrics,
        }
