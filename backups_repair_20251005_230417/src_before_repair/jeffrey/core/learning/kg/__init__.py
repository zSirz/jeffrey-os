"""
Knowledge Graph Adapter Interface - Preparation for P3
Currently uses in-memory implementation, will switch to Neo4j in P3
"""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class KnowledgeGraphProvider(ABC):
    """Abstract interface for Knowledge Graph operations"""

    @abstractmethod
    async def add_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        pass

    @abstractmethod
    async def add_edge(
        self,
        source: str,
        target: str,
        relationship: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        pass

    @abstractmethod
    async def query(self, cypher: str) -> list[dict]:
        pass

    @abstractmethod
    async def find_path(self, start: str, end: str, max_depth: int = 5) -> list[str] | None:
        pass


class InMemoryKnowledgeGraph(KnowledgeGraphProvider):
    """In-memory implementation for development"""

    def __init__(self):
        self.nodes = {}
        self.edges = []

    async def add_node(self, node_id: str, properties: dict[str, Any]) -> bool:
        self.nodes[node_id] = properties
        return True

    async def add_edge(
        self,
        source: str,
        target: str,
        relationship: str,
        properties: dict[str, Any] | None = None,
    ) -> bool:
        self.edges.append(
            {
                "source": source,
                "target": target,
                "relationship": relationship,
                "properties": properties or {},
            }
        )
        return True

    async def query(self, cypher: str) -> list[dict]:
        # Stub for memory implementation
        return []

    async def find_path(self, start: str, end: str, max_depth: int = 5) -> list[str] | None:
        # Simple BFS for memory implementation
        if start not in self.nodes or end not in self.nodes:
            return None
        # Simplified - would implement actual pathfinding
        return [start, end] if start != end else [start]


# Factory function
def get_knowledge_graph() -> KnowledgeGraphProvider:
    """Factory to get the appropriate KG provider based on config"""
    provider = os.getenv("KG_PROVIDER", "memory")

    if provider == "memory":
        return InMemoryKnowledgeGraph()
    elif provider == "neo4j":
        # Will be implemented in P3
        raise NotImplementedError("Neo4j provider will be available in P3")
    else:
        raise ValueError(f"Unknown KG provider: {provider}")
