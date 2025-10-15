"""
Module Registry - Bridge between services and ServiceRegistry
"""

from typing import Any


class ModuleRegistry:
    """
    Registry wrapper that provides async interface expected by modules
    """

    def __init__(self):
        self.modules: dict[str, dict[str, Any]] = {}
        self.topic_map: dict[str, list[str]] = {}

    async def register(
        self,
        name: str,
        module: Any,
        topics_in: list[str] = None,
        topics_out: list[str] = None,
        metadata: dict[str, Any] = None,
    ) -> None:
        """Register a module with topic information"""

        self.modules[name] = {
            "instance": module,
            "topics_in": topics_in or [],
            "topics_out": topics_out or [],
            "metadata": metadata or {},
            "status": "active",
        }

        # Update topic map
        for topic in topics_in or []:
            if topic not in self.topic_map:
                self.topic_map[topic] = []
            self.topic_map[topic].append(name)

    def get(self, name: str) -> Any | None:
        """Get a registered module"""
        if name in self.modules:
            return self.modules[name]["instance"]
        return None

    def get_all(self) -> dict[str, Any]:
        """Get all registered modules"""
        return {name: data["instance"] for name, data in self.modules.items()}

    def get_topics_for(self, module_name: str) -> dict[str, list[str]]:
        """Get topics for a module"""
        if module_name in self.modules:
            return {
                "in": self.modules[module_name]["topics_in"],
                "out": self.modules[module_name]["topics_out"],
            }
        return {"in": [], "out": []}

    def check_orphans(self) -> list[str]:
        """Check for orphaned modules (no topic connections)"""
        orphans = []
        for name, data in self.modules.items():
            if not data["topics_in"] and not data["topics_out"]:
                orphans.append(name)
        return orphans

    def __contains__(self, name: str) -> bool:
        """Check if a module is registered"""
        return name in self.modules

    def __len__(self) -> int:
        """Get number of registered modules"""
        return len(self.modules)
