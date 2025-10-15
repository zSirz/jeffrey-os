"""
NeuralBus Registry - Chargeur de configuration YAML
"""

import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Registry centralisé pour tous les services Jeffrey OS"""

    def __init__(self):
        self.services: dict[str, dict[str, Any]] = {}
        self.events: dict[str, list[dict[str, Any]]] = {}
        self.routing_rules: list[dict[str, Any]] = []
        self.config: dict[str, Any] = {}

    def load_registry(self, path: str = None) -> bool:
        """
        Charge le registre depuis le fichier YAML

        Args:
            path: Chemin vers registry.yaml

        Returns:
            True si chargé avec succès
        """
        if not path:
            # Utiliser le chemin par défaut
            current_dir = Path(__file__).parent
            path = current_dir / "registry.yaml"

        try:
            with open(path) as f:
                data = yaml.safe_load(f)

            # Charger les services
            self.services = data.get("services", {})

            # Charger les types d'événements
            self.events = data.get("events", {})

            # Charger les règles de routage
            self.routing_rules = data.get("routing", {}).get("rules", [])

            # Charger la configuration globale
            self.config = {
                "version": data.get("version"),
                "updated": data.get("updated"),
                "security": data.get("security", {}),
                "routing": data.get("routing", {}),
            }

            logger.info(
                f"✅ Registry loaded: {len(self.services)} services, "
                f"{sum(len(v) for v in self.events.values())} event types"
            )
            return True

        except FileNotFoundError:
            logger.error(f"Registry file not found: {path}")
            return False
        except yaml.YAMLError as e:
            logger.error(f"Failed to parse registry YAML: {e}")
            return False
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return False

    def get_service(self, service_id: str) -> dict[str, Any] | None:
        """
        Récupère les informations d'un service

        Args:
            service_id: ID du service

        Returns:
            Configuration du service ou None
        """
        return self.services.get(service_id)

    def get_services_by_type(self, service_type: str) -> list[dict[str, Any]]:
        """
        Récupère tous les services d'un certain type

        Args:
            service_type: Type de service (security, guardian, etc.)

        Returns:
            Liste des services
        """
        return [
            {**config, "id": service_id}
            for service_id, config in self.services.items()
            if config.get("type") == service_type
        ]

    def get_critical_services(self) -> list[str]:
        """
        Récupère la liste des services critiques

        Returns:
            Liste des IDs de services critiques
        """
        return [service_id for service_id, config in self.services.items() if config.get("priority") == "critical"]

    def get_service_dependencies(self, service_id: str) -> list[str]:
        """
        Récupère les dépendances d'un service

        Args:
            service_id: ID du service

        Returns:
            Liste des dépendances
        """
        service = self.services.get(service_id, {})
        return service.get("dependencies", [])

    def get_event_types(self, category: str | None = None) -> list[dict[str, Any]]:
        """
        Récupère les types d'événements

        Args:
            category: Catégorie d'événements (optionnel)

        Returns:
            Liste des types d'événements
        """
        if category:
            return self.events.get(category, [])
        else:
            # Retourner tous les événements
            all_events = []
            for cat_events in self.events.values():
                all_events.extend(cat_events)
            return all_events

    def get_routing_rules(self, pattern: str | None = None) -> list[dict[str, Any]]:
        """
        Récupère les règles de routage

        Args:
            pattern: Pattern à matcher (optionnel)

        Returns:
            Liste des règles de routage
        """
        if not pattern:
            return self.routing_rules

        # Filtrer par pattern
        import re

        matching_rules = []
        for rule in self.routing_rules:
            rule_pattern = rule.get("pattern", "")
            if re.match(rule_pattern.replace("*", ".*"), pattern):
                matching_rules.append(rule)

        return matching_rules

    def validate_health(self) -> dict[str, Any]:
        """
        Valide la santé du registry

        Returns:
            Rapport de santé
        """
        health = {
            "status": "healthy",
            "services_loaded": len(self.services),
            "event_types": sum(len(v) for v in self.events.values()),
            "routing_rules": len(self.routing_rules),
            "critical_services": self.get_critical_services(),
            "issues": [],
        }

        # Vérifier les dépendances circulaires
        for service_id in self.services:
            if self._has_circular_dependency(service_id):
                health["issues"].append(f"Circular dependency detected: {service_id}")
                health["status"] = "degraded"

        # Vérifier les services critiques sans health check
        for service_id in health["critical_services"]:
            service = self.services[service_id]
            if not service.get("health_check"):
                health["issues"].append(f"Critical service without health check: {service_id}")
                health["status"] = "degraded"

        return health

    def _has_circular_dependency(self, service_id: str, visited: set | None = None) -> bool:
        """
        Détecte les dépendances circulaires

        Args:
            service_id: ID du service
            visited: Services déjà visités

        Returns:
            True si dépendance circulaire détectée
        """
        if visited is None:
            visited = set()

        if service_id in visited:
            return True

        visited.add(service_id)

        dependencies = self.get_service_dependencies(service_id)
        for dep in dependencies:
            if dep in self.services:  # Ignorer les dépendances externes
                if self._has_circular_dependency(dep, visited.copy()):
                    return True

        return False

    def get_status(self) -> dict[str, Any]:
        """Retourne le statut du registry"""
        return {
            "version": self.config.get("version"),
            "updated": self.config.get("updated"),
            "services": len(self.services),
            "event_types": sum(len(v) for v in self.events.values()),
            "routing_rules": len(self.routing_rules),
            "health": self.validate_health(),
        }


# Instance globale
registry = ServiceRegistry()
