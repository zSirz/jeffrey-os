"""
Module d'initialisation émotionnelle pour Jeffrey.

Ce module gère l'initialisation et la configuration des composants
émotionnels du système Jeffrey.
"""

import importlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

# Configuration du logger
logger = logging.getLogger(__name__)


class InitializationStatus(Enum):
    """Statuts d'initialisation possibles."""

    NON_INITIALISE = "non_initialise"
    EN_COURS = "en_cours"
    REUSSI = "reussi"
    ECHEC = "echec"


@dataclass
class ComponentStatus:
    """État d'initialisation d'un composant."""

    name: str
    status: InitializationStatus
    timestamp: datetime
    error: str | None = None
    dependencies: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convertit le statut en dictionnaire pour la sérialisation."""
        return {
            "name": self.name,
            "status": self.status.value,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
            "dependencies": self.dependencies or [],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ComponentStatus":
        """Crée un statut à partir d'un dictionnaire."""
        return cls(
            name=data["name"],
            status=InitializationStatus(data["status"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            error=data.get("error"),
            dependencies=data.get("dependencies", []),
        )


class EmotionalInitializer:
    """
    Gestionnaire d'initialisation des composants émotionnels.

    Cette classe gère l'initialisation et la configuration de tous
    les composants émotionnels du système Jeffrey.
    """

    def __init__(self, config_path: Path | None = None, components_path: Path | None = None):
        """
        Initialise le gestionnaire d'initialisation.

        Args:
            config_path: Chemin vers le fichier de configuration
            components_path: Chemin vers le dossier des composants
        """
        self.config_path = config_path
        self.components_path = components_path

        self.components: dict[str, ComponentStatus] = {}
        self.config: dict[str, Any] = {}
        self.initialization_order: list[str] = []

        # Initialisation du logger
        self._setup_logging()

        # Chargement de la configuration
        if config_path and config_path.exists():
            self._load_config()
        else:
            self._initialize_default_config()

    def _setup_logging(self) -> None:
        """Configure le logging pour le module."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    def _load_config(self) -> None:
        """Charge la configuration depuis le fichier."""
        try:
            with open(str(self.config_path)) as f:
                self.config = json.load(f)
            logger.info("Configuration chargée avec succès")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de la configuration: {e}")
            self._initialize_default_config()

    def _initialize_default_config(self) -> None:
        """Initialise la configuration par défaut."""
        self.config = {
            "components": {
                "emotional_sync": {
                    "enabled": True,
                    "dependencies": [],
                    "config": {"sync_interval": 0.5, "history_size": 100},
                },
                "emotional_effects": {
                    "enabled": True,
                    "dependencies": ["emotional_sync"],
                    "config": {"max_concurrent_effects": 5},
                },
                "emotional_handlers": {
                    "enabled": True,
                    "dependencies": ["emotional_sync"],
                    "config": {"handler_timeout": 1.0},
                },
                "emotional_interfaces": {
                    "enabled": True,
                    "dependencies": ["emotional_effects"],
                    "config": {"update_interval": 0.1},
                },
                "emotional_visuals": {
                    "enabled": True,
                    "dependencies": ["emotional_effects"],
                    "config": {"max_visual_effects": 3},
                },
                "emotional_logic": {
                    "enabled": True,
                    "dependencies": ["emotional_sync", "emotional_handlers"],
                    "config": {"logic_update_interval": 0.2},
                },
            },
            "initialization": {"retry_attempts": 3, "retry_delay": 1.0, "timeout": 10.0},
        }

        if self.config_path:
            try:
                with open(self.config_path, "w") as f:
                    json.dump(self.config, f, indent=2)
                logger.info("Configuration par défaut sauvegardée")
            except Exception as e:
                logger.error(f"Erreur lors de la sauvegarde de la configuration: {e}")

    def _determine_initialization_order(self) -> None:
        """Détermine l'ordre d'initialisation des composants."""
        # Créer un graphe de dépendances
        graph = {}
        for name, component in self.config["components"].items():
            if component.get("enabled", False):
                graph[name] = set(component.get("dependencies", []))

        # Algorithme de tri topologique
        visited = set()
        temp = set()
        order: list[str] = []

        def visit(node: str) -> None:
            if node in temp:
                raise ValueError(f"Dépendance cyclique détectée: {node}")
            if node in visited:
                return

            temp.add(node)
            for dependency in graph.get(node, set()):
                if dependency in graph:  # Vérifier que la dépendance existe
                    visit(dependency)
            temp.remove(node)
            visited.add(node)
            order.append(node)

        # Visiter tous les nœuds
        for node in graph:
            if node not in visited:
                visit(node)

        self.initialization_order = order
        logger.info(f"Ordre d'initialisation déterminé: {order}")

    def initialize_component(self, component_name: str, force: bool = False) -> bool:
        """
        Initialise un composant spécifique.

        Args:
            component_name: Nom du composant à initialiser
            force: Force la réinitialisation même si déjà initialisé

        Returns:
            bool: True si l'initialisation a réussi
        """
        if component_name not in self.config["components"]:
            logger.error(f"Composant inconnu: {component_name}")
            return False

        component_config = self.config["components"][component_name]
        if not component_config.get("enabled", False):
            logger.info(f"Composant désactivé: {component_name}")
            return False

        # Vérifier si déjà initialisé
        current_status = self.components.get(component_name)
        if current_status and current_status.status == InitializationStatus.REUSSI and not force:
            logger.info(f"Composant déjà initialisé: {component_name}")
            return True

        # Mettre à jour le statut
        self.components[component_name] = ComponentStatus(
            name=component_name,
            status=InitializationStatus.EN_COURS,
            timestamp=datetime.now(),
            dependencies=component_config.get("dependencies", []),
        )

        try:
            # Vérifier les dépendances
            for dependency in component_config.get("dependencies", []):
                if not self.initialize_component(dependency):
                    raise RuntimeError(f"Dépendance non initialisée: {dependency}")

            # Importer et initialiser le composant
            module_path = f"jeffrey.core.emotions.{component_name}"
            class_name = "".join(part.capitalize() for part in component_name.split("_"))

            module = importlib.import_module(module_path)
            component_class = getattr(module, class_name)

            # Instancier avec la config du composant (par défaut dictionnaire vide)
            cfg = component_config.get("config", {}) or {}
            component_class(**cfg)

            # Mettre à jour le statut
            self.components[component_name].status = InitializationStatus.REUSSI
            logger.info(f"Composant initialisé avec succès: {component_name}")
            return True

        except ImportError as e:
            err = f"Erreur d'importation: {e}"
        except AttributeError as e:
            err = f"Classe non trouvée: {e}"
        except Exception as e:
            err = str(e)

        # Gestion des erreurs centralisée
        self.components[component_name].status = InitializationStatus.ECHEC
        self.components[component_name].error = err
        logger.error("Échec de l'initialisation: %s - %s", component_name, err)
        return False

    def initialize_all(self, force: bool = False) -> bool:
        """
        Initialise tous les composants dans l'ordre correct.

        Args:
            force: Force la réinitialisation même si déjà initialisés

        Returns:
            bool: True si tous les composants ont été initialisés avec succès
        """
        # Déterminer l'ordre d'initialisation
        self._determine_initialization_order()

        # Initialiser chaque composant
        success = True
        for component_name in self.initialization_order:
            if not self.initialize_component(component_name, force):
                success = False
                # Continuer avec les autres composants même en cas d'échec

        return success

    def get_component_status(
        self, component_name: str | None = None
    ) -> ComponentStatus | dict[str, ComponentStatus] | None:
        """
        Retourne le statut d'initialisation d'un ou tous les composants.

        Args:
            component_name: Nom du composant (optionnel)

        Returns:
            Statut du composant ou dictionnaire de tous les statuts
        """
        if component_name:
            return self.components.get(component_name)
        return self.components.copy()

    def resumer(self) -> dict[str, Any]:
        """
        Génère un résumé de l'état d'initialisation.

        Returns:
            Dictionnaire contenant le résumé
        """
        return {
            "total_components": len(self.config.get("components", {})),
            "initialized": len([c for c in self.components.values() if c.status == InitializationStatus.REUSSI]),
            "failed": len([c for c in self.components.values() if c.status == InitializationStatus.ECHEC]),
            "pending": len([c for c in self.components.values() if c.status == InitializationStatus.EN_COURS]),
            "components": {name: status.to_dict() for name, status in self.components.items()},
        }

    def tick(self) -> None:
        """
        Méthode appelée régulièrement pour la vérification des composants.
        Cette méthode peut être utilisée pour surveiller l'état des
        composants et tenter de réinitialiser ceux qui ont échoué.
        """
        # Vérifier les composants en échec
        for name, status in self.components.items():
            if status.status == InitializationStatus.ECHEC:
                logger.info(f"Tentative de réinitialisation: {name}")
                self.initialize_component(name, force=True)
