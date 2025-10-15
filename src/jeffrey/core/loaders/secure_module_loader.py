"""
Chargeur de modules sécurisé basé sur un manifeste YAML
Sécurité : seuls les modules whitelistés dans src.jeffrey.* sont autorisés
"""

import asyncio
import importlib
import inspect
import logging
from dataclasses import dataclass
from typing import Any

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModuleSpec:
    """Spécification d'un module"""

    name: str
    import_path: str
    enabled: bool = True
    critical: bool = False
    instance: Any = None


class SecureModuleLoader:
    """Chargeur de modules sécurisé avec whitelist"""

    def __init__(self, config_path: str = "config/modules.yaml"):
        self.config_path = config_path
        self.specs: dict[str, ModuleSpec] = {}
        self.loaded: dict[str, Any] = {}

    def load_config(self):
        """Charge la configuration depuis le YAML"""
        try:
            with open(self.config_path) as f:
                config = yaml.safe_load(f) or {}

            modules = {}
            for module_config in config.get("modules", []):
                import_path = module_config["import"]

                # SÉCURITÉ : n'autoriser que src.jeffrey.*
                module_path = import_path.split(":")[0]
                if not module_path.startswith("src.jeffrey."):
                    logger.warning(f"Blocked import outside whitelist: {import_path}")
                    continue

                spec = ModuleSpec(
                    name=module_config["name"],
                    import_path=import_path,
                    enabled=module_config.get("enabled", True),
                    critical=module_config.get("critical", False),
                )
                modules[spec.name] = spec

            self.specs = modules
            logger.info(f"Loaded {len(modules)} module specs from {self.config_path}")

        except FileNotFoundError:
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self.specs = {}

    async def load_all_enabled(self) -> dict[str, Any]:
        """Charge tous les modules activés"""
        loaded = {}

        for name, spec in self.specs.items():
            if not spec.enabled:
                continue

            instance = await self._safe_import(spec)
            if instance:
                loaded[name] = instance
                self.loaded[name] = instance

        logger.info(f"Loaded {len(loaded)} modules")
        return loaded

    async def _safe_import(self, spec: ModuleSpec) -> Any | None:
        """Import sécurisé d'un module"""
        try:
            # Parser le chemin d'import
            if ":" in spec.import_path:
                module_path, class_name = spec.import_path.split(":", 1)
            else:
                module_path = spec.import_path
                class_name = None

            # Importer le module
            module = await asyncio.to_thread(importlib.import_module, module_path)

            # Extraire la classe si spécifiée
            if class_name:
                cls = getattr(module, class_name)
                # Instancier si c'est une classe
                if callable(cls):
                    try:
                        sig = inspect.signature(cls)
                        if "loader" in sig.parameters:
                            instance = cls(loader=self)
                        else:
                            instance = cls()
                    except Exception:
                        instance = cls()
                else:
                    instance = cls
            else:
                instance = module

            spec.instance = instance
            logger.info(f"✅ Loaded module: {spec.name}")
            return instance

        except Exception as e:
            msg = f"Failed to load {spec.name} ({spec.import_path}): {e}"

            if spec.critical:
                logger.error(msg)
                raise
            else:
                logger.warning(msg)
                return None
