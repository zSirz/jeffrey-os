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

import ast
import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .graph_engine import ModuleNode, SimpleEvolutionaryGraph
from .namespace_firewall import NamespaceFirewall
from .policy_bus import Domain, PolicyBus
from .security_analyzer import SimpleSecurity


@dataclass
class BrainModuleContract:
    """Contrat JEFFREY_PLUGIN"""

    topics_in: list[str] = None
    topics_out: list[str] = None
    handler: str = "process"
    dependencies: list[str] = None

    def __post_init__(self):
        if self.topics_in is None:
            self.topics_in = []
        if self.topics_out is None:
            self.topics_out = []
        if self.dependencies is None:
            self.dependencies = []


@dataclass
class DiscoveredBrainModule:
    """Module découvert"""

    name: str
    path: Path
    category: str
    contract: BrainModuleContract | None = None
    complexity: int = 0
    security_passed: bool = True


class BrainDiscoveryFinal:
    """Discovery pour le cerveau uniquement"""

    def __init__(self, brain, dry_run: bool = True) -> None:
        self.brain = brain
        self.dry_run = dry_run
        self.logger = logging.getLogger("discovery.brain")

        # Composants
        self.firewall = NamespaceFirewall()
        self.policy_bus = PolicyBus(brain.bus, self.firewall)
        self.graph = SimpleEvolutionaryGraph(pruning_threshold=0.2)
        self.security = SimpleSecurity(strict_mode=False)

        # Modules découverts
        self.modules: dict[str, DiscoveredBrainModule] = {}

        # Chemins brain uniquement
        self.brain_paths = [
            "core",
            "consciousness",
            "emotions",
            "memory",
            "perception",
            "cognition",
            "orchestrator",
            "thalamus",
            "workspace",
            "limbic",
            "tissues",
            "orchestration",
            "learning",
            "personality",
        ]

    async def discover_and_connect(self) -> tuple[int, list[str]]:
        """Découvrir et connecter les modules"""

        self.logger.info("🧠 Starting Brain Discovery")

        # Phase 1: Scan
        self.logger.info("📡 Phase 1: Scanning")
        self._scan_brain_modules()

        # Phase 2: Sécurité
        self.logger.info("🔒 Phase 2: Security")
        self._check_security()

        # Phase 3: Graphe
        self.logger.info("🕸️ Phase 3: Graph")
        self._build_graph()

        # Phase 4: Tri
        self.logger.info("📊 Phase 4: Sort")
        try:
            load_order = self.graph.topological_sort()
        except Exception as e:
            self.logger.error(f"Sort failed: {e}")
            load_order = list(self.modules.keys())

        self.logger.info(f"📋 Order: {load_order[:10]}...")

        if self.dry_run:
            self.logger.info(f"✅ DRY-RUN: {len(self.modules)} modules")
            self._save_manifest()
            return len(self.modules), []

        # Phase 5: Connexion
        self.logger.info("🔌 Phase 5: Connect")
        connected, failed = await self._connect_modules(load_order)

        # Phase 6: Optimisation
        self.logger.info("🧹 Phase 6: Optimize")
        self.graph.optimize()

        metrics = self.graph.get_metrics()
        self.logger.info(f"📊 Metrics: {metrics}")

        return connected, failed

    def _scan_brain_modules(self):
        """Scanner tous les modules (plugin + legacy) de manière récursive et optimisée"""

        # Trouver le répertoire de base
        base_path = Path("src/jeffrey")
        if not base_path.exists():
            base_path = Path("jeffrey")
            if not base_path.exists():
                base_path = Path(".")

        # Garantir src dans sys.path pour les imports
        src_root = Path("src")
        if src_root.exists():
            p = str(src_root.resolve())
            if p not in sys.path:
                sys.path.insert(0, p)

        # Compteurs pour statistiques
        found_count = 0
        plugin_count = 0
        legacy_count = 0
        skipped_count = 0

        # Patterns à exclure (optimisé)
        SKIP_PATTERNS = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".venv",
            "venv",
            "node_modules",
            ".idea",
            ".vscode",
        }

        # Patterns de fichiers à exclure
        SKIP_FILES = {"test_", "_test", "conftest", "setup.py", "__init__.py"}

        # Dictionnaire pour éviter les doublons de noms avec clés uniques
        seen_keys = {}

        self.logger.info("🔍 Starting comprehensive module scan...")

        for brain_dir in self.brain_paths:
            dir_path = base_path / brain_dir

            if not dir_path.exists():
                # Essayer aussi dans core/
                alt_path = base_path / "core" / brain_dir
                if alt_path.exists():
                    dir_path = alt_path
                else:
                    self.logger.debug(f"Path not found: {dir_path}")
                    continue

            self.logger.debug(f"Scanning directory: {dir_path}")

            # Scan récursif avec rglob
            for py_file in dir_path.rglob("*.py"):
                # Skip discovery itself (cross-platform)
                if (
                    Path("src/jeffrey/core/discovery") in py_file.parents
                    or Path("jeffrey/core/discovery") in py_file.parents
                ):
                    skipped_count += 1
                    continue

                # Convertir en string pour les checks
                file_str = str(py_file)
                file_name = py_file.stem

                # Skip patterns optimisé
                if any(skip in file_str for skip in SKIP_PATTERNS):
                    skipped_count += 1
                    continue

                # Skip files spécifiques
                if any(skip in file_name for skip in SKIP_FILES):
                    skipped_count += 1
                    continue

                # Créer une clé unique basée sur le chemin relatif
                try:
                    module_key = str(py_file.relative_to(base_path)).replace(".py", "").replace("/", ".")
                except ValueError:
                    module_key = file_name  # Fallback si relative_to échoue

                # Skip si déjà vu (garde le premier trouvé)
                if module_key in seen_keys:
                    self.logger.debug(f"Duplicate module key {module_key}, keeping first: {seen_keys[module_key]}")
                    skipped_count += 1
                    continue

                try:
                    # Lire le code
                    code = py_file.read_text(encoding="utf-8", errors="ignore")

                    # Vérifier que c'est un module Python valide
                    has_class = "class " in code
                    has_function = "def " in code
                    has_jeffrey_plugin = "JEFFREY_PLUGIN" in code

                    if not (has_class or has_function):
                        skipped_count += 1
                        continue

                    # Extraire le contrat (peut être vide pour legacy)
                    contract = self._extract_contract(code)

                    # Calculer la complexité
                    complexity = self._calculate_complexity(code)

                    # Créer le module découvert avec nom pour affichage mais clé unique pour indexation
                    module = DiscoveredBrainModule(
                        name=file_name,  # Garder le stem pour l'affichage
                        path=py_file,
                        category=brain_dir,
                        contract=contract,
                        complexity=complexity,
                        security_passed=True,  # Will be checked later
                    )

                    # Utiliser la clé unique pour l'indexation
                    self.modules[module_key] = module
                    seen_keys[module_key] = str(py_file)
                    found_count += 1

                    # Statistiques
                    if has_jeffrey_plugin and contract and contract.topics_in:
                        plugin_count += 1
                        self.logger.debug(f"  ✓ Plugin: {file_name} | Topics: {contract.topics_in}")
                    else:
                        legacy_count += 1
                        self.logger.debug(f"  ○ Legacy: {file_name} | Category: {brain_dir}")

                except Exception as e:
                    self.logger.warning(f"Error scanning {py_file}: {e}")
                    skipped_count += 1

        # Rapport final détaillé
        self.logger.info("=" * 50)
        self.logger.info("📊 SCAN COMPLETE - SUMMARY:")
        self.logger.info(f"  Total modules found: {found_count}")
        self.logger.info(f"  ├─ Plugins (with JEFFREY_PLUGIN): {plugin_count}")
        self.logger.info(f"  ├─ Legacy modules: {legacy_count}")
        self.logger.info(f"  └─ Files skipped: {skipped_count}")

        # Détail par catégorie
        category_counts = {}
        for module in self.modules.values():
            cat = module.category
            category_counts[cat] = category_counts.get(cat, 0) + 1

        self.logger.info("\n  By category:")
        for cat, count in sorted(category_counts.items()):
            self.logger.info(f"    - {cat}: {count} modules")

        self.logger.info("=" * 50)

        # Stocker les stats pour le manifest
        self.scan_stats = {
            "total": found_count,
            "plugin": plugin_count,
            "legacy": legacy_count,
            "skipped": skipped_count,
            "by_category": category_counts,
        }

    def _extract_contract(self, code: str) -> BrainModuleContract:
        """Extraire le contrat du code"""

        contract = BrainModuleContract()

        try:
            tree = ast.parse(code)

            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "JEFFREY_PLUGIN":
                            if isinstance(node.value, ast.Dict):
                                plugin_dict = self._ast_dict_to_dict(node.value)

                                contract.topics_in = plugin_dict.get("topics_in", [])
                                contract.topics_out = plugin_dict.get("topics_out", [])
                                contract.handler = plugin_dict.get("handler", "process")
                                contract.dependencies = plugin_dict.get("dependencies", [])

                                return contract

        except Exception as e:
            self.logger.warning(f"Contract parse failed: {e}")

        # Contrat par défaut
        contract.topics_in = []
        contract.handler = "process"

        return contract

    def _ast_dict_to_dict(self, node: ast.Dict) -> dict:
        """Convertir AST dict en Python dict"""
        result = {}

        for key, value in zip(node.keys, node.values):
            if isinstance(key, ast.Constant):
                key_str = key.value
            elif isinstance(key, ast.Str):
                key_str = key.s
            else:
                continue

            if isinstance(value, ast.Constant):
                result[key_str] = value.value
            elif isinstance(value, ast.Str):
                result[key_str] = value.s
            elif isinstance(value, ast.List):
                list_items = []
                for item in value.elts:
                    if isinstance(item, (ast.Constant, ast.Str)):
                        list_items.append(item.value if isinstance(item, ast.Constant) else item.s)
                result[key_str] = list_items

        return result

    def _check_security(self):
        """Vérifier la sécurité"""

        for name, module in self.modules.items():
            try:
                with open(module.path, encoding="utf-8") as f:
                    code = f.read()

                report = self.security.analyze_code(code)
                module.security_passed = report.passed

                if not report.passed:
                    self.logger.warning(f"⚠️ {name} security: {report.violations}")

            except Exception as e:
                self.logger.error(f"Security check failed {name}: {e}")
                module.security_passed = False

    def _build_graph(self):
        """Construire le graphe"""

        for name, module in self.modules.items():
            node = ModuleNode(name=name, category=module.category, complexity=module.complexity)
            self.graph.add_module(node)

        for name, module in self.modules.items():
            if module.contract and module.contract.dependencies:
                for dep in module.contract.dependencies:
                    if "." in dep:
                        dep = dep.split(".")[-1]

                    if dep in self.modules:
                        self.graph.add_dependency(dep, name)

    async def _connect_modules(self, load_order: list[str]) -> tuple[int, list[str]]:
        """
        Connecter UNIQUEMENT les modules plugin.
        Les legacy sont déjà connectés par le boot, on les documente seulement.
        """

        connected = 0
        failed = []
        skipped_legacy = 0
        skipped_security = 0

        total = len(load_order)

        self.logger.info(f"🔌 Starting module connection phase ({total} modules to process)")

        for idx, module_key in enumerate(load_order, 1):
            module = self.modules.get(module_key)
            if not module:
                continue

            # Progress indicator
            if idx % 10 == 0:
                self.logger.debug(f"  Progress: {idx}/{total}")

            # En dry-run: ne rien connecter, juste simuler
            if self.dry_run:
                # Feedback positif pour le graphe évolutif
                self.graph.update_edge_feedback(module.name, module.name, success=True, delta=0.1)
                continue

            # Skip si échec sécurité
            if not module.security_passed:
                self.logger.warning(f"  ⚠️ Security failed: {module.name}")
                skipped_security += 1
                failed.append(module.name)
                self.graph.update_edge_feedback(module.name, module.name, success=False)
                continue

            try:
                # Déterminer le type de module
                is_plugin = bool(module.contract and module.contract.topics_in)

                if is_plugin:
                    # CAS 1: Module Plugin avec JEFFREY_PLUGIN
                    # On peut le connecter via discovery

                    self.logger.debug(f"  Connecting plugin: {module.name}")

                    # Charger le module
                    instance = self._load_module(module)
                    if instance is None:
                        raise ValueError(f"Failed to load module {module.name}")

                    # Créer le handler
                    handler_method = module.contract.handler or "process"
                    handler = self.policy_bus.make_handler(instance, handler_method)

                    if handler is None:
                        raise ValueError(f"No valid handler for {module.name}")

                    # Souscrire aux topics
                    for topic in module.contract.topics_in:
                        try:
                            self.policy_bus.subscribe_guarded(Domain.BRAIN, topic, handler)
                            self.logger.debug(f"    → Subscribed to {topic}")
                        except Exception as e:
                            self.logger.warning(f"    Failed to subscribe {module.name} to {topic}: {e}")

                    connected += 1
                    self.logger.info(f"  ✅ Connected plugin: {module.name}")

                    # Feedback positif fort pour les plugins
                    self.graph.update_edge_feedback(module.name, module.name, success=True, delta=0.2)

                else:
                    # CAS 2: Module Legacy sans JEFFREY_PLUGIN
                    # Déjà connecté par le boot, on ne fait rien

                    skipped_legacy += 1
                    self.logger.debug(f"  ↩️ Skipped legacy: {module.name} (already connected by boot)")

                    # Feedback neutre pour les legacy
                    self.graph.update_edge_feedback(module.name, module.name, success=True, delta=0.05)

            except Exception as e:
                failed.append(module.name)
                self.logger.error(f"  ❌ Failed: {module.name} | Error: {str(e)[:100]}")

                # Feedback négatif
                self.graph.update_edge_feedback(module.name, module.name, success=False)

        # Rapport final de connexion
        if not self.dry_run:
            self.logger.info("=" * 50)
            self.logger.info("📊 CONNECTION PHASE COMPLETE:")
            self.logger.info(f"  ✅ Plugins connected: {connected}")
            self.logger.info(f"  ↩️ Legacy skipped (already connected): {skipped_legacy}")
            self.logger.info(f"  ⚠️ Security skipped: {skipped_security}")
            self.logger.info(f"  ❌ Failed: {len(failed)}")

            if failed and len(failed) <= 10:
                self.logger.info(f"  Failed modules: {', '.join(failed)}")
            elif failed:
                self.logger.info(f"  Failed modules (first 10): {', '.join(failed[:10])}...")

            self.logger.info("=" * 50)

        return connected, failed

    def _load_module(self, module: DiscoveredBrainModule) -> Any | None:
        """Charger un module avec imports sécurisés"""

        # Garantir src dans sys.path
        src_root = Path("src")
        if src_root.exists():
            p = str(src_root.resolve())
            if p not in sys.path:
                sys.path.insert(0, p)

        try:
            spec = importlib.util.spec_from_file_location(f"jeffrey.{module.category}.{module.name}", module.path)

            if spec is None or spec.loader is None:
                return None

            py_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(py_module)

            # Chercher une classe
            for attr_name in dir(py_module):
                if attr_name.startswith("_"):
                    continue

                attr = getattr(py_module, attr_name)

                if isinstance(attr, type):
                    try:
                        instance = attr()

                        handler_name = module.contract.handler if module.contract else "process"
                        if hasattr(instance, handler_name):
                            return instance

                    except Exception:
                        continue

            return py_module

        except Exception as e:
            self.logger.error(f"Load failed {module.name}: {e}")
            return None

    def _calculate_complexity(self, code: str) -> int:
        """Calculer complexité simple"""
        complexity = 1

        keywords = ["if ", "for ", "while ", "try:", "except", "elif "]
        for line in code.split("\n"):
            if any(kw in line for kw in keywords):
                complexity += 1

        return min(complexity, 20)

    def _save_manifest(self):
        """Sauvegarder un manifest détaillé avec toutes les informations utiles"""

        import json
        from datetime import datetime

        import yaml

        # Compter les types
        plugin_modules = {key: m for key, m in self.modules.items() if m.contract and m.contract.topics_in}
        legacy_modules = {key: m for key, m in self.modules.items() if not (m.contract and m.contract.topics_in)}

        # Créer le manifest enrichi
        manifest = {
            "metadata": {
                "version": "1.0",
                "generated_at": datetime.now().isoformat(),
                "discovery_mode": "dry_run" if self.dry_run else "live",
            },
            "summary": {
                "total_discovered": len(self.modules),
                "plugin_count": len(plugin_modules),
                "legacy_count": len(legacy_modules),
                "scan_stats": getattr(self, "scan_stats", {}),
            },
            "graph_metrics": self.graph.get_metrics(),
            "modules": {},
        }

        # Ajouter les détails de chaque module
        for module_key, module in sorted(self.modules.items()):
            # Déterminer le type et le statut
            module_type = "plugin" if (module.contract and module.contract.topics_in) else "legacy"
            connection_status = "discovery" if module_type == "plugin" else "boot"

            module_info = {
                "display_name": module.name,  # Nom d'affichage (stem)
                "type": module_type,
                "path": str(module.path),
                "category": module.category,
                "complexity": module.complexity,
                "security_passed": module.security_passed,
                "connection": {
                    "status": connection_status,
                    "method": "PolicyBus" if module_type == "plugin" else "direct_boot",
                },
            }

            # Ajouter les infos du contrat si présent
            if module.contract:
                module_info["contract"] = {
                    "topics_in": module.contract.topics_in or [],
                    "topics_out": module.contract.topics_out or [],
                    "handler": module.contract.handler,
                    "dependencies": module.contract.dependencies or [],
                }

            manifest["modules"][module_key] = module_info

        # Sauvegarder en YAML
        manifest_path = Path("discovered_brain.yaml")
        try:
            with open(manifest_path, "w") as f:
                yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
            self.logger.info(f"📄 Manifest saved: {manifest_path}")
        except Exception as e:
            self.logger.warning(f"Could not save YAML manifest: {e}")

        # Créer aussi une version JSON pour compatibilité
        json_path = Path("discovered_brain.json")
        with open(json_path, "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        self.logger.info(f"📄 JSON manifest saved: {json_path}")
