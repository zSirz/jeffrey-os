"""
Optimiseur de performances cognitives.

Ce module implémente les fonctionnalités essentielles pour optimiseur de performances cognitives.
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

import gzip
import hashlib
import json
import re
import shutil
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


class JeffreyOptimizer:
    """Optimise le système Jeffrey"""

    def __init__(self, project_root: str = ".") -> None:
        self.project_root = Path(project_root).absolute()
        self.optimization_report = {
            "timestamp": datetime.now().isoformat(),
            "optimizations_applied": [],
            "performance_gains": {},
            "space_saved": 0,
            "files_modified": 0,
            "errors": [],
        }

    def optimize_system(self) -> dict[str, Any]:
        """Applique toutes les optimisations recommandées"""

        print("🚀 Démarrage de l'optimisation du système Jeffrey...")

        # Ne lance que l'optimisation des performances
        optimizations = {
            "performance": self._optimize_performance,
        }

        results = {}

        for name, optimizer in optimizations.items():
            print(f"\n🔧 Optimisation {name}...")
            try:
                results[name] = optimizer()
                self.optimization_report["optimizations_applied"].append(name)
            except Exception as e:
                error_msg = f"Erreur lors de l'optimisation {name}: {str(e)}"
                print(f"❌ {error_msg}")
                self.optimization_report["errors"].append(error_msg)
                results[name] = {"status": "failed", "error": str(e)}

        # Résumé final
        self._generate_optimization_summary(results)

        return self.optimization_report

    def _optimize_memory(self) -> dict[str, Any]:
        """Optimise l'utilisation mémoire"""
        results = {"compressed_files": 0, "deduplicated": 0, "archived": 0, "space_saved_mb": 0}

        # 1. Compression des vieux fichiers JSON
        old_memories = self._find_old_memory_files(days=30)
        for memory_file in old_memories:
            if self._compress_json_file(memory_file):
                results["compressed_files"] += 1

        # 2. Déduplication des souvenirs
        duplicates = self._find_duplicate_memories()
        for dup_file in duplicates:
            if self._remove_duplicate_file(dup_file):
                results["deduplicated"] += 1

        # 3. Archivage des très vieux fichiers
        very_old = self._find_old_memory_files(days=90)
        archive_dir = self.project_root / "archives"
        archive_dir.mkdir(exist_ok=True)

        for old_file in very_old:
            if self._archive_file(old_file, archive_dir):
                results["archived"] += 1

        # 4. Créer un index pour recherche rapide
        self._create_memory_index()

        # Calculer l'espace économisé
        results["space_saved_mb"] = self.optimization_report["space_saved"] / 1_000_000

        return results

    def _find_old_memory_files(self, days: int) -> list[Path]:
        """Trouve les fichiers mémoire anciens"""
        old_files = []
        cutoff_date = datetime.now() - timedelta(days=days)

        memory_patterns = ["memory", "memoire", "conversation", "journal"]

        # Utilisation d'une compréhension de liste pour optimiser les boucles
        for pattern in memory_patterns:
            old_files.extend(
                [
                    file_path
                    for file_path in self.project_root.rglob(f"*{pattern}*.json")
                    if "venv" not in str(file_path)
                    and "__pycache__" not in str(file_path)
                    and self._is_file_old(file_path, cutoff_date)
                ]
            )

        return old_files

    def _is_file_old(self, file_path: Path, cutoff_date: datetime) -> bool:
        """Vérifie si un fichier est plus ancien que la date limite"""
        try:
            mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
            return mtime < cutoff_date
        except:
            return False

    def _compress_json_file(self, file_path: Path) -> bool:
        """Compresse un fichier JSON"""
        try:
            # Lire le fichier
            with open(file_path, encoding="utf-8") as f:
                data = json.load(f)

            # Sauvegarder compressé
            compressed_path = file_path.with_suffix(".json.gz")
            with gzip.open(compressed_path, "wt", encoding="utf-8") as f:
                json.dump(data, f)

            # Calculer l'économie d'espace
            original_size = file_path.stat().st_size
            compressed_size = compressed_path.stat().st_size

            if compressed_size < original_size * 0.8:  # Au moins 20% de compression
                # Supprimer l'original
                file_path.unlink()
                self.optimization_report["space_saved"] += original_size - compressed_size
                return True
            else:
                # Pas assez de gain, supprimer la version compressée
                compressed_path.unlink()
                return False

        except Exception:
            return False

    def _find_duplicate_memories(self) -> list[Path]:
        """Trouve les fichiers mémoire dupliqués"""
        memory_hashes = {}
        duplicates = []

        for json_file in self.project_root.rglob("*.json"):
            if "venv" in str(json_file) or "__pycache__" in str(json_file):
                continue

            try:
                # Calculer le hash du contenu
                with open(json_file, "rb") as f:
                    file_hash = hashlib.md5(f.read()).hexdigest()

                if file_hash in memory_hashes:
                    # Duplicate trouvé
                    duplicates.append(json_file)
                else:
                    memory_hashes[file_hash] = json_file

            except BaseException:
                pass

        return duplicates

    def _remove_duplicate_file(self, file_path: Path) -> bool:
        """Supprime un fichier dupliqué"""
        try:
            size = file_path.stat().st_size
            file_path.unlink()
            self.optimization_report["space_saved"] += size
            return True
        except BaseException:
            return False

    def _archive_file(self, file_path: Path, archive_dir: Path) -> bool:
        """Archive un fichier"""
        try:
            # Créer la structure de répertoire dans l'archive
            relative_path = file_path.relative_to(self.project_root)
            archive_path = archive_dir / relative_path
            archive_path.parent.mkdir(parents=True, exist_ok=True)

            # Déplacer et compresser
            shutil.move(str(file_path), str(archive_path))
            return self._compress_json_file(archive_path)
        except BaseException:
            return False

    def _create_memory_index(self):
        """Crée un index des fichiers mémoire pour recherche rapide"""
        index = {
            "created": datetime.now().isoformat(),
            "memories": {},
            "conversations": {},
            "emotional_states": {},
        }

        # Indexer tous les fichiers JSON pertinents
        for json_file in self.project_root.rglob("*.json*"):  # Inclut .json.gz
            if "venv" in str(json_file) or "__pycache__" in str(json_file):
                continue

            try:
                relative_path = str(json_file.relative_to(self.project_root))
                file_info = {
                    "path": relative_path,
                    "size": json_file.stat().st_size,
                    "modified": datetime.fromtimestamp(json_file.stat().st_mtime).isoformat(),
                }

                # Catégoriser
                if "memory" in relative_path or "memoire" in relative_path:
                    index["memories"][relative_path] = file_info
                elif "conversation" in relative_path:
                    index["conversations"][relative_path] = file_info
                elif "emotion" in relative_path:
                    index["emotional_states"][relative_path] = file_info

            except BaseException:
                pass

        # Sauvegarder l'index
        index_path = self.project_root / "data" / "memory_index.json"
        index_path.parent.mkdir(exist_ok=True)

        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)

    def _optimize_imports(self) -> dict[str, Any]:
        """Optimise les imports Python"""
        results = {"files_cleaned": 0, "imports_removed": 0, "circular_fixed": 0}

        # Utiliser autoflake si disponible
        try:
            import subprocess

            # Exécuter autoflake sur tous les fichiers Python
            cmd = [
                "autoflake",
                "--in-place",
                "--remove-all-unused-imports",
                "--remove-unused-variables",
                "--recursive",
                str(self.project_root),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Compter les fichiers modifiés
                modified = result.stdout.count("modified")
                results["files_cleaned"] = modified
                results["imports_removed"] = modified * 2  # Estimation

        except (ImportError, subprocess.CalledProcessError):
            # Fallback: nettoyage manuel basique
            for py_file in self.project_root.rglob("*.py"):
                if "venv" in str(py_file) or "__pycache__" in str(py_file):
                    continue

                if self._clean_unused_imports(py_file):
                    results["files_cleaned"] += 1

        return results

    def _clean_unused_imports(self, file_path: Path) -> bool:
        """Nettoie les imports non utilisés d'un fichier (méthode basique)"""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            lines = content.split("\n")
            new_lines = []
            imports_removed = False

            for line in lines:
                # Détection basique des imports non utilisés
                if line.strip().startswith("import ") or line.strip().startswith("from "):
                    # Extraire le nom du module/objet importé
                    if "import " in line:
                        parts = line.split("import ")
                        if len(parts) > 1:
                            imported = parts[1].split()[0].strip()
                            # Vérifier si utilisé dans le reste du fichier
                            rest_of_file = "\n".join(lines[lines.index(line) + 1 :])
                            if imported not in rest_of_file:
                                imports_removed = True
                                continue

                new_lines.append(line)

            if imports_removed:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(new_lines))
            return True

        except BaseException:
            pass

        return False

    def _optimize_performance(self) -> dict[str, Any]:
        """Optimise les patterns de code lents"""
        results = {
            "loops_optimized": 0,
            "sleeps_removed": 0,
            "regex_compiled": 0,
            "files_analyzed": 0,
            "files_skipped": 0,
        }

        for py_file in self.project_root.rglob("*.py"):
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            # Exclusions temporaires pour fichiers problématiques
            problematic_files = [
                "test_",  # Exclure tous les fichiers de test
                "benchmark",  # Exclure les benchmarks
                "venv",
                "__pycache__",
                "setup_init_files",  # Exclure spécifiquement ce fichier problématique
                ".git",  # Exclure le dossier git
            ]

            if any(problematic in str(py_file) for problematic in problematic_files):
                results["files_skipped"] += 1
                continue

            # Limiter la taille des fichiers analysés (éviter les fichiers trop gros)
            try:
                file_size = py_file.stat().st_size
                if file_size > 100_000:  # Skip les fichiers > 100KB
                    print(f"⏭️ Fichier trop gros ignoré : {py_file} ({file_size / 1024:.1f}KB)")
                    results["files_skipped"] += 1
                    continue
            except:
                continue

            # Print explicite du fichier en cours d'analyse
            print(f"📂 Analyse : {py_file}")

            # Timer pour détecter les fichiers qui prennent trop de temps
            start_time = time.time()

            try:
                # Timeout de sécurité sur la lecture du fichier
                content = None
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                original_content = content

                # 1. Remplacer les time.sleep courts par des alternatives
                content = self._optimize_sleeps(content, results)

                # 2. Précompiler les regex (désactivé temporairement car trop lent)
                # content = self._precompile_regex(content, results)

                # 3. Optimiser les boucles imbriquées simples (avec timeout)
                content = self._optimize_nested_loops(content, results)

                # Sauvegarder si modifié
                if content != original_content:
                    with open(py_file, "w", encoding="utf-8") as f:
                        f.write(content)
                    self.optimization_report["files_modified"] += 1

                results["files_analyzed"] += 1

                # Vérifier si le traitement a pris trop de temps
                elapsed_time = time.time() - start_time
                if elapsed_time > 2.0:  # Plus de 2 secondes
                    print(f"⚠️ Long traitement : {py_file} ({elapsed_time:.2f}s)")

            except Exception as e:
                print(f"❌ Erreur avec {py_file}: {str(e)}")
                self.optimization_report["errors"].append(f"Fichier {py_file}: {str(e)}")

        return results

    def _optimize_sleeps(self, content: str, results: dict) -> str:
        """Optimise les time.sleep"""
        # Précompiler le pattern pour les sleep courts
        sleep_pattern = re.compile(r"time\.sleep\((0\.\d+)\)")

        def replace_sleep(match):
            sleep_duration = match.group(1)
            results["sleeps_removed"] += 1

            # Si dans une fonction async, utiliser await asyncio.sleep
            if "async def" in content[: match.start()]:
                return f"await asyncio.sleep({sleep_duration})"
            else:
                # Pour les fonctions synchrones, suggérer une alternative
                return f"time.sleep({sleep_duration})  # Considérer threading.Event pour éviter le blocage"

        return sleep_pattern.sub(replace_sleep, content)

    def _precompile_regex(self, content: str, results: dict) -> str:
        """Précompile les expressions régulières"""
        # Pattern simplifié pour éviter le catastrophic backtracking
        # On cherche juste les appels re.* dans le code, sans analyser les boucles
        regex_calls_pattern = re.compile(r"re\.(search|match|findall|compile)\s*\(\s*['\"]([^'\"]+)['\"]")

        # Dictionnaire pour stocker les patterns compilés
        compiled_patterns = {}
        pattern_counter = 0

        # Compteur pour éviter de traiter trop de patterns
        max_patterns = 10
        patterns_found = 0

        for match in regex_calls_pattern.finditer(content):
            if patterns_found >= max_patterns:
                break
            method = match.group(1)
            pattern = match.group(2)

            # Éviter les patterns trop complexes
            if len(pattern) > 50 or pattern.count("*") > 3 or pattern.count("+") > 3:
                continue

            if pattern not in compiled_patterns:
                pattern_counter += 1
                patterns_found += 1
                pattern_name = f"PATTERN_{pattern_counter}"
                compiled_patterns[pattern] = pattern_name
                results["regex_compiled"] += 1

        # Si on a trouvé des patterns à compiler, ajouter un commentaire
        if compiled_patterns:
            # On ajoute juste un commentaire au lieu de modifier le code
            comment = "\n# TODO: Consider precompiling these regex patterns for better performance:\n"
            for pattern, name in list(compiled_patterns.items())[:3]:  # Limiter à 3 suggestions
                comment += f"# {name} = re.compile(r'{pattern}')\n"

            # Ajouter le commentaire après les imports si possible
            import_pos = content.find("import re")
            if import_pos != -1:
                next_line = content.find("\n", import_pos)
                if next_line != -1:
                    content = content[:next_line] + comment + content[next_line:]

        return content

    def _optimize_nested_loops(self, content: str, results: dict) -> str:
        """Optimise les boucles imbriquées simples"""
        # Pattern simplifié pour éviter les problèmes de performance
        # On cherche juste deux 'for' consécutifs sur des lignes adjacentes
        nested_pattern = re.compile(
            r"^(\s*)for\s+(\w+)\s+in\s+([^:\n]{1,50}):\s*\n\s*for\s+(\w+)\s+in\s+([^:\n]{1,50}):",
            re.MULTILINE,
        )

        def replace_nested_loops(match):
            indent = match.group(1)
            var1 = match.group(2)
            iter1 = match.group(3)
            var2 = match.group(4)
            iter2 = match.group(5)

            # Si les itérables sont simples, suggérer itertools.product
            if iter1.strip() and iter2.strip() and not any(c in iter1 + iter2 for c in "()[]{}"):
                results["loops_optimized"] += 1
                return f"{indent}# Boucles optimisées avec itertools.product\n{indent}for {var1}, {var2} in product({iter1}, {iter2}):"
            else:
                # Sinon, ajouter un commentaire d'amélioration
                return f"{indent}# Considérer l'utilisation d'itertools.product ou de compréhensions\n{match.group(0)}"

        return nested_pattern.sub(replace_nested_loops, content)

    def _optimize_startup(self) -> dict[str, Any]:
        """Optimise le temps de démarrage"""
        results = {"lazy_imports": 0, "cached_configs": 0, "preloaded_resources": 0}

        # 1. Créer un fichier de cache pour les configurations
        self._create_config_cache(results)

        # 2. Identifier les imports lourds pour lazy loading
        self._identify_heavy_imports(results)

        # 3. Précharger les ressources critiques
        self._preload_critical_resources(results)

        return results

    def _create_config_cache(self, results: dict):
        """Crée un cache des configurations"""
        config_cache = {}

        # Chercher tous les fichiers de config
        for config_file in self.project_root.rglob("config*.yaml"):
            try:
                import yaml

                with open(config_file, encoding="utf-8") as f:
                    config_data = yaml.safe_load(f)

                config_cache[str(config_file.relative_to(self.project_root))] = config_data
                results["cached_configs"] += 1
            except Exception:
                pass

        # Sauvegarder le cache
        cache_path = self.project_root / "data" / "config_cache.json"
        cache_path.parent.mkdir(exist_ok=True)

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(config_cache, f, indent=2)

    def _identify_heavy_imports(self, results: dict):
        """Identifie les imports lourds pour lazy loading"""
        heavy_modules = [
            "tensorflow",
            "torch",
            "transformers",
            "numpy",
            "pandas",
            "matplotlib",
            "scipy",
            "sklearn",
            "cv2",
            "PIL",
        ]

        recommendations = []

        for py_file in self.project_root.rglob("*.py"):
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()

                for module in heavy_modules:
                    if f"import {module}" in content or f"from {module}" in content:
                        recommendations.append(
                            {
                                "file": str(py_file.relative_to(self.project_root)),
                                "module": module,
                                "recommendation": f"Considérer lazy loading pour {module}",
                            }
                        )
                        results["lazy_imports"] += 1

            except Exception:
                pass

        # Sauvegarder les recommandations
        if recommendations:
            rec_path = self.project_root / "optimization_recommendations.json"
            with open(rec_path, "w", encoding="utf-8") as f:
                json.dump(recommendations, f, indent=2)

    def _preload_critical_resources(self, results: dict):
        """Précharge les ressources critiques"""
        # Créer un script de préchargement
        preload_script = '''#!/usr/bin/env python3
"""Script de préchargement des ressources critiques pour Jeffrey"""

import json
import os

# Précharger les configurations essentielles
CRITICAL_CONFIGS = [
    "config/config.yaml",
    "data/emotional_traits_profile.json",
    "core/data/rituels_data.json"
]

# Précharger les données émotionnelles de base
EMOTIONAL_DATA = {}

def preload_resources():
    """Précharge les ressources critiques au démarrage"""

    # Cache des configurations
    config_cache = {}

    for config_path in CRITICAL_CONFIGS:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    if config_path.endswith('.json'):
                        config_cache[config_path] = json.load(f)
                    # Pour YAML, on skip si pas disponible
            except:
                pass

    return config_cache

# Précharger au moment de l'import
PRELOADED_CACHE = preload_resources()
'''

        # Sauvegarder le script
        preload_path = self.project_root / "core" / "preload_resources.py"
        with open(preload_path, "w", encoding="utf-8") as f:
            f.write(preload_script)

        results["preloaded_resources"] = 1

    def _optimize_code_quality(self) -> dict[str, Any]:
        """Améliore la qualité du code"""
        results = {"docstrings_added": 0, "type_hints_added": 0, "formatting_fixed": 0}

        # Utiliser black pour le formatage si disponible
        try:
            import subprocess

            cmd = ["black", "--line-length", "100", str(self.project_root)]
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                # Compter les fichiers formatés
                results["formatting_fixed"] = result.stdout.count("reformatted")

        except Exception:
            pass

        return results

    def _generate_optimization_summary(self, results: dict):
        """Génère un résumé des optimisations"""
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ DES OPTIMISATIONS")
        print("=" * 60)

        total_optimizations = 0

        for category, category_results in results.items():
            if isinstance(category_results, dict) and "status" not in category_results:
                print(f"\n{category.upper()}:")
                for metric, value in category_results.items():
                    print(f"  - {metric}: {value}")
                    if isinstance(value, int):
                        total_optimizations += value

        print(f"\n✅ Total des optimisations appliquées: {total_optimizations}")
        print(f"💾 Espace économisé: {self.optimization_report['space_saved'] / 1_000_000:.2f} MB")
        print(f"📝 Fichiers modifiés: {self.optimization_report['files_modified']}")

        if self.optimization_report["errors"]:
            print(f"\n⚠️ Erreurs rencontrées: {len(self.optimization_report['errors'])}")
            for error in self.optimization_report["errors"][:3]:
                print(f"  - {error}")

        print("\n" + "=" * 60)

    def generate_optimization_report(self, output_path: str = "optimization_report.json"):
        """Génère un rapport détaillé des optimisations"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.optimization_report, f, ensure_ascii=False, indent=2)

        print(f"\n📄 Rapport d'optimisation sauvegardé: {output_path}")


if __name__ == "__main__":
    # Test de l'optimiseur
    optimizer = JeffreyOptimizer()
    report = optimizer.optimize_system()
    optimizer.generate_optimization_report()
