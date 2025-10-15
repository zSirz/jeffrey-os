# TODO: Précompiler les regex utilisées dans les boucles
# TODO: Précompiler les regex utilisées dans les boucles
# TODO: Précompiler les regex utilisées dans les boucles
"""
Moniteur de santé système global.

Ce module implémente les fonctionnalités essentielles pour moniteur de santé système global.
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
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

# Essayer d'importer psutil, mais continuer sans si non disponible
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class JeffreySystemHealth:
    """Vérifie la santé du système Jeffrey"""

    def __init__(self, project_root: str = ".") -> None:
        self.project_root = Path(project_root).absolute()
        self.report = {
            "status": "unknown",
            "issues": [],
            "warnings": [],
            "optimizations": [],
            "orphaned_files": [],
            "unused_imports": [],
            "circular_dependencies": [],
            "missing_dependencies": [],
            "performance_issues": [],
            "memory_issues": [],
            "timestamp": datetime.now().isoformat(),
        }

    def run_full_diagnostic(self) -> dict[str, Any]:
        """Diagnostic complet du système"""
        print("🔍 Démarrage du diagnostic système Jeffrey...")

        # 1. Vérifier la structure du projet
        print("\n1️⃣ Vérification de la structure...")
        self._check_project_structure()

        # 2. Vérifier les imports et dépendances
        print("\n2️⃣ Analyse des imports et dépendances...")
        self._check_imports_and_dependencies()

        # 3. Vérifier la mémoire et les données
        print("\n3️⃣ Vérification du système de mémoire...")
        self._check_memory_system()

        # 4. Vérifier les performances
        print("\n4️⃣ Analyse des performances...")
        self._check_performance()

        # 5. Vérifier l'intégration des modules
        print("\n5️⃣ Vérification de l'intégration...")
        self._check_module_integration()

        # 6. Vérifier la cohérence des données
        print("\n6️⃣ Vérification de la cohérence des données...")
        self._check_data_consistency()

        # 7. Chercher le code mort et les TODOs
        print("\n7️⃣ Recherche du code mort et TODOs...")
        self._check_dead_code_and_todos()

        # 8. Analyser l'utilisation des ressources
        print("\n8️⃣ Analyse des ressources système...")
        self._check_resource_usage()

        # Déterminer le statut global
        self._determine_overall_status()

        return self.report

    def _check_project_structure(self):
        """Vérifie la structure du projet"""
        required_dirs = [
            "core",
            "core/emotions",
            "core/consciousness",
            "core/memory",
            "data",
            "Jeffrey_Memoire",
            "ui",
            "widgets",
            "tests",
        ]

        required_files = [
            "core/__init__.py",
            "core/jeffrey_emotional_core.py",
            "core/consciousness/jeffrey_living_consciousness.py",
        ]

        # Vérifier les répertoires
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if not full_path.exists():
                self.report["issues"].append(f"Répertoire manquant: {dir_path}")
            elif not full_path.is_dir():
                self.report["issues"].append(f"'{dir_path}' n'est pas un répertoire")

        # Vérifier les fichiers essentiels
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                self.report["issues"].append(f"Fichier essentiel manquant: {file_path}")

    def _check_imports_and_dependencies(self):
        """Analyse tous les imports et détecte les problèmes"""
        python_files = list(self.project_root.rglob("*.py"))
        imports_map = {}
        circular_imports = []
        unused_imports = []

        for py_file in python_files:
            if "__pycache__" in str(py_file) or "venv" in str(py_file):
                continue

            relative_path = py_file.relative_to(self.project_root)

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()
            except Exception:
                pass

            # Parser le fichier pour extraire les imports
            try:
                tree = ast.parse(content)
                imports = []
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            imports_map[str(relative_path)] = imports

            # Vérifier les imports non utilisés
            for imp in imports:
                imp_name = imp.split(".")[-1]
                # Recherche simple dans le contenu
                if content.count(imp_name) == 1:  # Seulement dans l'import
                    unused_imports.append(
                        {
                            "file": str(relative_path),
                            "import": imp,
                            "line": self._find_import_line(content, imp),
                        }
                    )

        # Détecter les imports circulaires
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        # TODO: Optimiser cette boucle imbriquée
        for file1, imports1 in imports_map.items():
            for file2, imports2 in imports_map.items():
                if file1 != file2:
                    # Convertir les chemins en modules
                    module1 = file1.replace(".py", "").replace("/", ".")
                    module2 = file2.replace(".py", "").replace("/", ".")

                    if module2 in imports1 and module1 in imports2:
                        circular_pair = tuple(sorted([file1, file2]))
                        if circular_pair not in circular_imports:
                            circular_imports.append(circular_pair)

        self.report["unused_imports"] = unused_imports[:20]  # Limiter à 20
        self.report["circular_dependencies"] = [f"{a} <-> {b}" for a, b in circular_imports]

    def _find_import_line(self, content: str, import_name: str) -> int:
        """Trouve le numéro de ligne d'un import"""
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if f"import {import_name}" in line or f"from {import_name}" in line:
                return i
        return 0

    def _check_memory_system(self):
        """Vérifie le système de mémoire de Jeffrey"""
        memory_dirs = ["Jeffrey_Memoire", "data/memory", "data"]

        total_memory_size = 0
        memory_files = []

        for mem_dir in memory_dirs:
            dir_path = self.project_root / mem_dir
            if dir_path.exists():
                for file_path in dir_path.rglob("*.json"):
                    try:
                        size = file_path.stat().st_size
                        total_memory_size += size
                        memory_files.append({"path": str(file_path.relative_to(self.project_root)), "size": size})
                    except Exception:
                        pass

                    # Vérifier l'intégrité JSON
                    try:
                        with open(file_path, encoding="utf-8") as f:
                            json.load(f)

                    except json.JSONDecodeError:
                        self.report["issues"].append(f"Fichier JSON corrompu: {file_path}")
                    except Exception as e:
                        self.report["warnings"].append(f"Erreur lecture mémoire {file_path}: {e}")

        # Analyser la taille totale
        if total_memory_size > 100_000_000:  # 100MB
            self.report["memory_issues"].append(
                {
                    "type": "memory_size",
                    "size": total_memory_size,
                    "files": len(memory_files),
                    "recommendation": "Considérer la compression ou l'archivage des vieilles données",
                }
            )

        # Vérifier les fichiers volumineux
        large_files = [f for f in memory_files if f["size"] > 10_000_000]  # 10MB
        if large_files:
            self.report["warnings"].append(f"Fichiers mémoire volumineux: {len(large_files)}")

    def _check_performance(self):
        """Analyse les problèmes de performance potentiels"""
        slow_patterns = [
            (r"for .+ in .+:\s*for .+ in .+:", "Boucles imbriquées"),
            (r"time\.sleep\(\d+\)", "Sleep bloquant"),
            (r"while True:", "Boucle infinie potentielle"),
            (r"\.append\(.+\) in .+ loop", "Append dans boucle"),
        ]

        performance_issues = []

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or "venv" in str(py_file):
                continue

            try:
                with open(py_file, encoding="utf-8") as f:
                    content = f.read()
                lines = content.split("\n")
            except Exception:
                continue

            # TODO: Optimiser cette boucle imbriquée
            # TODO: Optimiser cette boucle imbriquée
            # TODO: Optimiser cette boucle imbriquée
            for pattern, issue_type in slow_patterns:
                for i, line in enumerate(lines, 1):
                    if re.search(pattern, line):
                        performance_issues.append(
                            {
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": i,
                                "type": issue_type,
                                "code": line.strip()[:80],
                            }
                        )

        self.report["performance_issues"] = performance_issues[:20]  # Limiter

    def _check_module_integration(self):
        """Vérifie l'intégration des modules critiques"""
        critical_modules = [
            "jeffrey_curiosity_engine",
            "jeffrey_living_consciousness",
            "jeffrey_emotional_core",
            "jeffrey_dream_system",
            "jeffrey_secret_diary",
        ]

        integration_status = {}

        for module in critical_modules:
            # Chercher où le module est importé
            import_locations = []

            for py_file in self.project_root.rglob("*.py"):
                if "__pycache__" in str(py_file) or "venv" in str(py_file):
                    continue

                try:
                    with open(py_file, encoding="utf-8") as f:
                        content = f.read()
                    if f"import {module}" in content or f"from .{module}" in content:
                        import_locations.append(str(py_file.relative_to(self.project_root)))
                except Exception:
                    pass

            integration_status[module] = {
                "imported_in": import_locations,
                "is_integrated": len(import_locations) > 0,
            }

        # Reporter les modules non intégrés
        for module, status in integration_status.items():
            if not status["is_integrated"]:
                self.report["warnings"].append(f"Module non intégré: {module}")

    def _check_data_consistency(self):
        """Vérifie la cohérence des données"""
        # Vérifier les formats de date
        date_issues = []

        json_files = list(self.project_root.rglob("*.json"))
        for json_file in json_files:
            if "venv" in str(json_file) or "__pycache__" in str(json_file):
                continue

            try:
                with open(json_file, encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue

            # Vérifier les timestamps
            self._check_timestamps_recursive(data, str(json_file), date_issues)

        if date_issues:
            self.report["warnings"].append(f"Problèmes de format de date: {len(date_issues)} fichiers")

    def _check_timestamps_recursive(self, data: Any, file_path: str, issues: list):
        """Vérifie récursivement les timestamps dans les données"""
        if isinstance(data, dict):
            for key, value in data.items():
                if key in ["timestamp", "date", "created_at", "updated_at"]:
                    if isinstance(value, str):
                        try:
                            # Essayer de parser la date
                            datetime.fromisoformat(value.replace("Z", "+00:00"))
                        except Exception:
                            issues.append(f"{file_path}: {key} = {value}")
                else:
                    self._check_timestamps_recursive(value, file_path, issues)
        elif isinstance(data, list):
            for item in data:
                self._check_timestamps_recursive(item, file_path, issues)

    def _check_dead_code_and_todos(self):
        """Cherche le code mort et les TODOs"""
        todos = []
        potential_dead_code = []

        for py_file in self.project_root.rglob("*.py"):
            if "__pycache__" in str(py_file) or "venv" in str(py_file):
                continue

            try:
                with open(py_file, encoding="utf-8") as f:
                    lines = f.readlines()
            except Exception:
                continue

            for i, line in enumerate(lines, 1):
                # Chercher les TODOs
                if any(marker in line for marker in ["TODO", "FIXME", "XXX", "HACK"]):
                    todos.append(
                        {
                            "file": str(py_file.relative_to(self.project_root)),
                            "line": i,
                            "text": line.strip(),
                        }
                    )

                # Chercher le code commenté (potentiellement mort)
                if line.strip().startswith("#") and len(line.strip()) > 30:
                    if any(keyword in line for keyword in ["def ", "class ", "import "]):
                        potential_dead_code.append(
                            {
                                "file": str(py_file.relative_to(self.project_root)),
                                "line": i,
                                "code": line.strip(),
                            }
                        )

        self.report["todos"] = todos[:20]  # Limiter
        self.report["potential_dead_code"] = potential_dead_code[:10]

    def _check_resource_usage(self):
        """Vérifie l'utilisation des ressources système"""
        if PSUTIL_AVAILABLE:
            try:
                # Utilisation CPU et mémoire du processus actuel
                process = psutil.Process()
                cpu_percent = process.cpu_percent(interval=0.1)
                memory_info = process.memory_info()

                # Vérifier si les ressources sont excessives
                if memory_info.rss > 500_000_000:  # 500MB
                    self.report["warnings"].append(f"Utilisation mémoire élevée: {memory_info.rss / 1_000_000:.1f}MB")

                # Compter les threads
                thread_count = process.num_threads()
                if thread_count > 50:
                    self.report["warnings"].append(f"Nombre de threads élevé: {thread_count}")

            except Exception as e:
                self.report["warnings"].append(f"Impossible de vérifier les ressources: {e}")
        else:
            # Alternative sans psutil
            try:
                # Vérifier la taille totale du projet
                total_size = sum(f.stat().st_size for f in self.project_root.rglob("*") if f.is_file())
                if total_size > 1_000_000_000:  # 1GB
                    self.report["warnings"].append(f"Taille totale du projet élevée: {total_size / 1_000_000:.1f}MB")
            except Exception:
                pass

    def _determine_overall_status(self):
        """Détermine le statut global du système"""
        critical_issues = len(self.report["issues"])
        warnings = len(self.report["warnings"])

        if critical_issues > 0:
            self.report["status"] = "critical"
        elif warnings > 10:
            self.report["status"] = "warning"
        elif warnings > 0:
            self.report["status"] = "good"
        else:
            self.report["status"] = "excellent"

        # Ajouter un résumé
        self.report["summary"] = {
            "critical_issues": critical_issues,
            "warnings": warnings,
            "unused_imports": len(self.report["unused_imports"]),
            "circular_dependencies": len(self.report["circular_dependencies"]),
            "performance_issues": len(self.report["performance_issues"]),
            "todos": len(self.report.get("todos", [])),
            "recommendations": self._generate_recommendations(),
        }

    def _generate_recommendations(self) -> list[str]:
        """Génère des recommandations basées sur l'analyse"""
        recommendations = []

        if len(self.report["unused_imports"]) > 10:
            recommendations.append("Nettoyer les imports non utilisés avec 'autoflake'")

        if len(self.report["circular_dependencies"]) > 0:
            recommendations.append("Refactoriser pour éliminer les dépendances circulaires")

        if len(self.report["performance_issues"]) > 5:
            recommendations.append("Optimiser les boucles et opérations coûteuses")

        if any("memory_size" in issue.get("type", "") for issue in self.report.get("memory_issues", [])):
            recommendations.append("Implémenter un système d'archivage pour les vieilles données")

        return recommendations

    def generate_html_report(self, output_path: str = "diagnostic_report.html"):
        """Génère un rapport HTML visuel"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <title>Diagnostic Jeffrey - {datetime.now().strftime('%Y-%m-%d %H:%M')}</title>
        <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; text-align: center; }}
        .status {{ padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; margin: 20px 0; }}
        .status.excellent {{ background-color: #4CAF50; color: white; }}
        .status.good {{ background-color: #8BC34A; color: white; }}
        .status.warning {{ background-color: #FF9800; color: white; }}
        .status.critical {{ background-color: #F44336; color: white; }}
        .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
        .issue {{ background-color: #ffebee; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .warning {{ background-color: #fff3e0; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        .info {{ background-color: #e3f2fd; padding: 10px; margin: 5px 0; border-radius: 3px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
        th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f2f2f2; font-weight: bold; }}
        code {{ background-color: #f5f5f5; padding: 2px 4px; border-radius: 3px; font-family: monospace; }}
        </style>
        </head>
        <body>
        <div class="container">
        <h1>🏥 Diagnostic Système Jeffrey</h1>
        <div class="status {self.report['status']}">
        Statut: {self.report['status'].upper()}
        </div>

        <div class="section">
        <h2>📊 Résumé</h2>
        <table>
        <tr><th>Métrique</th><th>Valeur</th></tr>
        <tr><td>Issues critiques</td><td>{self.report['summary']['critical_issues']}</td></tr>
        <tr><td>Avertissements</td><td>{self.report['summary']['warnings']}</td></tr>
        <tr><td>Imports non utilisés</td><td>{self.report['summary']['unused_imports']}</td></tr>
        <tr><td>Dépendances circulaires</td><td>{self.report['summary']['circular_dependencies']}</td></tr>
        <tr><td>Problèmes de performance</td><td>{self.report['summary']['performance_issues']}</td></tr>
        <tr><td>TODOs</td><td>{self.report['summary']['todos']}</td></tr>
        </table>
        </div>

        <div class="section">
        <h2>🚨 Issues Critiques</h2>
        {"".join(f'<div class="issue">{issue}</div>' for issue in self.report['issues']) if self.report['issues'] else '<div class="info">Aucune issue critique détectée</div>'}
        </div>

        <div class="section">
        <h2>⚠️ Avertissements</h2>
        {"".join(f'<div class="warning">{warning}</div>' for warning in self.report['warnings'][:10]) if self.report['warnings'] else '<div class="info">Aucun avertissement</div>'}
        </div>

        <div class="section">
        <h2>💡 Recommandations</h2>
        <ul>
        {"".join(f'<li>{rec}</li>' for rec in self.report['summary']['recommendations'])}
        </ul>
        </div>

        <div class="section">
        <h2>📝 Détails Techniques</h2>
        <details>
        <summary>Imports non utilisés ({len(self.report['unused_imports'])})</summary>
        <table>
        <tr><th>Fichier</th><th>Import</th><th>Ligne</th></tr>
        {"".join(f'<tr><td><code>{imp["file"]}</code></td><td><code>{imp["import"]}</code></td><td>{imp["line"]}</td></tr>' for imp in self.report['unused_imports'][:10])}
        </table>
        </details>

        <details>
        <summary>Problèmes de performance ({len(self.report['performance_issues'])})</summary>
        <table>
        <tr><th>Fichier</th><th>Type</th><th>Ligne</th><th>Code</th></tr>
        {"".join(f'<tr><td><code>{perf["file"]}</code></td><td>{perf["type"]}</td><td>{perf["line"]}</td><td><code>{perf["code"]}</code></td></tr>' for perf in self.report['performance_issues'][:10])}
        </table>
        </details>
        </div>

        <div style="text-align: center; color: #666; margin-top: 40px;">
        <p>Généré le {datetime.now().strftime('%Y-%m-%d à %H:%M:%S')}</p>
        </div>
        </div>
        </body>
        </html>
        """

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        print(f"\n📄 Rapport HTML généré: {output_path}")

    def print_summary(self):
        """Affiche un résumé dans la console"""
        print("\n" + "=" * 60)
        print("📊 RÉSUMÉ DU DIAGNOSTIC SYSTÈME JEFFREY")
        print("=" * 60)

        status_emoji = {"excellent": "✅", "good": "🟢", "warning": "🟡", "critical": "🔴"}

        print(f"\nStatut Global: {status_emoji.get(self.report['status'], '❓')} {self.report['status'].upper()}")
        print(f"\nIssues critiques: {self.report['summary']['critical_issues']}")
        print(f"Avertissements: {self.report['summary']['warnings']}")
        print(f"Imports non utilisés: {self.report['summary']['unused_imports']}")
        print(f"Dépendances circulaires: {self.report['summary']['circular_dependencies']}")
        print(f"Problèmes de performance: {self.report['summary']['performance_issues']}")
        print(f"TODOs: {self.report['summary']['todos']}")

        if self.report["summary"]["recommendations"]:
            print("\n💡 Recommandations:")
            for i, rec in enumerate(self.report["summary"]["recommendations"], 1):
                print(f"   {i}. {rec}")

        print("\n" + "=" * 60)


if __name__ == "__main__":
    # Test du diagnostic
    health_checker = JeffreySystemHealth()
    report = health_checker.run_full_diagnostic()

    # Afficher le résumé
    health_checker.print_summary()

    # Générer le rapport HTML
    health_checker.generate_html_report()

    # Sauvegarder le rapport JSON
    with open("diagnostic_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
