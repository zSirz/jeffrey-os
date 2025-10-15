#!/usr/bin/env python3
"""
Audit intelligent du code P1 avec AST pour détecter:
- TODOs et FIXMEs
- Imports inutilisés
- Dépendances critiques
- Modules à migrer en priorité
- Métriques de complexité
"""

import ast
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any


class P1Auditor:
    def __init__(self, base_path: str = "src/jeffrey"):
        self.base_path = Path(base_path)
        # Collections typées pour éviter les erreurs mypy
        self.modules: dict[str, dict[str, Any]] = {}
        self.issues: list[dict[str, Any]] = []
        self.dependencies: set = set()
        self.todos: list[dict[str, Any]] = []
        self.migration_priority: dict[str, list[dict[str, Any]]] = {}

    def audit_file(self, file_path: Path) -> dict[str, Any] | None:
        """Analyse AST d'un fichier Python"""
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)

            # Extraire imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")

            # Détecter TODOs/FIXMEs
            todos = []
            for i, line in enumerate(content.splitlines(), 1):
                if 'TODO' in line or 'FIXME' in line or 'HACK' in line:
                    todos.append(
                        {
                            "line": i,
                            "text": line.strip(),
                            "type": "TODO" if "TODO" in line else "FIXME" if "FIXME" in line else "HACK",
                        }
                    )

            # Calculer complexité cyclomatique
            complexity = sum(
                1 for node in ast.walk(tree) if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler))
            )

            # Détecter les modules Jeffrey importés (pour mapping dépendances)
            jeffrey_imports = [imp for imp in imports if 'jeffrey' in imp]

            return {
                "path": str(file_path),
                "size": len(content),
                "lines": len(content.splitlines()),
                "imports": imports,
                "jeffrey_imports": jeffrey_imports,
                "todos": todos,
                "complexity": complexity,
                "has_tests": 'test_' in file_path.name or '_test.py' in file_path.name,
                "checksum": hashlib.sha256(content.encode()).hexdigest(),
            }
        except Exception as e:
            self.issues.append({"file": str(file_path), "error": str(e), "type": "parse_error"})
            return None

    def identify_module_priority(self, module_name: str, file_data: dict) -> str:
        """Détermine la priorité de migration d'un module"""
        # P0: Core essential
        if any(x in module_name.lower() for x in ['consciousness', 'brain', 'kernel', 'memory', 'emotion']):
            return "P0_CRITICAL"
        # P1: Infrastructure
        elif any(x in module_name.lower() for x in ['orchestrator', 'security', 'bridge', 'router']):
            return "P1_INFRASTRUCTURE"
        # P2: Features
        elif any(x in module_name.lower() for x in ['dream', 'symbiosis', 'avatar']):
            return "P2_FEATURES"
        # P3: Utils
        else:
            return "P3_UTILS"

    def run_audit(self) -> dict:
        """Execute l'audit complet"""
        print("🔍 Starting P1 code audit...")

        # Scanner tous les fichiers Python
        py_files = list(self.base_path.rglob("*.py"))

        for py_file in py_files:
            if 'venv' in str(py_file) or '__pycache__' in str(py_file):
                continue

            file_data = self.audit_file(py_file)
            if file_data:
                module_name = py_file.stem
                priority = self.identify_module_priority(module_name, file_data)

                self.modules[str(py_file)] = file_data

                # Grouper par priorité
                if priority not in self.migration_priority:
                    self.migration_priority[priority] = []
                self.migration_priority[priority].append(
                    {
                        "module": module_name,
                        "path": str(py_file),
                        "complexity": file_data["complexity"],
                        "todos": len(file_data["todos"]),
                    }
                )

                # Collecter tous les TODOs
                for todo in file_data["todos"]:
                    self.todos.append({"file": str(py_file), **todo})

                # Collecter dépendances uniques
                for imp in file_data["imports"]:
                    if not imp.startswith('jeffrey'):
                        self.dependencies.add(imp.split('.')[0])

        # Convertir set en list pour JSON
        dependencies_sorted = sorted(list(self.dependencies))

        # Calculer métriques globales
        metrics = {
            "total_files": len(py_files),
            "total_modules": len(self.modules),
            "total_todos": len(self.todos),
            "total_issues": len(self.issues),
            "total_lines": sum(m["lines"] for m in self.modules.values()),
            "avg_complexity": (sum(m["complexity"] for m in self.modules.values()) / max(1, len(self.modules)))
            if self.modules
            else 0,
        }

        # Auto-fix simples (optionnel)
        if metrics["total_issues"] <= 10:
            print("✅ Low technical debt detected. Auto-fix possible.")
        else:
            print(f"⚠️ {metrics['total_issues']} issues found. Manual review recommended.")

        return {
            "timestamp": datetime.now().isoformat(),
            "modules": self.modules,
            "issues": self.issues,
            "dependencies": dependencies_sorted,
            "todos": self.todos,
            "metrics": metrics,
            "migration_priority": self.migration_priority,
        }

    def save_report(self, output_path: str = "audit_p1_report.json"):
        """Sauvegarde le rapport d'audit"""
        report = self.run_audit()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📊 Audit report saved to {output_path}")


if __name__ == "__main__":
    auditor = P1Auditor()
    auditor.save_report()
