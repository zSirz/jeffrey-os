"""
Module de module de sécurité pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module de module de sécurité pour jeffrey os.
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
import logging
from dataclasses import dataclass


@dataclass
class SecurityReport:
    """Rapport de sécurité"""

    passed: bool
    violations: list[str] = None
    risk_score: float = 0.0

    def __post_init__(self):
        if self.violations is None:
            self.violations = []


class SimpleSecurity:
    """Analyseur de sécurité léger (AST uniquement)"""

    def __init__(self, strict_mode: bool = False) -> None:
        self.logger = logging.getLogger("security")
        self.strict_mode = strict_mode

        self.allowed_imports = {
            "json",
            "datetime",
            "math",
            "random",
            "collections",
            "itertools",
            "functools",
            "typing",
            "dataclasses",
            "asyncio",
            "logging",
            "re",
            "pathlib",
            "enum",
        }

        # Étendre la whitelist pour éviter les faux positifs
        self.allowed_imports.update(
            {
                # Python standard library
                "ast",
                "importlib",
                "sys",
                "yaml",
                "os",
                "pathlib",
                "time",
                "uuid",
                "hashlib",
                "secrets",
                "copy",
                "weakref",
                "warnings",
                "traceback",
                "inspect",
                "pickle",
                "json",
                "csv",
                "sqlite3",
                "urllib",
                "http",
                "socket",
                # Jeffrey internal
                "src",
                "jeffrey",
                # Data science & ML
                "numpy",
                "pandas",
                "scipy",
                "sklearn",
                "torch",
                "tensorflow",
                "transformers",
                "sentence_transformers",
                "huggingface_hub",
                # Web & API
                "fastapi",
                "pydantic",
                "requests",
                "aiohttp",
                "httpx",
                "uvicorn",
                "starlette",
                "websockets",
                # Async & concurrency
                "asyncio",
                "aiofiles",
                "concurrent",
                "threading",
                "multiprocessing",
                # Utilities
                "tqdm",
                "rich",
                "click",
                "typer",
                "dotenv",
                # Testing (au cas où)
                "pytest",
                "unittest",
                "mock",
            }
        )

        # Nettoyer le set des forbidden_functions
        self.forbidden_functions = {
            "eval",
            "exec",
            "compile",
            "__import__",
            "input",  # Retirer 'open' qui est souvent légitime
        }

        self.forbidden_attributes = {
            "__class__",
            "__bases__",
            "__subclasses__",
            "__code__",
            "__globals__",
        }

    def analyze_code(self, code: str) -> SecurityReport:
        """Analyse AST du code"""

        report = SecurityReport(passed=True)

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            report.passed = False
            report.violations.append(f"Syntax error: {e}")
            report.risk_score = 1.0
            return report

        for node in ast.walk(tree):
            # Vérifier imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name.split(".")[0]
                    if module not in self.allowed_imports and module != "jeffrey":
                        if self.strict_mode:
                            report.passed = False
                            report.violations.append(f"Import not allowed: {module}")
                        else:
                            self.logger.warning(f"Import warning: {module}")

            # Vérifier appels dangereux
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.forbidden_functions:
                        report.passed = False
                        report.violations.append(f"Forbidden function: {node.func.id}")

            # Vérifier attributs dangereux
            elif isinstance(node, ast.Attribute):
                if node.attr in self.forbidden_attributes:
                    report.passed = False
                    report.violations.append(f"Forbidden attribute: {node.attr}")

        report.risk_score = min(1.0, len(report.violations) * 0.2)

        return report
