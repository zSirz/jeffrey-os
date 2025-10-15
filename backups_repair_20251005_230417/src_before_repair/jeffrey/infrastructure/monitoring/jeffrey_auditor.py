"""
Module de infrastructure système de base pour Jeffrey OS.

Ce module implémente les fonctionnalités essentielles pour module de infrastructure système de base pour jeffrey os.
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
import gzip
import json
import logging
import math
import os
import statistics
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

try:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # Backend sans GUI
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@dataclass
class CodeMetrics:
    """Métriques détaillées pour un fichier de code"""

    file_path: str
    lines_of_code: int
    cyclomatic_complexity: int
    halstead_complexity: dict[str, float]
    maintainability_index: float
    estimated_coverage: float
    security_score: float
    documentation_score: float
    code_smells: list[str]
    suggestions: list[str]


@dataclass
class ProjectAuditResult:
    """Résultat complet d'audit de projet"""

    timestamp: datetime
    project_path: str
    total_files: int
    total_lines: int
    overall_scores: dict[str, float]
    files_metrics: list[CodeMetrics]
    trends: dict[str, Any]
    recommendations: list[str]
    visualization_data: dict[str, Any]


class JeffreyAuditor:
    """
    Auditeur avec analyses AST profondes et visualisations flexibles
    """

    def __init__(self, output_dir: str = "audit_reports", project_path: str = ".") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.project_path = Path(project_path)
        self.log_dir = Path("logs/audit")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Configuration des seuils
        self.thresholds = {
            "complexity": {"good": 10, "warning": 20, "critical": 30},
            "maintainability": {"good": 70, "warning": 50, "critical": 25},
            "coverage": {"good": 80, "warning": 60, "critical": 40},
            "security": {"good": 80, "warning": 60, "critical": 40},
            "documentation": {"good": 80, "warning": 60, "critical": 40},
        }

        # Setup logging
        self._setup_logging()

        # Détection environnement
        self.environment = self._detect_environment()

    def _setup_logging(self):
        """Configuration du logging"""
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m')}.log"

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def _detect_environment(self) -> str:
        """Détecte l'environnement d'exécution"""
        if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
            return "gui"
        elif sys.stdout.isatty():
            return "terminal"
        else:
            return "script"

    def audit_project(self, output_format: str = "auto") -> ProjectAuditResult:
        """
        Audit complet du projet
        output_format: 'gui', 'ascii', 'json', 'auto' (détecte environnement)
        """

        if output_format == "auto":
            output_format = self._choose_output_format()

        self.logger.info(f"Début audit projet: {self.project_path}")
        start_time = datetime.now()

        # Découverte des fichiers Python
        python_files = self._discover_python_files()
        self.logger.info(f"Fichiers Python trouvés: {len(python_files)}")

        # Analyse de chaque fichier
        files_metrics = []
        for file_path in python_files:
            try:
                metrics = self._analyze_file(file_path)
                files_metrics.append(metrics)
            except Exception as e:
                self.logger.error(f"Erreur analyse {file_path}: {e}")

        # Calcul des scores globaux
        overall_scores = self._calculate_overall_scores(files_metrics)

        # Analyse des tendances
        trends = self._analyze_trends(files_metrics)

        # Recommandations
        recommendations = self._generate_recommendations(files_metrics, overall_scores)

        # Données de visualisation
        visualization_data = self._prepare_visualization_data(files_metrics)

        # Résultat final
        result = ProjectAuditResult(
            timestamp=datetime.now(),
            project_path=str(self.project_path),
            total_files=len(files_metrics),
            total_lines=sum(m.lines_of_code for m in files_metrics),
            overall_scores=overall_scores,
            files_metrics=files_metrics,
            trends=trends,
            recommendations=recommendations,
            visualization_data=visualization_data,
        )

        # Génération du rapport selon le format
        if output_format == "gui":
            self._generate_gui_report(result)
        elif output_format == "ascii":
            self._generate_ascii_report(result)
        elif output_format == "json":
            self._generate_json_report(result)

        # Sauvegarde
        self._save_audit_result(result)

        duration = datetime.now() - start_time
        self.logger.info(f"Audit terminé en {duration.total_seconds():.2f}s")

        return result

    def _choose_output_format(self) -> str:
        """Choix automatique du format selon l'environnement"""
        if self.environment == "gui" and HAS_MATPLOTLIB:
            return "gui"
        elif self.environment == "terminal":
            return "ascii"
        else:
            return "json"

    def _discover_python_files(self) -> list[Path]:
        """Découverte des fichiers Python à analyser"""
        python_files = []

        # Extensions Python
        python_extensions = {".py", ".pyw", ".pyi"}

        # Dossiers à ignorer
        ignore_dirs = {
            "__pycache__",
            ".git",
            ".svn",
            ".hg",
            "node_modules",
            ".venv",
            "venv",
            "env",
            ".env",
            "build",
            "dist",
            ".pytest_cache",
            ".mypy_cache",
            ".tox",
        }

        for root, dirs, files in os.walk(self.project_path):
            # Filtrer les dossiers à ignorer
            dirs[:] = [d for d in dirs if d not in ignore_dirs]

            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in python_extensions:
                    python_files.append(file_path)

        return sorted(python_files)

    def _analyze_file(self, file_path: Path) -> CodeMetrics:
        """Analyse complète d'un fichier Python"""

        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            self.logger.error(f"Erreur lecture {file_path}: {e}")
            return self._create_empty_metrics(str(file_path))

        # Parsing AST
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self.logger.error(f"Erreur syntaxe {file_path}: {e}")
            return self._create_empty_metrics(str(file_path))

        # Calcul des métriques
        lines_of_code = len([line for line in content.split("\n") if line.strip() and not line.strip().startswith("#")])
        cyclomatic_complexity = self._calculate_cyclomatic_complexity(tree)
        halstead_complexity = self._calculate_halstead_complexity(tree)
        maintainability_index = self._calculate_maintainability_index(
            lines_of_code, cyclomatic_complexity, halstead_complexity
        )
        estimated_coverage = self._estimate_coverage_ast(tree)
        security_score = self._analyze_security_ast(tree, content)
        documentation_score = self._analyze_documentation(tree, content)
        code_smells = self._detect_code_smells(tree, content)
        suggestions = self._generate_suggestions(tree, content)

        return CodeMetrics(
            file_path=str(file_path),
            lines_of_code=lines_of_code,
            cyclomatic_complexity=cyclomatic_complexity,
            halstead_complexity=halstead_complexity,
            maintainability_index=maintainability_index,
            estimated_coverage=estimated_coverage,
            security_score=security_score,
            documentation_score=documentation_score,
            code_smells=code_smells,
            suggestions=suggestions,
        )

    def _create_empty_metrics(self, file_path: str) -> CodeMetrics:
        """Crée des métriques vides en cas d'erreur"""
        return CodeMetrics(
            file_path=file_path,
            lines_of_code=0,
            cyclomatic_complexity=0,
            halstead_complexity={"volume": 0, "difficulty": 0, "effort": 0},
            maintainability_index=0,
            estimated_coverage=0,
            security_score=0,
            documentation_score=0,
            code_smells=["PARSE_ERROR"],
            suggestions=["Vérifier la syntaxe du fichier"],
        )

    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calcul de la complexité cyclomatique"""
        complexity = 1  # Base

        for node in ast.walk(tree):
            # Nœuds qui augmentent la complexité
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
            elif isinstance(node, (ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1
            elif isinstance(node, ast.comprehension):
                complexity += 1

        return complexity

    def _calculate_halstead_complexity(self, tree: ast.AST) -> dict[str, float]:
        """Calcul des métriques de Halstead"""
        operators = set()
        operands = set()
        operator_count = 0
        operand_count = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp):
                operators.add(type(node.op).__name__)
                operator_count += 1
            elif isinstance(node, ast.UnaryOp):
                operators.add(type(node.op).__name__)
                operator_count += 1
            elif isinstance(node, ast.Compare):
                for op in node.ops:
                    operators.add(type(op).__name__)
                    operator_count += 1
            elif isinstance(node, ast.Name):
                operands.add(node.id)
                operand_count += 1
            elif isinstance(node, ast.Constant):
                operands.add(str(node.value))
                operand_count += 1

        n1 = len(operators)  # Nombre d'opérateurs uniques
        n2 = len(operands)  # Nombre d'opérandes uniques
        N1 = operator_count  # Nombre total d'opérateurs
        N2 = operand_count  # Nombre total d'opérandes

        # Calculs Halstead
        vocabulary = n1 + n2
        length = N1 + N2

        if n1 == 0 or n2 == 0:
            return {"volume": 0, "difficulty": 0, "effort": 0}

        volume = length * math.log2(vocabulary) if vocabulary > 0 else 0
        difficulty = (n1 / 2) * (N2 / n2) if n2 > 0 else 0
        effort = difficulty * volume

        return {"volume": volume, "difficulty": difficulty, "effort": effort}

    def _calculate_maintainability_index(self, loc: int, complexity: int, halstead: dict[str, float]) -> float:
        """Calcul de l'indice de maintenabilité"""
        if loc == 0:
            return 0

        # Formule de l'indice de maintenabilité
        halstead_volume = halstead.get("volume", 0)

        mi = 171 - 5.2 * math.log(halstead_volume) - 0.23 * complexity - 16.2 * math.log(loc)
        mi = max(0, min(100, mi))  # Normalisation 0-100

        return mi

    def _estimate_coverage_ast(self, tree: ast.AST) -> float:
        """
        Estimation coverage via AST sans outil externe
        - Compte branches exécutables
        - Détecte tests associés
        - Score probabiliste
        """

        # Comptage des branches exécutables
        branches = 0
        test_indicators = 0

        for node in ast.walk(tree):
            # Branches conditionnelles
            if isinstance(node, (ast.If, ast.While, ast.For)):
                branches += 1
            elif isinstance(node, ast.Try):
                branches += len(node.handlers)

            # Indicateurs de tests
            if isinstance(node, ast.FunctionDef):
                if node.name.startswith("test_") or "test" in node.name.lower():
                    test_indicators += 1

            # Assertions (indicateur de tests)
            if isinstance(node, ast.Assert):
                test_indicators += 1

        # Score probabiliste basé sur les indicateurs
        if branches == 0:
            return 90.0  # Pas de branches = facile à couvrir

        # Estimation basée sur le ratio tests/branches
        test_ratio = test_indicators / branches if branches > 0 else 0
        base_coverage = min(90, test_ratio * 100)

        # Ajustement selon la complexité
        if branches > 20:
            base_coverage *= 0.8
        elif branches > 10:
            base_coverage *= 0.9

        return max(0, min(100, base_coverage))

    def _analyze_security_ast(self, tree: ast.AST, content: str) -> float:
        """Analyse de sécurité basée sur l'AST"""
        security_issues = []

        # Patterns de sécurité à détecter
        dangerous_imports = {"subprocess", "os", "sys", "eval", "exec"}
        dangerous_functions = {"eval", "exec", "compile", "open"}

        for node in ast.walk(tree):
            # Imports dangereux
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in dangerous_imports:
                        security_issues.append(f"Import potentiellement dangereux: {alias.name}")

            # Appels de fonctions dangereuses
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in dangerous_functions:
                    security_issues.append(f"Appel fonction dangereuse: {node.func.id}")

            # Hardcoded secrets (patterns basiques)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if any(pattern in node.value.lower() for pattern in ["password", "secret", "token", "key"]):
                    if len(node.value) > 8:  # Probable secret
                        security_issues.append("Possible secret hardcodé")

        # Vérification du contenu pour patterns supplémentaires
        content_lower = content.lower()
        if "sql" in content_lower and any(op in content_lower for op in ["select", "insert", "update", "delete"]):
            if "format" in content_lower or "%" in content:
                security_issues.append("Possible injection SQL")

        # Score inversé (moins d'issues = score plus élevé)
        max_issues = 10
        security_score = max(0, (max_issues - len(security_issues)) / max_issues * 100)

        return security_score

    def _analyze_documentation(self, tree: ast.AST, content: str) -> float:
        """Analyse de la documentation"""

        functions = []
        classes = []
        documented_functions = 0
        documented_classes = 0

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node)
                if ast.get_docstring(node):
                    documented_functions += 1

            elif isinstance(node, ast.ClassDef):
                classes.append(node)
                if ast.get_docstring(node):
                    documented_classes += 1

        # Calcul du score de documentation
        total_items = len(functions) + len(classes)
        if total_items == 0:
            return 100.0  # Pas de fonctions/classes = pas besoin de doc

        documented_items = documented_functions + documented_classes
        doc_score = (documented_items / total_items) * 100

        # Bonus pour documentation du module
        if ast.get_docstring(tree):
            doc_score = min(100, doc_score + 10)

        return doc_score

    def _detect_code_smells(self, tree: ast.AST, content: str) -> list[str]:
        """Détection des code smells"""
        smells = []

        # Analyse des fonctions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Fonction trop longue
                if len(node.body) > 20:
                    smells.append(f"Fonction trop longue: {node.name}")

                # Trop de paramètres
                if len(node.args.args) > 5:
                    smells.append(f"Trop de paramètres: {node.name}")

            # Variables avec noms courts
            elif isinstance(node, ast.Name) and len(node.id) == 1:
                if node.id not in ["i", "j", "k", "x", "y", "z"]:  # Exceptions communes
                    smells.append(f"Nom de variable trop court: {node.id}")

        # Lignes trop longues
        lines = content.split("\n")
        for i, line in enumerate(lines, 1):
            if len(line) > 100:
                smells.append(f"Ligne trop longue: {i}")

        return smells[:10]  # Limiter à 10 premiers smells

    def _generate_suggestions(self, tree: ast.AST, content: str) -> list[str]:
        """Génération de suggestions d'amélioration"""
        suggestions = []

        # Analyse des imports
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend([alias.name for alias in node.names])

        # Suggestions basées sur les imports
        if "requests" in imports and "urllib3" not in imports:
            suggestions.append("Considérer urllib3 pour de meilleures performances HTTP")

        # Suggestions basées sur la structure
        function_count = len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)])
        if function_count > 10:
            suggestions.append("Considérer diviser ce fichier en modules plus petits")

        # Suggestions de typing
        if "typing" not in imports and function_count > 0:
            suggestions.append("Ajouter des annotations de type pour améliorer la lisibilité")

        return suggestions[:5]  # Limiter à 5 suggestions

    def _calculate_overall_scores(self, files_metrics: list[CodeMetrics]) -> dict[str, float]:
        """Calcul des scores globaux du projet"""
        if not files_metrics:
            return {}

        scores = {
            "complexity": statistics.mean([100 - min(100, m.cyclomatic_complexity * 3) for m in files_metrics]),
            "maintainability": statistics.mean([m.maintainability_index for m in files_metrics]),
            "coverage": statistics.mean([m.estimated_coverage for m in files_metrics]),
            "security": statistics.mean([m.security_score for m in files_metrics]),
            "documentation": statistics.mean([m.documentation_score for m in files_metrics]),
            "quality": 0,  # Calculé après
        }

        # Score de qualité global
        scores["quality"] = statistics.mean(
            [
                scores["complexity"],
                scores["maintainability"],
                scores["coverage"],
                scores["security"],
                scores["documentation"],
            ]
        )

        return scores

    def _analyze_trends(self, files_metrics: list[CodeMetrics]) -> dict[str, Any]:
        """Analyse des tendances dans le code"""

        # Distribution de la complexité
        complexities = [m.cyclomatic_complexity for m in files_metrics]

        trends = {
            "complexity_distribution": {
                "mean": statistics.mean(complexities) if complexities else 0,
                "median": statistics.median(complexities) if complexities else 0,
                "std": statistics.stdev(complexities) if len(complexities) > 1 else 0,
                "max": max(complexities) if complexities else 0,
            },
            "files_by_quality": {
                "excellent": len([m for m in files_metrics if m.maintainability_index >= 70]),
                "good": len([m for m in files_metrics if 50 <= m.maintainability_index < 70]),
                "poor": len([m for m in files_metrics if m.maintainability_index < 50]),
            },
            "common_smells": Counter([smell for metrics in files_metrics for smell in metrics.code_smells]).most_common(
                5
            ),
        }

        return trends

    def _generate_recommendations(
        self, files_metrics: list[CodeMetrics], overall_scores: dict[str, float]
    ) -> list[str]:
        """Génération de recommandations globales"""
        recommendations = []

        # Recommandations basées sur les scores
        if overall_scores.get("complexity", 0) < 60:
            recommendations.append("🔧 Réduire la complexité cyclomatique en refactorant les fonctions complexes")

        if overall_scores.get("documentation", 0) < 70:
            recommendations.append("📝 Améliorer la documentation en ajoutant des docstrings")

        if overall_scores.get("security", 0) < 70:
            recommendations.append("🔒 Réviser les pratiques de sécurité et éviter les fonctions dangereuses")

        if overall_scores.get("coverage", 0) < 60:
            recommendations.append("🧪 Augmenter la couverture de tests")

        # Recommandations basées sur les fichiers les plus problématiques
        worst_files = sorted(files_metrics, key=lambda m: m.maintainability_index)[:3]
        if worst_files:
            recommendations.append(
                f"🎯 Prioriser la refactorisation de: {', '.join([Path(f.file_path).name for f in worst_files])}"
            )

        return recommendations

    def _prepare_visualization_data(self, files_metrics: list[CodeMetrics]) -> dict[str, Any]:
        """Préparation des données pour visualisation"""

        return {
            "complexity_by_file": [(Path(m.file_path).name, m.cyclomatic_complexity) for m in files_metrics],
            "maintainability_distribution": [m.maintainability_index for m in files_metrics],
            "coverage_by_file": [(Path(m.file_path).name, m.estimated_coverage) for m in files_metrics],
            "security_scores": [m.security_score for m in files_metrics],
            "documentation_scores": [m.documentation_score for m in files_metrics],
            "lines_of_code": [m.lines_of_code for m in files_metrics],
        }

    def _generate_ascii_viz(self, metrics: dict[str, Any]) -> str:
        """
        Visualisations ASCII pour environnements sans GUI
        """

        output = []
        output.append("📊 VISUALISATION ASCII")
        output.append("=" * 50)

        # Graphique en barres ASCII pour la complexité
        output.append("\n🔧 COMPLEXITÉ PAR FICHIER:")
        complexity_data = metrics.get("complexity_by_file", [])

        if complexity_data:
            max_complexity = max(comp for _, comp in complexity_data)
            max_width = 40

            for filename, complexity in complexity_data[:10]:  # Top 10
                bar_width = int((complexity / max_complexity) * max_width) if max_complexity > 0 else 0
                bar = "█" * bar_width
                output.append(f"{filename[:25]:<25} {bar} {complexity}")

        # Distribution de la maintenabilité
        output.append("\n📈 DISTRIBUTION MAINTENABILITÉ:")
        maintainability_scores = metrics.get("maintainability_distribution", [])

        if maintainability_scores:
            # Histogramme ASCII
            bins = [0, 25, 50, 70, 100]
            counts = [0] * (len(bins) - 1)

            for score in maintainability_scores:
                for i in range(len(bins) - 1):
                    if bins[i] <= score < bins[i + 1]:
                        counts[i] += 1
                        break

            max_count = max(counts) if counts else 1

            for i, count in enumerate(counts):
                bar_width = int((count / max_count) * 20) if max_count > 0 else 0
                bar = "▓" * bar_width
                range_label = f"{bins[i]}-{bins[i + 1]}"
                output.append(f"{range_label:<8} {bar} {count}")

        return "\n".join(output)

    def _generate_gui_report(self, result: ProjectAuditResult):
        """Génération du rapport avec visualisations GUI"""

        if not HAS_MATPLOTLIB:
            self.logger.warning("Matplotlib non disponible, utilisation du format ASCII")
            self._generate_ascii_report(result)
            return

        # Création des graphiques
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Audit Jeffrey OS - {result.project_path}", fontsize=16)

        # Graphique 1: Complexité par fichier
        complexity_data = result.visualization_data.get("complexity_by_file", [])
        if complexity_data:
            files, complexities = zip(*complexity_data[:10])
            axes[0, 0].bar(range(len(files)), complexities)
            axes[0, 0].set_title("Complexité Cyclomatique")
            axes[0, 0].set_xticks(range(len(files)))
            axes[0, 0].set_xticklabels([f[:10] for f in files], rotation=45)

        # Graphique 2: Distribution maintenabilité
        maintainability = result.visualization_data.get("maintainability_distribution", [])
        if maintainability:
            axes[0, 1].hist(maintainability, bins=10, alpha=0.7)
            axes[0, 1].set_title("Distribution Maintenabilité")
            axes[0, 1].set_xlabel("Score")
            axes[0, 1].set_ylabel("Nombre de fichiers")

        # Graphique 3: Scores globaux
        scores = result.overall_scores
        if scores:
            score_names = list(scores.keys())
            score_values = list(scores.values())
            axes[1, 0].bar(score_names, score_values)
            axes[1, 0].set_title("Scores Globaux")
            axes[1, 0].set_ylim(0, 100)
            axes[1, 0].tick_params(axis="x", rotation=45)

        # Graphique 4: Couverture estimée
        coverage_data = result.visualization_data.get("coverage_by_file", [])
        if coverage_data:
            files, coverages = zip(*coverage_data[:10])
            axes[1, 1].bar(range(len(files)), coverages)
            axes[1, 1].set_title("Couverture Estimée")
            axes[1, 1].set_xticks(range(len(files)))
            axes[1, 1].set_xticklabels([f[:10] for f in files], rotation=45)
            axes[1, 1].set_ylim(0, 100)

        plt.tight_layout()

        # Sauvegarde
        report_path = self.output_dir / f"audit_gui_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(report_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Rapport GUI généré: {report_path}")

    def _generate_ascii_report(self, result: ProjectAuditResult):
        """Génération du rapport ASCII"""

        output = []
        output.append("🚀 JEFFREY OS - RAPPORT D'AUDIT")
        output.append("=" * 60)
        output.append(f"📅 Date: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        output.append(f"📁 Projet: {result.project_path}")
        output.append(f"📊 Fichiers analysés: {result.total_files}")
        output.append(f"📝 Lignes de code: {result.total_lines}")

        # Scores globaux
        output.append("\n📈 SCORES GLOBAUX:")
        output.append("-" * 30)

        for metric, score in result.overall_scores.items():
            emoji = "🟢" if score >= 70 else "🟡" if score >= 50 else "🔴"
            output.append(f"{emoji} {metric.capitalize():<15}: {score:.1f}%")

        # Visualisation ASCII
        ascii_viz = self._generate_ascii_viz(result.visualization_data)
        output.append(ascii_viz)

        # Recommandations
        output.append("\n💡 RECOMMANDATIONS:")
        output.append("-" * 30)

        for i, rec in enumerate(result.recommendations, 1):
            output.append(f"{i}. {rec}")

        # Fichiers les plus problématiques
        output.append("\n⚠️  FICHIERS À PRIORISER:")
        output.append("-" * 30)

        worst_files = sorted(result.files_metrics, key=lambda m: m.maintainability_index)[:5]
        for metrics in worst_files:
            filename = Path(metrics.file_path).name
            output.append(f"• {filename:<25} (Maintenabilité: {metrics.maintainability_index:.1f})")

        report_content = "\n".join(output)

        # Affichage console
        print(report_content)

        # Sauvegarde fichier
        report_path = self.output_dir / f"audit_report_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
        report_path.write_text(report_content, encoding="utf-8")

        self.logger.info(f"Rapport ASCII généré: {report_path}")

    def _generate_json_report(self, result: ProjectAuditResult):
        """Génération du rapport JSON pour dashboards"""

        report_data = asdict(result)
        report_data["timestamp"] = result.timestamp.isoformat()

        # Sauvegarde
        report_path = self.output_dir / f"audit_data_{result.timestamp.strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Rapport JSON généré: {report_path}")

    def _save_audit_result(self, result: ProjectAuditResult):
        """Sauvegarde des résultats d'audit"""

        # Fichier de log compressé
        log_file = self.log_dir / f"audit_results_{datetime.now().strftime('%Y%m')}.jsonl.gz"

        log_entry = {
            "timestamp": result.timestamp.isoformat(),
            "project_path": result.project_path,
            "total_files": result.total_files,
            "total_lines": result.total_lines,
            "overall_scores": result.overall_scores,
            "files_count_by_quality": result.trends.get("files_by_quality", {}),
            "recommendations_count": len(result.recommendations),
        }

        with gzip.open(log_file, "at", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

    def get_historical_trends(self, days: int = 30) -> dict[str, Any]:
        """Analyse des tendances historiques"""

        start_date = datetime.now() - timedelta(days=days)
        historical_data = []

        for log_file in self.log_dir.glob("audit_results_*.jsonl.gz"):
            try:
                with gzip.open(log_file, "rt", encoding="utf-8") as f:
                    for line in f:
                        entry = json.loads(line.strip())
                        entry_date = datetime.fromisoformat(entry["timestamp"])

                        if entry_date >= start_date:
                            historical_data.append(entry)

            except Exception as e:
                self.logger.error(f"Erreur lecture historique {log_file}: {e}")

        if not historical_data:
            return {"error": "Pas de données historiques disponibles"}

        # Analyse des tendances
        trends = {
            "total_audits": len(historical_data),
            "quality_evolution": [],
            "files_trend": [],
            "common_issues": [],
        }

        # Évolution de la qualité
        for entry in sorted(historical_data, key=lambda x: x["timestamp"]):
            trends["quality_evolution"].append(
                {
                    "date": entry["timestamp"][:10],
                    "quality_score": entry["overall_scores"].get("quality", 0),
                }
            )

        return trends


def main():
    """Test du module JeffreyAuditor"""

    auditor = JeffreyAuditor()

    print("🚀 LANCEMENT JEFFREY AUDITOR")
    print("=" * 50)

    # Audit du projet
    result = auditor.audit_project(output_format="ascii")

    print("\n✅ Audit terminé!")
    print(f"📊 Fichiers analysés: {result.total_files}")
    print(f"📈 Score qualité global: {result.overall_scores.get('quality', 0):.1f}%")

    # Tendances historiques
    trends = auditor.get_historical_trends(days=7)
    print(f"\n📈 Audits des 7 derniers jours: {trends.get('total_audits', 0)}")


if __name__ == "__main__":
    main()
