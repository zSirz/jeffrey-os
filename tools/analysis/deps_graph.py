# tools/analysis/deps_graph.py
import ast
from pathlib import Path

import networkx as nx


class DependencyAnalyzer:
    """Analyse les dépendances et détecte les cycles"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.modules = {}

    def analyze_file(self, filepath: Path) -> set[str]:
        """Extrait les dépendances d'un fichier"""
        deps = set()
        try:
            with open(filepath, encoding='utf-8') as f:
                tree = ast.parse(f.read())

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    if node.module and 'jeffrey' in node.module:
                        deps.add(node.module)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if 'jeffrey' in alias.name:
                            deps.add(alias.name)

            return deps
        except:
            return set()

    def build_graph(self, base_path: str = "src/jeffrey"):
        """Construit le graphe de dépendances"""
        for py_file in Path(base_path).rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue

            module_name = str(py_file).replace("/", ".").replace(".py", "")
            deps = self.analyze_file(py_file)

            self.modules[module_name] = deps
            self.graph.add_node(module_name)

            for dep in deps:
                self.graph.add_edge(module_name, dep)

    def find_cycles(self) -> list[list[str]]:
        """Trouve tous les cycles dans le graphe"""
        try:
            cycles = list(nx.simple_cycles(self.graph))
            return cycles
        except:
            return []

    def critical_cycles(self) -> list[list[str]]:
        """Trouve les cycles critiques (core modules)"""
        critical_modules = ['bus', 'security', 'orchestrator', 'memory', 'consciousness']
        cycles = self.find_cycles()

        critical = []
        for cycle in cycles:
            if any(crit in str(cycle) for crit in critical_modules):
                critical.append(cycle)

        return critical
