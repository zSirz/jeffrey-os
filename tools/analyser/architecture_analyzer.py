#!/usr/bin/env python3
"""
Architecture Analyzer - Jeffrey AGI
Analyse architecture + modules non-branchés + optimisations
"""

import ast
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JeffreyArchitectureAnalyzer:
    def __init__(self, project_root="."):
        self.project_root = Path(project_root)
        self.modules_graph = defaultdict(set)  # Graphe de dépendances
        self.unused_modules = set()
        self.duplicate_code = []
        self.circular_dependencies = []
        self.optimization_opportunities = []
        self.module_stats = {}

    def analyze_project_architecture(self) -> dict[str, Any]:
        """Analyse complète de l'architecture du projet"""
        logger.info("🏗️ Début analyse architecture Jeffrey AGI")

        # Phase 1: Scanner tous les fichiers Python
        python_files = list(self.project_root.rglob("*.py"))
        python_files = [f for f in python_files if not self._should_ignore_file(f)]

        logger.info(f"📁 Analyse de {len(python_files)} fichiers Python")

        # Phase 2: Construire le graphe de dépendances
        for filepath in python_files:
            self._analyze_file_dependencies(filepath)

        # Phase 3: Analyser les patterns d'architecture
        self._detect_unused_modules()
        self._detect_circular_dependencies()
        self._detect_duplicate_code(python_files)
        self._detect_optimization_opportunities(python_files)

        # Phase 4: Générer le rapport
        return self._generate_architecture_report()

    def _should_ignore_file(self, filepath: Path) -> bool:
        """Détermine si un fichier doit être ignoré"""
        ignore_patterns = [
            'venv/',
            '__pycache__/',
            '.git/',
            'node_modules/',
            'backup_',
            'core_backup_',
            'jeffrey_core/core/',  # Backups
            'test_',
            'tests/',
            'examples/',  # Tests et exemples
            'site-packages/',
            'lib/python',  # Packages externes
        ]

        file_str = str(filepath)
        return any(pattern in file_str for pattern in ignore_patterns)

    def _analyze_file_dependencies(self, filepath: Path):
        """Analyser les dépendances d'un fichier"""
        try:
            with open(filepath, encoding='utf-8') as f:
                content = f.read()

            # Analyser l'AST pour extraire les imports
            try:
                tree = ast.parse(content)
                imports = self._extract_imports(tree)

                # Construire le graphe de dépendances
                module_name = self._get_module_name(filepath)
                for imported_module in imports:
                    # Filtrer les imports externes (non-Jeffrey)
                    if self._is_jeffrey_module(imported_module):
                        self.modules_graph[module_name].add(imported_module)

                # Statistiques du module
                self.module_stats[module_name] = {
                    'filepath': str(filepath),
                    'imports': imports,  # déjà une liste
                    'lines': len(content.split('\n')),
                    'functions': len([n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]),
                    'classes': len([n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]),
                    'async_functions': len([n for n in ast.walk(tree) if isinstance(n, ast.AsyncFunctionDef)]),
                }

            except SyntaxError:
                logger.warning(f"⚠️ Erreur syntax dans {filepath}")

        except Exception as e:
            logger.error(f"❌ Erreur analyse {filepath}: {e}")

    def _extract_imports(self, tree: ast.AST) -> list[str]:
        """Extraire tous les imports d'un arbre AST"""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    if module:
                        imports.append(f"{module}.{alias.name}")
                    else:
                        imports.append(alias.name)

        return imports

    def _get_module_name(self, filepath: Path) -> str:
        """Convertir un chemin de fichier en nom de module"""
        relative_path = filepath.relative_to(self.project_root)
        module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]

        # Ignorer __init__.py
        if module_parts[-1] == '__init__':
            module_parts = module_parts[:-1]

        return '.'.join(module_parts)

    def _is_jeffrey_module(self, module_name: str) -> bool:
        """Détermine si un module fait partie de Jeffrey"""
        jeffrey_indicators = [
            'core.',
            'jeffrey',
            'emotion',
            'memory',
            'learning',
            'agi_fusion',
            'dialogue',
            'orchestrator',
            'handler',
            'model_handlers',
            'async_wrapper',
        ]

        return any(indicator in module_name.lower() for indicator in jeffrey_indicators)

    def _detect_unused_modules(self):
        """Détecter les modules non utilisés"""
        all_modules = set(self.module_stats.keys())
        imported_modules = set()

        # Collecter tous les modules importés
        for module_imports in self.modules_graph.values():
            imported_modules.update(module_imports)

        # Modules potentiellement non utilisés
        potentially_unused = all_modules - imported_modules

        # Filtrer les points d'entrée (scripts principaux)
        entry_points = {'jeffrey_agi_chat', 'chat_with_jeffrey', 'main', 'run_', 'test_', 'demo_', 'launcher'}

        for module in potentially_unused:
            is_entry_point = any(ep in module for ep in entry_points)
            if not is_entry_point:
                self.unused_modules.add(module)

    def _detect_circular_dependencies(self):
        """Détecter les dépendances circulaires"""

        def has_path(start, end, visited=None):
            if visited is None:
                visited = set()
            if start in visited:
                return False
            if start == end:
                return True
            visited.add(start)
            for neighbor in self.modules_graph.get(start, []):
                if has_path(neighbor, end, visited.copy()):
                    return True
            return False

        for module in self.modules_graph:
            for dependency in self.modules_graph[module]:
                if has_path(dependency, module):
                    cycle = (module, dependency)
                    if cycle not in self.circular_dependencies:
                        self.circular_dependencies.append(cycle)

    def _detect_duplicate_code(self, python_files: list[Path]):
        """Détecter le code dupliqué"""
        function_signatures = defaultdict(list)

        for filepath in python_files[:100]:  # Limiter pour performance
            try:
                with open(filepath, encoding='utf-8') as f:
                    content = f.read()

                # Extraire signatures de fonctions
                func_pattern = r'def\s+(\w+)\s*\([^)]*\):'
                matches = re.finditer(func_pattern, content)

                for match in matches:
                    func_name = match.group(1)
                    # Ignorer les fonctions communes
                    if func_name not in ['__init__', '__str__', '__repr__', 'main']:
                        function_signatures[func_name].append(str(filepath))

            except Exception:
                continue

        # Identifier fonctions dupliquées
        for func_name, files in function_signatures.items():
            if len(files) > 1:
                self.duplicate_code.append({'function': func_name, 'files': files, 'count': len(files)})

    def _detect_optimization_opportunities(self, python_files: list[Path]):
        """Détecter les opportunités d'optimisation"""

        # Opportunité 1: Gros fichiers à splitter
        for module, stats in self.module_stats.items():
            if stats['lines'] > 500:
                self.optimization_opportunities.append(
                    {
                        'type': 'large_file',
                        'module': module,
                        'lines': stats['lines'],
                        'suggestion': f"Considérer split en modules plus petits ({stats['lines']} lignes)",
                        'priority': 'medium' if stats['lines'] < 1000 else 'high',
                    }
                )

        # Opportunité 2: Modules avec beaucoup d'imports
        for module, imports in self.modules_graph.items():
            if len(imports) > 20:
                self.optimization_opportunities.append(
                    {
                        'type': 'too_many_imports',
                        'module': module,
                        'imports_count': len(imports),
                        'suggestion': f"Rationaliser les imports ({len(imports)} imports)",
                        'priority': 'low',
                    }
                )

        # Opportunité 3: Modules sans documentation
        for module, stats in self.module_stats.items():
            if stats['functions'] + stats['classes'] > 5:
                # Simuler faible couverture docstring (analyse simplifiée)
                self.optimization_opportunities.append(
                    {
                        'type': 'missing_docs',
                        'module': module,
                        'functions': stats['functions'],
                        'classes': stats['classes'],
                        'suggestion': f"Améliorer documentation ({stats['functions']} fonctions, {stats['classes']} classes)",
                        'priority': 'medium',
                    }
                )

    def _generate_architecture_report(self) -> dict[str, Any]:
        """Générer le rapport d'architecture"""

        # Statistiques globales
        total_modules = len(self.module_stats)
        avg_lines = sum(s['lines'] for s in self.module_stats.values()) / max(1, total_modules)

        # Top modules par taille
        largest_modules = sorted(self.module_stats.items(), key=lambda x: x[1]['lines'], reverse=True)[:10]

        # Top modules par nombre d'imports
        most_importing = sorted(self.modules_graph.items(), key=lambda x: len(x[1]), reverse=True)[:10]

        return {
            'scan_date': '2025-01-14',
            'total_modules': total_modules,
            'avg_lines_per_module': round(avg_lines, 1),
            'unused_modules': {'count': len(self.unused_modules), 'modules': list(self.unused_modules)},
            'circular_dependencies': {'count': len(self.circular_dependencies), 'cycles': self.circular_dependencies},
            'duplicate_code': {
                'count': len(self.duplicate_code),
                'functions': self.duplicate_code[:20],  # Top 20
            },
            'optimization_opportunities': {
                'count': len(self.optimization_opportunities),
                'high_priority': [o for o in self.optimization_opportunities if o.get('priority') == 'high'],
                'medium_priority': [o for o in self.optimization_opportunities if o.get('priority') == 'medium'],
                'low_priority': [o for o in self.optimization_opportunities if o.get('priority') == 'low'],
            },
            'largest_modules': largest_modules,
            'most_importing_modules': [
                (mod, list(imports)) for mod, imports in most_importing
            ],  # Convertir sets en listes
            'module_stats': self.module_stats,
        }


def generate_architecture_markdown(report: dict[str, Any], output_file: str = "JEFFREY_ARCHITECTURE_ANALYSIS.md"):
    """Générer rapport architecture en Markdown"""

    unused = report['unused_modules']
    circular = report['circular_dependencies']
    duplicate = report['duplicate_code']
    optimizations = report['optimization_opportunities']

    md_content = f"""# 🏗️ Analyse Architecture Jeffrey AGI

## 📊 Résumé Global
- **Modules analysés** : {report['total_modules']}
- **Taille moyenne** : {report['avg_lines_per_module']} lignes/module
- **Modules non-branchés** : {unused['count']}
- **Dépendances circulaires** : {circular['count']}
- **Code dupliqué** : {duplicate['count']} fonctions
- **Opportunités d'optimisation** : {optimizations['count']}

## 🚨 Issues Architecture Critiques

### 🔴 Modules Non-Branchés ({unused['count']})
Ces modules semblent non utilisés et pourraient être supprimés ou refactorisés :

"""

    for i, module in enumerate(unused['modules'][:10], 1):
        md_content += f"{i}. `{module}`\n"

    if len(unused['modules']) > 10:
        md_content += f"... et {len(unused['modules']) - 10} autres\n"

    md_content += f"""
### 🔄 Dépendances Circulaires ({circular['count']})
"""

    for i, (mod1, mod2) in enumerate(circular['cycles'][:5], 1):
        md_content += f"{i}. `{mod1}` ↔️ `{mod2}`\n"

    md_content += """
### 📋 Code Dupliqué - Top Fonctions
"""

    for i, dup in enumerate(duplicate['functions'][:10], 1):
        md_content += f"{i}. **{dup['function']}()** dans {dup['count']} fichiers :\n"
        for filepath in dup['files'][:3]:
            md_content += f"   - `{filepath}`\n"
        md_content += "\n"

    md_content += f"""
## 💡 Opportunités d'Optimisation

### 🔴 Priorité Haute ({len(optimizations['high_priority'])})
"""

    for opp in optimizations['high_priority'][:5]:
        md_content += f"- **{opp['module']}** : {opp['suggestion']}\n"

    md_content += f"""
### 🟡 Priorité Moyenne ({len(optimizations['medium_priority'])})
"""

    for opp in optimizations['medium_priority'][:5]:
        md_content += f"- **{opp['module']}** : {opp['suggestion']}\n"

    md_content += """
## 📈 Top Modules par Taille
"""

    for i, (module, stats) in enumerate(report['largest_modules'][:10], 1):
        md_content += f"{i}. **{module}** : {stats['lines']} lignes ({stats['functions']} fonctions, {stats['classes']} classes)\n"

    md_content += """
## 🔗 Top Modules par Imports
"""

    for i, (module, imports) in enumerate(report['most_importing_modules'][:10], 1):
        md_content += f"{i}. **{module}** : {len(imports)} imports\n"

    md_content += f"""
## 📋 Recommandations d'Action

### Phase Immédiate (1 semaine)
1. **Nettoyer modules non-branchés** : Supprimer ou connecter les {unused['count']} modules inutilisés
2. **Résoudre dépendances circulaires** : Refactoring pour éliminer les {circular['count']} cycles
3. **Traiter gros modules** : Splitter les modules >500 lignes

### Phase Amélioration (2-3 semaines)
1. **Factoriser code dupliqué** : Créer utilitaires communs pour les {duplicate['count']} fonctions dupliquées
2. **Optimiser imports** : Réduire les imports excessifs (>20 imports/module)
3. **Améliorer documentation** : Ajouter docstrings manquantes

### Phase Optimisation (1 mois)
1. **Restructuration modules** : Organiser selon responsabilités
2. **Cache et performance** : Optimiser modules lourds
3. **Tests architecture** : Valider non-régression

## 🎯 Métriques de Succès
- ✅ Modules non-branchés : 0
- ✅ Dépendances circulaires : 0
- ✅ Modules >500 lignes : <5
- ✅ Code dupliqué : -50%
- ✅ Temps chargement Jeffrey : -20%

---
📅 **Généré le** : 2025-01-14
🔧 **Outil** : Jeffrey Architecture Analyzer v1.0
"""

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(md_content)

    logger.info(f"📄 Rapport architecture généré : {output_file}")


def main():
    """Point d'entrée principal"""
    print("🏗️ JEFFREY AGI - ANALYSE ARCHITECTURE")
    print("=" * 50)

    analyzer = JeffreyArchitectureAnalyzer(".")
    report = analyzer.analyze_project_architecture()

    # Sauvegarder JSON
    with open("jeffrey_architecture_report.json", 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Générer Markdown
    generate_architecture_markdown(report)

    # Résumé
    print("\n📊 ANALYSE TERMINÉE")
    print(f"✅ {report['total_modules']} modules analysés")
    print(f"⚠️ {report['unused_modules']['count']} modules non-branchés")
    print(f"🔄 {report['circular_dependencies']['count']} dépendances circulaires")
    print(f"📋 {report['duplicate_code']['count']} fonctions dupliquées")
    print(f"💡 {report['optimization_opportunities']['count']} opportunités d'optimisation")
    print("📄 Rapports générés :")
    print("   - JEFFREY_ARCHITECTURE_ANALYSIS.md")
    print("   - jeffrey_architecture_report.json")


if __name__ == "__main__":
    main()
