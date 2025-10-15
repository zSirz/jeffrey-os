#!/usr/bin/env python3
# Fichier: diagnostics/analyze_async.py

import ast
import json
import re
from pathlib import Path

# --- injected helper: safe relative path ---
from pathlib import Path as _Path

__CWD = _Path.cwd().resolve()


def _safe_rel(p) -> str:
    try:
        p = p if isinstance(p, _Path) else _Path(p)
        p = p.resolve()
        return str(p.relative_to(__CWD))
    except Exception:
        return str(p)


# --- end helper ---


def check_missing_await(tree, file_path, issues):
    """
    Détecte les appels async non-await dans les fonctions async
    """
    try:
        src = Path(file_path).read_text(encoding="utf-8").splitlines()
    except:
        return

    class AsyncVisitor(ast.NodeVisitor):
        def __init__(self):
            self.in_async = False
            self.async_stack = []

        def visit_AsyncFunctionDef(self, node):
            self.async_stack.append(node.name)
            old_in_async = self.in_async
            self.in_async = True
            self.generic_visit(node)
            self.in_async = old_in_async
            self.async_stack.pop()

        def visit_Call(self, node):
            if not self.in_async:
                self.generic_visit(node)
                return

            # Extraire le nom de la fonction appelée
            func_name = self._get_func_name(node.func)

            # Vérifier si c'est un appel qui devrait être await
            if self._is_async_call(func_name):
                # Vérifier si await est présent
                line_num = getattr(node, 'lineno', None)
                if line_num and 0 <= line_num - 1 < len(src):
                    line = src[line_num - 1]
                    if 'await' not in line:
                        issues['missing_await'].append(
                            {
                                'file': _safe_rel(file_path),
                                'line': line_num,
                                'call': func_name,
                                'function': self.async_stack[-1] if self.async_stack else 'unknown',
                            }
                        )

            self.generic_visit(node)

        def _get_func_name(self, node):
            """Extrait le nom complet d'une fonction"""
            if isinstance(node, ast.Attribute):
                value = self._get_func_name(node.value)
                return f"{value}.{node.attr}" if value else node.attr
            elif isinstance(node, ast.Name):
                return node.id
            return ""

        def _is_async_call(self, name):
            """Détermine si un appel devrait être await"""
            async_patterns = [
                'asyncio.sleep',
                'AsyncClient',
                'aiohttp',
                'aiofiles',
                '.gather',
                '.create_task',
                '.wait',
                'async_',  # Convention de nommage
            ]
            return any(pattern in name for pattern in async_patterns)

    visitor = AsyncVisitor()
    visitor.visit(tree)


def diagnose_async():
    """Analyse complète des problèmes async"""
    root = Path("src/jeffrey")
    issues = {
        'event_loop_problems': [],
        'blocking_calls': [],
        'missing_await': [],
        'sync_in_async': [],
        'entry_points_without_run': [],
    }

    print("⚡ Analyse des problèmes async...")

    for py_file in root.rglob("*.py"):
        if '__pycache__' in str(py_file):
            continue

        try:
            content = py_file.read_text(encoding='utf-8')

            # Problème 1: get_event_loop() sans asyncio.run()
            if 'get_event_loop()' in content and 'asyncio.run(' not in content:
                issues['event_loop_problems'].append(
                    {'file': _safe_rel(py_file), 'issue': 'get_event_loop() without asyncio.run()'}
                )

            # Problème 2: Appels bloquants dans async
            if 'async def ' in content:
                # Analyse ligne par ligne pour localisation précise
                for i, line in enumerate(content.splitlines(), 1):
                    blocking_patterns = [
                        (r'\btime\.sleep\(', 'time.sleep instead of asyncio.sleep', i),
                        (r'\brequests\.(get|post|put|delete)\(', 'requests instead of httpx/aiohttp', i),
                        (r'\bopen\([^)]*\)(?!.*async)', 'sync open instead of aiofiles', i),
                    ]

                    for pattern, description, line_num in blocking_patterns:
                        if re.search(pattern, line):
                            issues['blocking_calls'].append(
                                {'file': _safe_rel(py_file), 'issue': description, 'line': line_num}
                            )

            # Problème 3: Points d'entrée sans asyncio.run()
            if '__main__' in content and 'async def main' in content:
                if 'asyncio.run(main())' not in content:
                    issues['entry_points_without_run'].append(
                        {'file': _safe_rel(py_file), 'issue': 'async main without asyncio.run()'}
                    )

            # Problème 4: Analyse AST pour await manquants
            try:
                tree = ast.parse(content)
                check_missing_await(tree, py_file, issues)
            except SyntaxError:
                pass  # Sera traité dans compilation errors

        except Exception as e:
            print(f"  ⚠️ Erreur analyse {py_file.name}: {e}")

    # Rapport avec priorités
    report = {
        'total_issues': sum(len(v) for v in issues.values()),
        'by_category': {k: len(v) for k, v in issues.items()},
        'high_priority': [],  # Missing await et event_loop sont critiques
        'medium_priority': [],  # Blocking calls
        'low_priority': [],  # Entry points
        'details': issues,
    }

    # Trier par priorité
    report['high_priority'] = issues['missing_await'] + issues['event_loop_problems']
    report['medium_priority'] = issues['blocking_calls']
    report['low_priority'] = issues['entry_points_without_run']

    # Sauvegarder
    report_path = Path("diagnostics/async_report.json")
    report_path.write_text(json.dumps(report, indent=2))

    # Afficher résumé
    print("\n📊 RÉSUMÉ:")
    print(f"  Total: {report['total_issues']} problèmes")
    print(f"  - Haute priorité (await manquants): {len(report['high_priority'])}")
    print(f"  - Moyenne priorité (blocking): {len(report['medium_priority'])}")
    print(f"  - Basse priorité (entry points): {len(report['low_priority'])}")

    return report


if __name__ == "__main__":
    diagnose_async()
