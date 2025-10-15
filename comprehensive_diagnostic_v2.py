#!/usr/bin/env python3
"""
Diagnostic V2 PATCHÉ : Corrections des bugs count_references + runtime_check
Bugs corrigés :
1. count_references() : Match préfixe pour symbols (jeffrey.core.x.y.ClassName)
2. runtime_import_check_isolated() : Multiples racines dans sys.path
"""

import ast
import importlib.util
import json
import subprocess
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path


class ImportScanner(ast.NodeVisitor):
    """Visite l'AST pour extraire imports + détecter stubs."""

    def __init__(self, filepath: Path, module_name: str):
        self.filepath = filepath
        self.module_name = module_name
        self.imports = set()
        self.local_imports = set()

        # Détection de stubs
        self.is_stub = False
        self.stub_indicators = []

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.name)
            if alias.name.startswith(('jeffrey', 'core', 'unified', 'Orchestrateur_IA')):
                self.local_imports.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            # Gérer les imports relatifs (GPT)
            if node.level > 0:  # from .utils import x
                resolved = self._resolve_relative_import(node.module, node.level)
                if resolved:
                    for alias in node.names:
                        full_path = f"{resolved}.{alias.name}"
                        self.imports.add(full_path)
                        if resolved.startswith(('jeffrey', 'core', 'unified')):
                            self.local_imports.add(resolved)
            else:
                for alias in node.names:
                    full_path = f"{node.module}.{alias.name}"
                    self.imports.add(full_path)
                    if node.module.startswith(('jeffrey', 'core', 'unified', 'Orchestrateur_IA')):
                        self.local_imports.add(node.module)
        self.generic_visit(node)

    def _resolve_relative_import(self, module: str, level: int) -> str:
        """Résout un import relatif en import absolu (GPT)."""
        parts = self.module_name.split('.')
        if level > len(parts):
            return module or ''

        base_parts = parts[: len(parts) - level + 1]
        if module:
            return '.'.join(base_parts + [module])
        return '.'.join(base_parts)

    def visit_ClassDef(self, node):
        """Détecte les classes stubs."""
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            self.is_stub = True
            self.stub_indicators.append({'type': 'empty_class', 'name': node.name, 'line': node.lineno})

        if len(node.body) == 1:
            init = node.body[0]
            if isinstance(init, ast.FunctionDef) and init.name == '__init__':
                if len(init.body) == 1 and isinstance(init.body[0], ast.Pass):
                    self.is_stub = True
                    self.stub_indicators.append({'type': 'empty_init', 'name': node.name, 'line': node.lineno})

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Détecte les fonctions stubs."""
        if len(node.body) == 1:
            stmt = node.body[0]

            if isinstance(stmt, ast.Return):
                if stmt.value is None:
                    self.is_stub = True
                    self.stub_indicators.append({'type': 'return_none', 'name': node.name, 'line': node.lineno})
                elif isinstance(stmt.value, ast.Dict) and not stmt.value.keys:
                    self.is_stub = True
                    self.stub_indicators.append({'type': 'return_empty_dict', 'name': node.name, 'line': node.lineno})

        for stmt in node.body:
            if isinstance(stmt, ast.Raise):
                if isinstance(stmt.exc, ast.Call):
                    if isinstance(stmt.exc.func, ast.Name):
                        if stmt.exc.func.id == 'NotImplementedError':
                            self.is_stub = True
                            self.stub_indicators.append(
                                {'type': 'not_implemented', 'name': node.name, 'line': node.lineno}
                            )

        self.generic_visit(node)


def load_ignore_patterns() -> set[str]:
    """Charge les patterns depuis .restorationignore (GPT)."""
    ignore_file = Path('.restorationignore')
    if not ignore_file.exists():
        return set()

    patterns = set()
    with open(ignore_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                patterns.add(line)
    return patterns


def should_ignore(filepath: Path, ignore_patterns: set[str]) -> bool:
    """Vérifie si un fichier doit être ignoré."""
    path_str = str(filepath)
    for pattern in ignore_patterns:
        if pattern.rstrip('/') in path_str:
            return True
    return False


def get_file_age_days(filepath: Path) -> int:
    """Retourne l'âge du fichier en jours via git log (Grok)."""
    try:
        result = subprocess.run(
            ['git', 'log', '-1', '--format=%at', str(filepath)], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            timestamp = int(result.stdout.strip())
            age_seconds = datetime.now().timestamp() - timestamp
            return int(age_seconds / 86400)
    except Exception:
        pass
    return 0


def scan_file_complete(
    filepath: Path, ignore_patterns: set[str]
) -> tuple[set[str], set[str], bool, list[dict], list[str], int]:
    """Scan complet d'un fichier : imports + stubs + âge"""
    if should_ignore(filepath, ignore_patterns):
        return set(), set(), False, [], [], 0

    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()

        module_name = str(filepath).replace('/', '.').replace('.py', '')
        if module_name.startswith('src.'):
            module_name = module_name[4:]

        tree = ast.parse(content, filename=str(filepath))
        scanner = ImportScanner(filepath, module_name)
        scanner.visit(tree)

        text_indicators = []
        if 'STUB' in content.upper() and 'STUB_TO' not in content:
            text_indicators.append("Marqueur obsolète")
        if 'TODO' in content and 'implement' in content.lower():
            text_indicators.append("'TODO implement'")
        if 'NotImplementedError' in content and 'test' not in str(filepath).lower():
            text_indicators.append("NotImplementedError hors tests")

        age_days = get_file_age_days(filepath)

        return (
            scanner.imports,
            scanner.local_imports,
            scanner.is_stub or bool(text_indicators),
            scanner.stub_indicators,
            text_indicators,
            age_days,
        )

    except Exception as e:
        return set(), set(), False, [], [f"Erreur : {e}"], 0


def runtime_import_check_isolated(module_name: str) -> dict:
    """
    🔧 PATCHÉ : Vérification runtime avec MULTIPLES racines (GPT fix)
    """
    root = Path.cwd()

    script = f"""
import sys
from pathlib import Path

root = Path('{root}')

# 🔧 FIX : Ajouter TOUTES les racines possibles
for p in [
    str(root),
    str(root / 'src'),
    str(root / 'services'),
    str(root / 'core'),
    str(root / 'unified'),
]:
    if p not in sys.path:
        sys.path.insert(0, p)

import importlib.util
import importlib

try:
    spec = importlib.util.find_spec('{module_name}')
    if spec is None:
        print("NOT_FOUND")
    else:
        module = importlib.import_module('{module_name}')
        print(f"OK|{{spec.origin}}")
except Exception as e:
    print(f"ERROR|{{str(e)}}")
"""

    try:
        result = subprocess.run(
            [sys.executable, '-c', script], capture_output=True, text=True, timeout=10, cwd=Path.cwd()
        )

        output = result.stdout.strip()

        if output == "NOT_FOUND":
            return {'importable': False, 'error': 'Module not found', 'spec': None}
        elif output.startswith("OK|"):
            return {'importable': True, 'error': None, 'spec': output.split('|', 1)[1]}
        elif output.startswith("ERROR|"):
            return {'importable': False, 'error': output.split('|', 1)[1], 'spec': None}
        else:
            return {'importable': False, 'error': 'Unknown error', 'spec': None}

    except subprocess.TimeoutExpired:
        return {'importable': False, 'error': 'Import timeout', 'spec': None}
    except Exception as e:
        return {'importable': False, 'error': str(e), 'spec': None}


def check_external_deps() -> dict[str, bool]:
    """Vérifie les dépendances externes installées (Grok)."""
    common_deps = ['torch', 'numpy', 'pandas', 'sklearn', 'transformers', 'requests']
    results = {}

    for dep in common_deps:
        try:
            spec = importlib.util.find_spec(dep)
            results[dep] = spec is not None
        except Exception:
            results[dep] = False

    return results


def find_source_candidates(import_name: str, search_dirs: list[Path]) -> list[str]:
    """Cherche les sources candidates pour un import."""
    candidates = []
    parts = import_name.split('.')

    possible_files = [
        f"{'/'.join(parts)}.py",
        f"{'/'.join(parts)}/__init__.py",
        f"src/{'/'.join(parts)}.py",
        f"services/{'/'.join(parts)}.py",
        f"unified/{'/'.join(parts)}.py",
    ]

    for search_dir in search_dirs:
        for possible_file in possible_files:
            file_path = search_dir / possible_file
            if file_path.exists():
                try:
                    candidates.append(str(file_path.relative_to(Path.cwd())))
                except ValueError:
                    candidates.append(str(file_path))

    return list(set(candidates))


def build_dependency_graph(files_imports: dict[Path, set[str]]) -> dict[str, set[str]]:
    """Construit le graphe de dépendances."""
    graph = defaultdict(set)

    for filepath, imports in files_imports.items():
        module_name = str(filepath).replace('/', '.').replace('.py', '')
        if module_name.startswith('src.'):
            module_name = module_name[4:]

        for imp in imports:
            if imp.startswith(('jeffrey', 'core', 'unified', 'Orchestrateur_IA')):
                graph[module_name].add(imp)

    return dict(graph)


def calculate_centrality(graph: dict[str, set[str]]) -> dict[str, int]:
    """
    Calcule la centralité (in-degree) de chaque module (GPT)
    = nombre de modules qui dépendent de lui
    """
    in_degree = defaultdict(int)

    for deps in graph.values():
        for dep in deps:
            in_degree[dep] += 1

    return dict(in_degree)


def topological_sort(graph: dict[str, set[str]]) -> tuple[list[str], list[str]]:
    """Tri topologique avec détection de cycles."""
    in_degree = defaultdict(int)
    all_nodes = set(graph.keys())

    for deps in graph.values():
        all_nodes.update(deps)

    for node in all_nodes:
        in_degree[node] = 0

    for deps in graph.values():
        for dep in deps:
            in_degree[dep] += 1

    queue = deque([node for node in all_nodes if in_degree[node] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        if node in graph:
            for dependent in graph[node]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

    cycles = [node for node in all_nodes if node not in result]
    return result, cycles


def count_references(import_name: str, all_imports: dict[Path, set[str]]) -> int:
    """
    🔧 PATCHÉ : Compte les références avec match préfixe (GPT fix)
    Match exact OU préfixe (ex: 'pkg.mod' compte aussi 'pkg.mod.ClassName')
    """
    count = 0
    prefix = import_name + "."

    for imports in all_imports.values():
        for imp in imports:
            # Match exact OU commence par le préfixe
            if imp == import_name or imp.startswith(prefix):
                count += 1

    return count


def calculate_obsolescence_score(stub_info: dict, ref_count: int, age_days: int) -> int:
    """Score d'obsolescence (Grok) : 0-100"""
    score = 0

    if ref_count == 0:
        score += 50
    elif ref_count == 1:
        score += 30
    elif ref_count == 2:
        score += 10

    if age_days > 365:
        score += 30
    elif age_days > 180:
        score += 15

    if stub_info.get('stub_indicators'):
        for ind in stub_info['stub_indicators']:
            if ind['type'] in ['empty_class', 'empty_init']:
                score += 10

    return min(score, 100)


def main():
    print("🔍 DIAGNOSTIC COMPLET V2 PATCHÉ (Bugs corrigés)")
    print("=" * 60)
    print()

    # Configuration avec homogénéisation des chemins (recommandation GPT)
    for path in ['.', 'src', 'core', 'services', 'unified']:
        path_str = str(Path.cwd() / path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    ignore_patterns = load_ignore_patterns()

    if ignore_patterns:
        print(f"📁 Chargé {len(ignore_patterns)} patterns d'exclusion")

    # 🔧 FIX : Dossiers principaux seulement (GPT recommandation)
    search_dirs = [
        Path('src'),
        Path('core'),
        Path('services'),
        Path('unified'),
    ]

    print()
    print("📁 Phase unique : Scan optimisé (imports + stubs + âge)...")

    all_imports = {}
    local_imports_by_file = {}
    stub_files = []
    scanned_files = 0

    for search_dir in search_dirs:
        if not search_dir.exists():
            continue

        for py_file in search_dir.rglob('*.py'):
            all_imp, local_imp, is_stub, stub_ind, text_ind, age = scan_file_complete(py_file, ignore_patterns)

            if not all_imp and not is_stub:
                continue

            if all_imp:
                all_imports[py_file] = all_imp
                local_imports_by_file[py_file] = local_imp

            if is_stub:
                stub_files.append(
                    {'file': str(py_file), 'ast_indicators': stub_ind, 'text_indicators': text_ind, 'age_days': age}
                )

            scanned_files += 1

            if scanned_files % 100 == 0:
                print(f"   Progression : {scanned_files} fichiers...")

    print(f"   ✅ {scanned_files} fichiers analysés")

    all_local_imports = set()
    for imports in local_imports_by_file.values():
        all_local_imports.update(imports)

    print(f"   📦 {len(all_local_imports)} imports locaux uniques")
    print(f"   ❌ {len(stub_files)} stubs détectés")
    print()

    print("📁 Vérification runtime isolée (subprocess avec multiples racines)...")
    runtime_results = {}
    imports_to_check = [
        imp
        for imp in all_local_imports
        if imp.split('.')[0]
        not in [
            'os',
            'sys',
            'json',
            'time',
            'pathlib',
            'typing',
            'logging',
            'asyncio',
            'numpy',
            'pandas',
            'torch',
            'sklearn',
        ]
    ]

    for i, imp in enumerate(imports_to_check, 1):
        if i % 10 == 0:
            print(f"   {i}/{len(imports_to_check)}")
        runtime_results[imp] = runtime_import_check_isolated(imp)

    truly_broken_runtime = {imp: res for imp, res in runtime_results.items() if not res['importable']}

    print(f"   ❌ {len(truly_broken_runtime)} imports cassés")
    print()

    print("📁 Vérification dépendances externes...")
    external_deps = check_external_deps()
    missing_deps = [dep for dep, installed in external_deps.items() if not installed]
    if missing_deps:
        print(f"   ⚠️  Dépendances manquantes : {', '.join(missing_deps)}")
    else:
        print("   ✅ Toutes les dépendances communes installées")
    print()

    print("📁 Recherche sources candidates...")
    missing_with_candidates = {}
    missing_without_candidates = {}

    for imp in truly_broken_runtime.keys():
        candidates = find_source_candidates(imp, search_dirs)
        ref_count = count_references(imp, all_imports)  # 🔧 Utilise la version patchée

        if candidates:
            missing_with_candidates[imp] = {
                'candidates': candidates,
                'references': ref_count,
                'error': truly_broken_runtime[imp]['error'],
            }
        else:
            missing_without_candidates[imp] = {'references': ref_count, 'error': truly_broken_runtime[imp]['error']}

    print(f"   ✅ {len(missing_with_candidates)} avec sources")
    print(f"   ❌ {len(missing_without_candidates)} vraiment manquants")
    print()

    print("📁 Construction graphe de dépendances...")
    dep_graph = build_dependency_graph(local_imports_by_file)
    centrality = calculate_centrality(dep_graph)
    topo_order, cycles = topological_sort(dep_graph)

    print(f"   🔗 {len(dep_graph)} modules dans le graphe")
    print(f"   📊 {len(topo_order)} nœuds triés")
    if cycles:
        print(f"   ⚠️  {len(cycles)} cycles détectés : {cycles[:5]}...")
    print()

    print("📁 Calcul scores d'obsolescence...")
    stub_scores = []
    for stub in stub_files:
        filepath = Path(stub['file'])
        module_name = str(filepath).replace('/', '.').replace('.py', '')
        ref_count = count_references(module_name, all_imports)  # 🔧 Utilise la version patchée
        score = calculate_obsolescence_score(stub, ref_count, stub['age_days'])

        stub_scores.append(
            {
                **stub,
                'references': ref_count,
                'obsolescence_score': score,
                'suggested_action': 'DELETE' if score > 70 else 'CLEAN' if score > 40 else 'KEEP',
            }
        )

    stub_scores.sort(key=lambda x: x['obsolescence_score'], reverse=True)

    print(f"   📊 {len([s for s in stub_scores if s['obsolescence_score'] > 70])} stubs probablement obsolètes")
    print()

    # Rapport complet
    report = {
        'timestamp': datetime.now().isoformat(),
        'stats': {
            'scanned_files': scanned_files,
            'total_local_imports': len(all_local_imports),
            'runtime_broken': len(truly_broken_runtime),
            'with_candidates': len(missing_with_candidates),
            'without_candidates': len(missing_without_candidates),
            'stubs_detected': len(stub_files),
            'dependency_graph_nodes': len(dep_graph),
            'cycles': len(cycles),
        },
        'external_deps': external_deps,
        'runtime_broken': truly_broken_runtime,
        'missing_with_candidates': missing_with_candidates,
        'missing_without_candidates': missing_without_candidates,
        'stubs_with_scores': stub_scores,
        'dependency_graph': {k: list(v) for k, v in dep_graph.items()},
        'centrality': centrality,
        'topological_order': topo_order,
        'cycles': cycles,
    }

    with open('COMPREHENSIVE_DIAGNOSTIC_V2.json', 'w') as f:
        json.dump(report, f, indent=2)

    print("=" * 60)
    print("📊 RÉSUMÉ")
    print("=" * 60)
    print(f"✅ Imports avec sources : {len(missing_with_candidates)}")
    print(f"❌ Imports manquants : {len(missing_without_candidates)}")
    print(f"🔗 Modules graphe : {len(dep_graph)}")
    print(f"❌ Stubs détectés : {len(stub_files)}")
    print()
    print("📄 Rapport : COMPREHENSIVE_DIAGNOSTIC_V2.json")
    print()

    if cycles:
        print("⚠️  CYCLES DÉTECTÉS dans le graphe :")
        for cycle in cycles[:5]:
            print(f"   • {cycle}")
    print()

    # TOP 10 critiques avec les VRAIES références
    if missing_without_candidates:
        print("🔴 TOP 10 IMPORTS CRITIQUES (avec références corrigées) :")
        sorted_missing = sorted(
            missing_without_candidates.items(),
            key=lambda x: (centrality.get(x[0], 0) * 2 + x[1]['references']),
            reverse=True,
        )[:10]

        for imp, info in sorted_missing:
            cent = centrality.get(imp, 0)
            refs = info['references']
            print(f"   • {imp}")
            print(f"     Centralité: {cent}, Références: {refs}")
    print()

    # Stubs obsolètes
    if stub_scores:
        print("🗑️  TOP 10 MODULES PROBABLEMENT OBSOLÈTES :")
        for stub in stub_scores[:10]:
            if stub['obsolescence_score'] > 70:
                print(f"   • {stub['file']} (score: {stub['obsolescence_score']})")
                print(f"     Action: {stub['suggested_action']}, Refs: {stub['references']}, Âge: {stub['age_days']}j")


if __name__ == "__main__":
    main()
