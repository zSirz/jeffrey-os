#!/usr/bin/env python3
"""
AST Analyzer Module - Jeffrey Phoenix Super Analyzer
Analyse sémantique avancée avec AST Python
Version: 1.0.0 (Fixed)
"""

import ast
import hashlib
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO)


class ASTAnalyzer:
    """Analyseur AST avancé avec détection de patterns et métriques"""

    def __init__(self):
        self.patterns_found = defaultdict(int)
        self.security_issues = []
        self.code_smells = []
        self.secret_patterns = {
            'api_key': [
                r'["\']?api[_\-]?key["\']?\s*[:=]\s*["\'][a-zA-Z0-9]{20,}["\']',
                r'["\']?apikey["\']?\s*[:=]\s*["\'][a-zA-Z0-9]{20,}["\']',
            ],
            'private_key': [
                r'-----BEGIN (RSA |EC |DSA |OPENSSH |)PRIVATE KEY-----',
                r'["\']?private[_\-]?key["\']?\s*[:=]\s*["\'][^"\']{40,}["\']',
            ],
            'aws_key': [
                r'AKIA[0-9A-Z]{16}',
                r'aws[_\-]?access[_\-]?key[_\-]?id["\']?\s*[:=]\s*["\'][A-Z0-9]{20}["\']',
                r'aws[_\-]?secret[_\-]?access[_\-]?key["\']?\s*[:=]\s*["\'][a-zA-Z0-9/+=]{40}["\']',
            ],
            'token': [
                r'["\']?token["\']?\s*[:=]\s*["\'][a-zA-Z0-9._\-]{20,}["\']',
                r'["\']?auth[_\-]?token["\']?\s*[:=]\s*["\'][a-zA-Z0-9._\-]{20,}["\']',
            ],
            'password': [
                r'["\']?password["\']?\s*[:=]\s*["\'][^"\']{8,}["\']',
                r'["\']?passwd["\']?\s*[:=]\s*["\'][^"\']{8,}["\']',
                r'["\']?pwd["\']?\s*[:=]\s*["\'][^"\']{8,}["\']',
            ],
        }

    def analyze_file(self, file_path: Path) -> dict | None:
        """Analyse un fichier Python avec AST"""
        try:
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                source = f.read()

            # Parser l'AST
            try:
                tree = ast.parse(source)
            except SyntaxError as e:
                logging.debug(f"Syntax error in {file_path}: {e}")
                return None

            # Collecter les métriques
            metrics = self._collect_metrics(tree, file_path)

            # Détecter les patterns
            patterns = self._detect_patterns(tree)

            # Analyser les imports
            imports = self._analyze_imports(tree, file_path)

            # Scanner les secrets avec contexte AST
            secrets = self.scan_for_secrets_with_context(file_path, tree)

            return {
                'file': str(file_path),
                'metrics': metrics,
                'patterns': patterns,
                'imports': imports,
                'secrets': secrets,
                'code_smells': self._detect_code_smells(tree, metrics),
            }

        except Exception as e:
            logging.debug(f"Failed to analyze {file_path}: {e}")
            return None

    def calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """
        Calcule la vraie complexité cyclomatique McCabe
        Compte les nœuds de décision, pas les appels
        """
        # Types de nœuds qui augmentent la complexité
        decision_nodes = (
            ast.If,  # if statement
            ast.For,  # for loop
            ast.While,  # while loop
            ast.Try,  # try block
            ast.With,  # with statement
            ast.ExceptHandler,  # except clause
            ast.comprehension,  # list/dict/set comprehensions
            ast.Lambda,  # lambda expressions
            ast.IfExp,  # ternary operators
        )

        complexity = 1  # Base complexity

        # Parcourir l'AST et compter les nœuds de décision
        for child_node in ast.walk(node):
            if isinstance(child_node, decision_nodes):
                complexity += 1
            # Cas spécial pour BoolOp (and/or) : +1 par opérateur
            elif isinstance(child_node, ast.BoolOp):
                complexity += len(child_node.values) - 1

        return complexity

    def _hash_body(self, node: ast.AST) -> str:
        """
        Hash du corps d'une fonction compatible Python 3.8+
        """
        try:
            # Enlever la docstring si présente
            body = node.body[:] if hasattr(node, 'body') else []
            if body and isinstance(body[0], ast.Expr):
                # Check for docstring (Constant in 3.8+, Str in older)
                value = body[0].value
                if isinstance(value, ast.Constant):
                    if isinstance(value.value, str):
                        body = body[1:]  # Skip docstring
                elif hasattr(ast, 'Str') and isinstance(value, ast.Str):
                    body = body[1:]  # Skip old-style docstring

            # Créer un module temporaire pour pouvoir unparse/dump
            if body:
                temp_module = ast.Module(body=body, type_ignores=[])

                # Try unparse first (3.9+), fallback to dump
                if sys.version_info >= (3, 9) and hasattr(ast, 'unparse'):
                    source = ast.unparse(temp_module)
                else:
                    # Fallback pour Python < 3.9
                    source = ast.dump(temp_module, include_attributes=False)

                # Hash avec BLAKE2b (plus rapide que MD5/SHA)
                return hashlib.blake2b(source.encode(), digest_size=8).hexdigest()
            else:
                return "empty"

        except Exception:
            # En cas d'erreur, fallback sur un hash du nom
            node_name = getattr(node, 'name', 'unknown')
            return hashlib.blake2b(f"{node_name}_error".encode(), digest_size=8).hexdigest()

    def _collect_metrics(self, tree: ast.AST, file_path: Path) -> dict:
        """Collecte les métriques du code"""
        metrics = {
            'loc': 0,  # Lines of code
            'sloc': 0,  # Source lines of code (sans commentaires/blancs)
            'functions': 0,
            'classes': 0,
            'methods': 0,
            'avg_complexity': 0,
            'max_complexity': 0,
            'function_signatures': [],
            'class_hierarchy': [],
        }

        # Compter les lignes
        try:
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                metrics['loc'] = len(lines)
                metrics['sloc'] = sum(1 for line in lines if line.strip() and not line.strip().startswith('#'))
        except:
            pass

        complexities = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                metrics['functions'] += 1

                # Calculer la complexité
                complexity = self.calculate_cyclomatic_complexity(node)
                complexities.append(complexity)

                # Signature de fonction avec hash du corps
                body_hash = self._hash_body(node)
                metrics['function_signatures'].append(
                    {
                        'name': node.name,
                        'args': len(node.args.args),
                        'complexity': complexity,
                        'body_hash': body_hash,
                        'decorators': [d.id if isinstance(d, ast.Name) else ast.dump(d) for d in node.decorator_list],
                    }
                )

            elif isinstance(node, ast.ClassDef):
                metrics['classes'] += 1

                # Hiérarchie de classes
                bases = []
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        bases.append(base.id)
                    elif isinstance(base, ast.Attribute):
                        bases.append(ast.unparse(base) if hasattr(ast, 'unparse') else str(base))

                metrics['class_hierarchy'].append(
                    {
                        'name': node.name,
                        'bases': bases,
                        'methods': sum(1 for n in node.body if isinstance(n, ast.FunctionDef)),
                    }
                )

                # Compter les méthodes
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        metrics['methods'] += 1
                        complexity = self.calculate_cyclomatic_complexity(item)
                        complexities.append(complexity)

        # Calculer les moyennes
        if complexities:
            metrics['avg_complexity'] = sum(complexities) / len(complexities)
            metrics['max_complexity'] = max(complexities)

        return metrics

    def _detect_patterns(self, tree: ast.AST) -> dict[str, int]:
        """Détecte les patterns de design et anti-patterns"""
        patterns = defaultdict(int)

        for node in ast.walk(tree):
            # Singleton pattern
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == '__new__':
                        patterns['singleton'] += 1
                    elif isinstance(item, ast.FunctionDef) and item.name == '__init__':
                        # Check for factory pattern
                        for subnode in ast.walk(item):
                            if isinstance(subnode, ast.Return) and isinstance(subnode.value, ast.Call):
                                patterns['factory'] += 1
                                break

            # Decorator pattern
            elif isinstance(node, ast.FunctionDef) and node.decorator_list:
                patterns['decorator'] += 1

            # Context manager pattern
            elif isinstance(node, ast.With):
                patterns['context_manager'] += 1

            # Generator pattern
            elif isinstance(node, ast.Yield):
                patterns['generator'] += 1

            # Async patterns
            elif isinstance(node, (ast.AsyncFunctionDef, ast.AsyncWith, ast.AsyncFor)):
                patterns['async'] += 1

            # List comprehension
            elif isinstance(node, ast.ListComp):
                patterns['list_comprehension'] += 1

            # Lambda usage
            elif isinstance(node, ast.Lambda):
                patterns['lambda'] += 1

        return dict(patterns)

    def _build_import_graph_smart(self, tree: ast.AST, file_path: Path) -> dict:
        """
        Construit un graphe d'imports intelligent basé sur le mapping réel
        """
        # Trouver le package root (où il y a __init__.py ou setup.py)
        package_root = file_path.parent
        while package_root.parent != package_root:
            if (package_root / '__init__.py').exists() or (package_root / 'setup.py').exists():
                break
            package_root = package_root.parent

        # Construire le mapping module_name -> file_path
        module_map = {}
        try:
            for py_file in package_root.rglob('*.py'):
                # Calculer le nom du module relatif au package root
                try:
                    relative = py_file.relative_to(package_root)
                    module_name = str(relative).replace('/', '.').replace('\\', '.')[:-3]  # Remove .py
                    if module_name.endswith('.__init__'):
                        module_name = module_name[:-9]  # Remove .__init__
                    module_map[module_name] = str(py_file)
                except:
                    continue
        except:
            pass

        # Extraire les imports réels
        imports_resolved = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name in module_map:
                        imports_resolved.append(module_map[alias.name])

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                # Résoudre les imports relatifs
                if module.startswith('.'):
                    # Import relatif, résoudre par rapport au fichier actuel
                    try:
                        current_module = str(file_path.relative_to(package_root))[:-3].replace('/', '.')
                        parent_parts = current_module.split('.')

                        # Compter les points pour remonter
                        level = len(module) - len(module.lstrip('.'))
                        if level <= len(parent_parts):
                            parent_module = '.'.join(parent_parts[:-level]) if level > 0 else current_module
                            resolved_module = (
                                parent_module + '.' + module.lstrip('.') if module.lstrip('.') else parent_module
                            )
                            if resolved_module in module_map:
                                imports_resolved.append(module_map[resolved_module])
                    except:
                        pass
                else:
                    # Import absolu
                    if module in module_map:
                        imports_resolved.append(module_map[module])

        return {
            'file': str(file_path),
            'imports': imports_resolved,
            'module_map': module_map,  # Pour debug/visualisation
        }

    def _analyze_imports(self, tree: ast.AST, file_path: Path) -> dict:
        """Analyse les imports du fichier"""
        # Utiliser la version smart du graph builder
        import_graph = self._build_import_graph_smart(tree, file_path)

        # Analyser les imports standards vs tiers vs locaux
        imports = {'stdlib': [], 'third_party': [], 'local': [], 'graph': import_graph['imports']}

        stdlib_modules = {
            'os',
            'sys',
            'json',
            'pathlib',
            'logging',
            'ast',
            'hashlib',
            're',
            'time',
            'datetime',
            'collections',
            'itertools',
            'functools',
            'typing',
            'enum',
            'dataclasses',
            'asyncio',
            'threading',
            'multiprocessing',
            'sqlite3',
            'urllib',
            'http',
            'email',
            'csv',
            'xml',
            'html',
        }

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name.split('.')[0]
                    if module_name in stdlib_modules:
                        imports['stdlib'].append(alias.name)
                    elif alias.name in import_graph.get('module_map', {}):
                        imports['local'].append(alias.name)
                    else:
                        imports['third_party'].append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module_name = node.module.split('.')[0]
                    if module_name in stdlib_modules:
                        imports['stdlib'].append(node.module)
                    elif node.module in import_graph.get('module_map', {}):
                        imports['local'].append(node.module)
                    else:
                        imports['third_party'].append(node.module)

        return imports

    def scan_for_secrets_with_context(self, file_path: Path, ast_tree: ast.AST | None = None) -> list[dict]:
        """
        Scan secrets avec contexte AST pour réduire faux positifs
        """
        secrets_found = []

        try:
            with open(file_path, encoding='utf-8', errors='ignore') as f:
                content = f.read()

            # Si on a l'AST, extraire les zones de code réel (pas commentaires/docstrings)
            code_zones = []
            if ast_tree:
                for node in ast.walk(ast_tree):
                    # Ignorer les docstrings et commentaires
                    if isinstance(node, ast.Expr) and isinstance(node.value, (ast.Constant, ast.Str)):
                        continue

                    # Garder seulement les assignations et appels
                    if isinstance(node, (ast.Assign, ast.Call, ast.keyword)):
                        if hasattr(node, 'lineno'):
                            code_zones.append((node.lineno, getattr(node, 'end_lineno', node.lineno)))

            # Scanner ligne par ligne
            for line_num, line in enumerate(content.splitlines(), 1):
                # Skip si dans commentaire
                if line.strip().startswith('#'):
                    continue

                # Skip si pas dans zone de code (si AST disponible)
                if ast_tree and code_zones:
                    in_code = any(start <= line_num <= end for start, end in code_zones)
                    if not in_code:
                        continue

                # Check patterns
                for secret_type, patterns in self.secret_patterns.items():
                    for pattern in patterns:
                        if match := re.search(pattern, line, re.IGNORECASE):
                            # Filtres supplémentaires
                            matched_text = match.group()

                            # Skip si contient "example", "test", "dummy", "fake"
                            if any(
                                skip in matched_text.lower()
                                for skip in ['example', 'test', 'dummy', 'fake', 'xxx', 'sample', 'demo']
                            ):
                                continue

                            # Skip si dans un test file
                            if 'test' in str(file_path).lower():
                                continue

                            # Masquer intelligemment
                            if len(matched_text) > 20:
                                masked = matched_text[:6] + '...' + matched_text[-4:]
                            else:
                                masked = '***REDACTED***'

                            secrets_found.append(
                                {
                                    'file': str(file_path),
                                    'line': line_num,
                                    'type': secret_type,
                                    'confidence': 'high'
                                    if not any(w in line.lower() for w in ['sample', 'default', 'config'])
                                    else 'medium',
                                    'masked_value': masked,
                                    'severity': 'critical' if secret_type in ['private_key', 'aws_key'] else 'high',
                                }
                            )

        except Exception as e:
            logging.debug(f"Secret scan error on {file_path}: {e}")

        return secrets_found

    def _detect_code_smells(self, tree: ast.AST, metrics: dict) -> list[dict]:
        """Détecte les code smells courants"""
        smells = []

        # Complexité excessive
        if metrics['max_complexity'] > 10:
            smells.append(
                {
                    'type': 'high_complexity',
                    'severity': 'high' if metrics['max_complexity'] > 15 else 'medium',
                    'message': f"Max complexity {metrics['max_complexity']} exceeds threshold",
                }
            )

        # Fichier trop long
        if metrics['sloc'] > 500:
            smells.append(
                {'type': 'long_file', 'severity': 'medium', 'message': f"File has {metrics['sloc']} SLOC (>500)"}
            )

        # Trop de paramètres dans les fonctions
        for func in metrics['function_signatures']:
            if func['args'] > 5:
                smells.append(
                    {
                        'type': 'too_many_parameters',
                        'severity': 'medium',
                        'function': func['name'],
                        'message': f"Function {func['name']} has {func['args']} parameters (>5)",
                    }
                )

        # Classes avec trop de méthodes
        for cls in metrics['class_hierarchy']:
            if cls['methods'] > 20:
                smells.append(
                    {
                        'type': 'god_class',
                        'severity': 'high',
                        'class': cls['name'],
                        'message': f"Class {cls['name']} has {cls['methods']} methods (>20)",
                    }
                )

        # Détection de code dupliqué (via hash des corps de fonction)
        func_hashes = defaultdict(list)
        for func in metrics['function_signatures']:
            func_hashes[func['body_hash']].append(func['name'])

        for body_hash, func_names in func_hashes.items():
            if len(func_names) > 1 and body_hash != 'empty':
                smells.append(
                    {
                        'type': 'duplicate_code',
                        'severity': 'high',
                        'functions': func_names,
                        'message': f"Functions {', '.join(func_names)} have identical bodies",
                    }
                )

        return smells


def main():
    """Test de l'analyseur AST"""
    analyzer = ASTAnalyzer()

    # Test sur un fichier
    test_file = Path(__file__)
    result = analyzer.analyze_file(test_file)

    if result:
        print(f"Analysis of {test_file}:")
        print(f"  Metrics: {result['metrics']}")
        print(f"  Patterns: {result['patterns']}")
        print(f"  Code smells: {len(result['code_smells'])}")
        print(f"  Secrets: {len(result['secrets'])}")
    else:
        print("Analysis failed")


if __name__ == "__main__":
    main()
