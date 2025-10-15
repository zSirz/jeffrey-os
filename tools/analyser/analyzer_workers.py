"""Workers picklables pour ProcessPoolExecutor (macOS = spawn)"""

import ast
from pathlib import Path


def analyze_python_fast_worker(path_str: str) -> dict:
    """Fonction top-level pour ProcessPoolExecutor."""
    p = Path(path_str)
    try:
        txt = p.read_text(encoding='utf-8', errors='ignore')
        tree = ast.parse(txt)

        lines = txt.count('\n') + 1
        classes = []
        functions = []
        imports = []
        complexity = 0
        has_tests = False
        has_docs = False

        for n in ast.walk(tree):
            if isinstance(n, ast.ClassDef):
                classes.append(n.name)
                has_docs |= bool(ast.get_docstring(n))
            elif isinstance(n, ast.FunctionDef):
                functions.append(n.name)
                has_tests |= n.name.startswith('test_')
                has_docs |= bool(ast.get_docstring(n))
            elif isinstance(n, ast.Import):
                imports += [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module:
                imports.append(n.module)
            elif isinstance(n, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1

        return {
            'lines': lines,
            'classes': classes,
            'functions': functions,
            'imports': imports,
            'complexity': min(complexity, 100),  # Cap à 100
            'has_tests': has_tests,
            'has_docs': has_docs,
        }
    except Exception as e:
        return {'error': str(e)[:100]}  # Limiter taille erreur


def hash_file_worker(path_str: str, partial: bool = True) -> str:
    """Worker pour hash parallèle - version corrigée pour compatibilité."""
    from pathlib import Path

    p = Path(path_str)

    # Essayer blake3 d'abord
    try:
        from blake3 import blake3

        hash_func = blake3
    except ImportError:
        import hashlib

        hash_func = hashlib.sha256

    try:
        h = hash_func()
        with p.open("rb") as f:
            if partial:
                h.update(f.read(128 * 1024))
                return f"P:{h.hexdigest()[:32]}"
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return f"F:{h.hexdigest()}"
    except Exception as e:
        return f"ERROR:{type(e).__name__}"


# Version ancienne pour compatibilité si besoin
def hash_file_worker_legacy(args: tuple) -> tuple:
    """Worker pour hash parallèle - ancienne version."""
    path_str, partial = args
    result = hash_file_worker(path_str, partial)
    return (path_str, result)
