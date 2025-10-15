#!/usr/bin/env python3
import ast
import importlib.util
from collections import defaultdict
from pathlib import Path


def extract_imports(file_path):
    """Extraire tous les imports d'un fichier Python"""
    imports = set()
    try:
        with open(file_path, encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=file_path)

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
    except Exception as e:
        print(f"⚠️ Erreur parsing {file_path}: {e}")
    return imports


def is_local_module(module_name, project_root):
    """Vérifier si c'est un module local du projet"""
    possible_paths = [
        project_root / module_name,
        project_root / 'src' / module_name,
        project_root / 'src' / 'jeffrey' / module_name,
        project_root / 'apps' / module_name,
        project_root / 'shared' / module_name,
        project_root / 'rust' / module_name,
    ]
    for path in possible_paths:
        if path.exists():
            return True
    return False


def is_stdlib_module(module_name):
    """Vérifier si c'est un module de la stdlib Python"""
    stdlib_modules = {
        'os',
        'sys',
        'json',
        'time',
        'datetime',
        'random',
        'math',
        're',
        'collections',
        'itertools',
        'functools',
        'typing',
        'pathlib',
        'asyncio',
        'threading',
        'multiprocessing',
        'subprocess',
        'queue',
        'logging',
        'warnings',
        'traceback',
        'inspect',
        'ast',
        'dis',
        'pickle',
        'copy',
        'weakref',
        'gc',
        'enum',
        'dataclasses',
        'unittest',
        'doctest',
        'pytest',
        'mock',
        'abc',
        'contextlib',
        'decimal',
        'fractions',
        'numbers',
        'hashlib',
        'hmac',
        'secrets',
        'uuid',
        'base64',
        'sqlite3',
        'csv',
        'configparser',
        'argparse',
        'urllib',
        'http',
        'socket',
        'email',
        'ftplib',
        'io',
        'gzip',
        'zipfile',
        'tarfile',
        'tempfile',
        'shutil',
        'glob',
        'fnmatch',
        'difflib',
        'textwrap',
        'pprint',
        'statistics',
        'operator',
        'heapq',
        'bisect',
        'array',
        'struct',
        'codecs',
        'locale',
        'gettext',
        'platform',
        'sysconfig',
        'importlib',
        'pkgutil',
        'types',
        'builtins',
        '__future__',
        'string',
    }
    return module_name in stdlib_modules


def analyze_project(project_root):
    """Analyser tous les fichiers Python du projet"""
    project_root = Path(project_root)
    all_imports = defaultdict(list)
    missing_imports = defaultdict(list)

    # Collecter tous les fichiers Python
    py_files = list(project_root.glob('**/*.py'))
    print(f"\n📊 Analyse de {len(py_files)} fichiers Python...")

    for py_file in py_files:
        relative_path = py_file.relative_to(project_root)
        imports = extract_imports(py_file)

        for imp in imports:
            all_imports[imp].append(str(relative_path))

            # Vérifier si c'est un import manquant
            if not is_stdlib_module(imp) and not is_local_module(imp, project_root):
                # Essayer de l'importer pour voir s'il est installé
                spec = importlib.util.find_spec(imp)
                if spec is None:
                    missing_imports[imp].append(str(relative_path))

    return all_imports, missing_imports


def main():
    project_root = Path('/Users/davidproz/Desktop/Jeffrey_OS')

    print("🔍 Analyse des dépendances Jeffrey OS...")
    all_imports, missing_imports = analyze_project(project_root)

    # Rapport
    print("\n" + "=" * 60)
    print("📋 RAPPORT D'ANALYSE DES DÉPENDANCES")
    print("=" * 60)

    # Statistiques globales
    print("\n📊 STATISTIQUES:")
    print(f"   • Total imports uniques: {len(all_imports)}")
    print(f"   • Imports manquants: {len(missing_imports)}")

    # Modules manquants
    if missing_imports:
        print("\n❌ MODULES MANQUANTS (non installés):")
        sorted_missing = sorted(missing_imports.items(), key=lambda x: len(x[1]), reverse=True)
        for module, files in sorted_missing[:20]:  # Top 20
            print(f"\n   📦 {module} ({len(files)} fichiers)")
            for file in files[:3]:  # Montrer max 3 fichiers
                print(f"      └─ {file}")
            if len(files) > 3:
                print(f"      └─ ... et {len(files) - 3} autres fichiers")

    # Modules les plus utilisés
    print("\n🔥 TOP 10 IMPORTS (tous types):")
    sorted_all = sorted(all_imports.items(), key=lambda x: len(x[1]), reverse=True)
    for module, files in sorted_all[:10]:
        status = "✅" if is_stdlib_module(module) else "📦"
        if module in missing_imports:
            status = "❌"
        print(f"   {status} {module}: {len(files)} fichiers")

    # Générer requirements
    if missing_imports:
        print("\n📝 MODULES À AJOUTER DANS requirements.txt:")
        for module in sorted(missing_imports.keys()):
            # Mapper certains imports à leurs packages pip
            pip_mapping = {
                'cv2': 'opencv-python',
                'PIL': 'Pillow',
                'yaml': 'PyYAML',
                'sklearn': 'scikit-learn',
                'wx': 'wxPython',
                'pyautogui': 'pyautogui',
                'plyer': 'plyer',
                'transformers': 'transformers',
                'sentence_transformers': 'sentence-transformers',
                'faiss': 'faiss-cpu',
                'firebase_admin': 'firebase-admin',
                'google': 'google-cloud-storage',
                'openai': 'openai',
                'elevenlabs': 'elevenlabs',
                'sounddevice': 'sounddevice',
                'speech_recognition': 'SpeechRecognition',
                'pyttsx3': 'pyttsx3',
                'kivy': 'kivy',
                'kivymd': 'kivymd',
            }
            package = pip_mapping.get(module, module)
            print(f"   {package}")


if __name__ == "__main__":
    main()
