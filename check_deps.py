#!/usr/bin/env python3
"""
Vérifie toutes les dépendances du projet Jeffrey OS
Scan les imports et vérifie leur disponibilité
"""

import importlib
import subprocess
import sys
from pathlib import Path

# Modules connus et leurs packages pip correspondants
KNOWN_MAPPINGS = {
    "cv2": "opencv-python",
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "msgpack": "msgpack",
    "dotenv": "python-dotenv",
    "nats": "nats-py",
    "uvloop": "uvloop",
    "openai": "openai",
    "kivy": "kivy",
    "streamlit": "streamlit",
    "redis": "redis",
    "httpx": "httpx",
    "aiohttp": "aiohttp",
    "pydantic": "pydantic",
    "networkx": "networkx",
    "cachetools": "cachetools",
    "numpy": "numpy",
    "scipy": "scipy",
    "prometheus_client": "prometheus-client",
    "psutil": "psutil",
    "cryptography": "cryptography",
    "joblib": "joblib",
    "langdetect": "langdetect",
}

# Modules de la stdlib Python (à ignorer)
STDLIB_MODULES = {
    "asyncio",
    "time",
    "json",
    "hashlib",
    "re",
    "logging",
    "uuid",
    "typing",
    "dataclasses",
    "datetime",
    "enum",
    "collections",
    "os",
    "sys",
    "pathlib",
    "threading",
    "concurrent",
    "functools",
    "itertools",
    "warnings",
    "copy",
    "pickle",
    "base64",
    "random",
    "math",
    "statistics",
    "decimal",
    "fractions",
    "numbers",
    "string",
    "textwrap",
    "unicodedata",
    "codecs",
    "io",
    "importlib",
    "inspect",
    "traceback",
    "weakref",
    "abc",
    "contextlib",
    "types",
    "socket",
    "ssl",
    "subprocess",
    "multiprocessing",
    "queue",
    "struct",
    "binascii",
    "zlib",
    "gzip",
    "bz2",
    "lzma",
    "tarfile",
    "zipfile",
    "tempfile",
    "glob",
    "fnmatch",
    "shutil",
    "sqlite3",
    "urllib",
    "http",
    "email",
    "html",
    "xml",
    "csv",
    "configparser",
    "argparse",
    "getopt",
    "locale",
    "platform",
    "errno",
    "signal",
    "atexit",
    "gc",
    "builtins",
    "__future__",
    "cProfile",
    "pstats",
}


def check_import(module_name: str) -> tuple[bool, str]:
    """Vérifie si un module peut être importé"""
    try:
        importlib.import_module(module_name)
        return True, "OK"
    except ImportError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Error: {e}"


def scan_project_imports() -> set[str]:
    """Scan tous les fichiers Python pour trouver les imports"""
    imports = set()

    # Scan src directory
    src_path = Path("src")
    if not src_path.exists():
        print("⚠️ Directory 'src' not found, scanning current directory")
        src_path = Path(".")

    py_files = list(src_path.rglob("*.py"))
    print(f"🔍 Scanning {len(py_files)} Python files...")

    for py_file in py_files:
        if "__pycache__" in str(py_file):
            continue

        try:
            with open(py_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()

                    # import module
                    if line.startswith("import ") and not line.startswith("import src"):
                        parts = line.replace("import ", "").split(",")
                        for part in parts:
                            module = part.strip().split(" as ")[0].split(".")[0]
                            if module:
                                imports.add(module)

                    # from module import ...
                    elif line.startswith("from ") and not line.startswith("from src") and not line.startswith("from ."):
                        module = line.split()[1].split(".")[0]
                        if module:
                            imports.add(module)
        except Exception as e:
            print(f"⚠️ Error reading {py_file}: {e}")

    return imports


def get_installed_packages() -> set[str]:
    """Récupère la liste des packages installés via pip"""
    try:
        result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)
        packages = set()
        for line in result.stdout.strip().split("\n"):
            if "==" in line:
                package = line.split("==")[0].lower().replace("-", "_")
                packages.add(package)
        return packages
    except Exception as e:
        print(f"⚠️ Could not get pip packages: {e}")
        return set()


def main():
    print("=" * 60)
    print("🔍 Jeffrey OS - Dependency Checker")
    print("=" * 60)

    # Scan imports
    print("\n📦 Scanning project imports...")
    imports = scan_project_imports()

    # Filter out stdlib and local imports
    external_imports = {
        imp
        for imp in imports
        if imp not in STDLIB_MODULES and not imp.startswith("_") and imp not in ["src", "tests", "scripts", "config"]
    }

    print(f"Found {len(external_imports)} external dependencies\n")

    # Get installed packages
    installed_packages = get_installed_packages()

    # Check each import
    missing = []
    installed = []
    optional = []

    for module in sorted(external_imports):
        package = KNOWN_MAPPINGS.get(module, module)
        success, error = check_import(module)

        if success:
            installed.append(module)
            print(f"✅ {module:20} - Installed")
        else:
            # Check if it's in pip packages
            package_normalized = package.lower().replace("-", "_")
            if package_normalized in installed_packages:
                print(f"⚠️ {module:20} - Package installed but can't import")
                optional.append((module, package))
            else:
                missing.append((module, package))
                print(f"❌ {module:20} - Missing (install: pip install {package})")

    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)

    print(f"✅ Installed: {len(installed)} modules")
    print(f"❌ Missing:   {len(missing)} modules")
    print(f"⚠️ Optional:  {len(optional)} modules")

    if missing:
        print("\n🔧 To install missing packages:")
        packages_to_install = [pkg for _, pkg in missing]
        print(f"pip install {' '.join(set(packages_to_install))}")

    # Check critical dependencies
    print("\n" + "=" * 60)
    print("🎯 CRITICAL DEPENDENCIES CHECK")
    print("=" * 60)

    critical = [
        ("httpx", "API client"),
        ("networkx", "Graph operations"),
        ("redis", "Memory storage"),
        ("kivy", "UI framework"),
        ("numpy", "Numerical computing"),
        ("pydantic", "Data validation"),
        ("msgpack", "Serialization"),
    ]

    all_critical_ok = True
    for module, description in critical:
        success, _ = check_import(module)
        if success:
            print(f"✅ {module:15} ({description})")
        else:
            print(f"❌ {module:15} ({description}) - CRITICAL!")
            all_critical_ok = False

    if all_critical_ok:
        print("\n✅ All critical dependencies are installed!")
        print("Jeffrey OS is ready to run!")
    else:
        print("\n❌ Some critical dependencies are missing.")
        print("Run: pip install -r requirements.txt")

    return 0 if all_critical_ok else 1


if __name__ == "__main__":
    sys.exit(main())
