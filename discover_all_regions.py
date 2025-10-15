#!/usr/bin/env python3
"""
Découvre les 6 régions manquantes avec validation.
"""

import importlib.util
import inspect
import json
import os
import sys

EXCLUDES = (
    "venv",
    ".venv",
    "env",
    "site-packages",
    "node_modules",
    "__pycache__",
    ".git",
    "test",
    ".backup",
    "backup",
    "simple_modules",
    "stubs",
)

REGIONS = {
    "perception": {
        "keywords": ["perception", "sensor", "vision", "input", "parser"],
        "methods": ["parse", "process", "analyze"],
    },
    "memory": {
        "keywords": ["memory", "mnemonic", "recall", "working_memory", "cortex"],
        "methods": ["store", "recall", "process"],
    },
    "language": {
        "keywords": ["language", "broca", "wernicke", "nlp", "linguistic"],
        "methods": ["generate", "translate", "process"],
    },
    "executive": {
        "keywords": ["executive", "decision", "planner", "sovereign", "verdict"],
        "methods": ["decide", "plan", "execute", "process"],
    },
    "motor": {
        "keywords": ["motor", "action", "output", "executor", "generator"],
        "methods": ["execute", "generate", "act", "process"],
    },
    "integration": {
        "keywords": ["integration", "bridge", "connector", "orchestrator", "pipeline"],
        "methods": ["integrate", "connect", "orchestrate", "process"],
    },
}


def validate_module(filepath):
    """Valide qu'un module est importable et instanciable"""
    try:
        spec = importlib.util.spec_from_file_location("test_mod", filepath)
        if not spec or not spec.loader:
            return False

        mod = importlib.util.module_from_spec(spec)

        # Import sûr
        import os

        os.environ["JEFFREY_OFFLINE"] = "1"

        spec.loader.exec_module(mod)

        # Chercher une classe instanciable
        for name, obj in vars(mod).items():
            if inspect.isclass(obj) and "Stub" not in obj.__name__:
                try:
                    sig = inspect.signature(obj.__init__)
                    ok = True
                    for param_name, param in list(sig.parameters.items())[1:]:
                        if param.default is inspect._empty and param.kind not in (
                            param.VAR_POSITIONAL,
                            param.VAR_KEYWORD,
                        ):
                            ok = False
                            break

                    if ok:
                        inst = obj()
                        # Vérifier méthode
                        for method_name in ["process", "analyze", "run", "execute"]:
                            if hasattr(inst, method_name):
                                return True
                except Exception:
                    continue

        return False
    except Exception:
        return False


def analyze_file_for_region(filepath):
    """Analyse si un fichier correspond à une région"""
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            content = f.read()

        if "Stub" in content or "stub" in filepath.lower():
            return None
        if "simple_" in filepath.lower():
            return None
        if len(content) < 300:
            return None

        filename = os.path.basename(filepath).lower()
        results = {}

        for region, criteria in REGIONS.items():
            score = 0

            for kw in criteria["keywords"]:
                if kw in filename:
                    score += 30

            for method in criteria["methods"]:
                if f"def {method}(" in content:
                    score += 20

            if "src/jeffrey/core" in filepath:
                score += 15

            if "__jeffrey_meta__" in content:
                score += 25

            score += min(len(content) / 1000, 20)

            if score > 0:
                results[region] = score

        return results if results else None

    except Exception:
        return None


def discover_regions():
    """Découvre toutes les régions"""
    candidates = {region: [] for region in REGIONS.keys()}

    print("🔍 Scanning for 6 regions...", file=sys.stderr)

    search_paths = ["src/jeffrey/core", "src/jeffrey/services"]

    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue

        for root, _, files in os.walk(search_path):
            if any(x in root for x in EXCLUDES):
                continue

            for filename in files:
                if not filename.endswith(".py"):
                    continue

                filepath = os.path.join(root, filename)
                results = analyze_file_for_region(filepath)

                if results:
                    for region, score in results.items():
                        candidates[region].append((filepath, score))
                        print(f"   Found {region}: {filepath} (score: {score:.1f})", file=sys.stderr)

    # Sélectionner et valider
    best = {}
    for region in REGIONS.keys():
        if candidates[region]:
            # Trier par score
            sorted_candidates = sorted(candidates[region], key=lambda x: x[1], reverse=True)

            # Essayer les candidats jusqu'à trouver un valide
            found = False
            for filepath, score in sorted_candidates:
                if validate_module(filepath):
                    best[region] = filepath
                    print(f"\n✅ Best {region}: {filepath} (validated)", file=sys.stderr)
                    found = True
                    break

            if not found:
                print(f"\n⚠️  No valid {region} module found", file=sys.stderr)
                best[region] = ""
        else:
            print(f"\n⚠️  No {region} module found", file=sys.stderr)
            best[region] = ""

    return best


if __name__ == "__main__":
    discovered = discover_regions()

    with open("discovered_regions.json", "w") as f:
        json.dump(discovered, f, indent=2)

    print("\n✅ Results written to discovered_regions.json", file=sys.stderr)
