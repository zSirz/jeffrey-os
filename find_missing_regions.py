#!/usr/bin/env python3
"""
Script de détection des modules manquants pour les 4 régions
"""

import json
import os
import re

# Définir les patterns de recherche
REGION_PATTERNS = {
    "executive": {
        "class_keywords": [
            "Executive",
            "Decision",
            "Planner",
            "Sovereign",
            "Verdict",
            "Metacognition",
            "Command",
            "Control",
        ],
        "method_keywords": [
            "decide",
            "plan",
            "execute",
            "choose",
            "prioritize",
            "strategize",
            "command",
            "control",
        ],
        "required_methods": [
            "process",
            "decide",
            "plan",
            "execute",
        ],  # Au moins une de ces méthodes
    },
    "motor": {
        "class_keywords": [
            "Motor",
            "Output",
            "Generator",
            "Response",
            "Executor",
            "Action",
            "Actuator",
        ],
        "method_keywords": [
            "generate",
            "output",
            "execute",
            "respond",
            "act",
            "produce",
            "actuate",
        ],
        "required_methods": ["process", "generate", "execute", "output"],
    },
    "language": {
        "class_keywords": ["Language", "Broca", "Wernicke", "NLP", "Linguistic", "Speech", "Text"],
        "method_keywords": [
            "generate_text",
            "translate",
            "parse",
            "understand",
            "speak",
            "analyze_text",
        ],
        "required_methods": ["process", "generate", "translate", "parse"],
    },
    "integration": {
        "class_keywords": [
            "Integration",
            "Bridge",
            "Connector",
            "Pipeline",
            "Orchestrat",
            "Synthesis",
            "Fusion",
        ],
        "method_keywords": [
            "integrate",
            "connect",
            "orchestrate",
            "pipeline",
            "synthesize",
            "fuse",
            "bridge",
        ],
        "required_methods": ["process", "integrate", "orchestrate", "synthesize"],
    },
}

EXCLUDES = (
    "venv",
    ".venv",
    "env",
    "site-packages",
    "node_modules",
    "__pycache__",
    ".git",
    "test",
    "backup",
    "stubs",
)


def analyze_file(filepath):
    """Analyse un fichier pour détecter sa région et ses capacités"""
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Exclure les stubs
        if "Stub" in content or re.search(r"class\s+\w*Stub\b", content):
            return None

        # Extraire les classes
        classes = re.findall(r"class\s+(\w+)", content)
        if not classes:
            return None

        # Extraire les méthodes
        methods = re.findall(r"def\s+(\w+)\(", content)
        if not methods:
            return None

        # Tester chaque région
        results = {}
        for region, patterns in REGION_PATTERNS.items():
            score = 0
            matched_methods = []

            # Score basé sur les noms de classes
            for cls in classes:
                if any(kw.lower() in cls.lower() for kw in patterns["class_keywords"]):
                    score += 20
                    print(f"    Class match: {cls} for {region}")

            # Score basé sur les méthodes
            for method in methods:
                if any(kw in method for kw in patterns["method_keywords"]):
                    score += 10
                    matched_methods.append(method)

                # Bonus si méthode requise
                if method in patterns["required_methods"]:
                    score += 30
                    matched_methods.append(f"✅ {method}")

            # Bonus pour taille (modules non-triviaux)
            if len(content) > 500:  # Au moins 500 caractères
                score += min(len(content) / 1000, 20)

            if score > 15:  # Seuil plus élevé pour éviter les faux positifs
                results[region] = {
                    "score": score,
                    "classes": classes,
                    "methods": matched_methods,
                    "size": len(content),
                }

        return results if results else None

    except Exception as e:
        print(f"    Error analyzing {filepath}: {e}")
        return None


def find_missing_regions():
    """Trouve les meilleurs modules pour les 4 régions manquantes"""
    candidates = {"executive": [], "motor": [], "language": [], "integration": []}

    print("🔍 Scanning all Python files...")
    for root, _, files in os.walk("src/jeffrey"):
        if any(x in root for x in EXCLUDES):
            continue

        for filename in files:
            if not filename.endswith(".py"):
                continue

            filepath = os.path.join(root, filename)
            print(f"  Analyzing: {filepath}")
            results = analyze_file(filepath)

            if results:
                for region, info in results.items():
                    candidates[region].append((filepath, info))
                    print(f"  ✅ Found {region} candidate: {filepath} (score: {info['score']:.1f})")

    # Sélectionner les meilleurs
    best_modules = {}
    print("\n📊 BEST MODULES FOUND:")
    for region in ["executive", "motor", "language", "integration"]:
        if candidates[region]:
            # Trier par score
            sorted_candidates = sorted(candidates[region], key=lambda x: x[1]["score"], reverse=True)
            best = sorted_candidates[0]
            best_modules[region] = {
                "path": best[0],
                "score": best[1]["score"],
                "classes": best[1]["classes"],
                "methods": best[1]["methods"],
                "size": best[1]["size"],
            }
            print(f"\n✅ BEST {region.upper()}: {best[0]}")
            print(f"   Score: {best[1]['score']:.1f}")
            print(f"   Classes: {', '.join(best[1]['classes'][:3])}")
            print(f"   Key methods: {', '.join(best[1]['methods'][:5])}")

            # Afficher les 3 meilleurs
            if len(sorted_candidates) > 1:
                print("   Alternatives:")
                for i, (path, info) in enumerate(sorted_candidates[1:4]):
                    print(f"     {i + 2}. {path} (score: {info['score']:.1f})")
        else:
            print(f"\n❌ NO MODULE FOUND FOR {region.upper()}")

    # Sauvegarder
    with open("missing_regions.json", "w") as f:
        json.dump(best_modules, f, indent=2)

    print("\n💾 Results saved to missing_regions.json")
    return best_modules


if __name__ == "__main__":
    find_missing_regions()
