#!/usr/bin/env python3
"""
Détection stricte des modules emotion/conscience.
"""

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


def analyze_file(filepath):
    """Analyse un fichier pour détecter sa catégorie"""
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
        categories = {}

        # Émotion
        if any(kw in filename for kw in ["emotion", "feeling", "affect", "mood"]):
            score = 0
            if "src/jeffrey/core" in filepath:
                score += 20
            if "__jeffrey_meta__" in content:
                score += 50
            if "def process(" in content:
                score += 30
            if "async def" in content:
                score += 10
            if len(content) < 500:
                score -= 20
            score += min(len(content) / 1000, 30)
            categories["emotion"] = score

        # Conscience
        if any(kw in filename for kw in ["conscience", "consciousness", "awareness", "cognitive"]):
            score = 0
            if "src/jeffrey/core" in filepath:
                score += 20
            if "__jeffrey_meta__" in content:
                score += 50
            if "def process(" in content or "def analyze(" in content:
                score += 30
            if "async def" in content:
                score += 10
            if len(content) < 500:
                score -= 20
            score += min(len(content) / 1000, 30)
            categories["conscience"] = score

        return categories if categories else None

    except Exception:
        return None


def find_best_modules():
    """Trouve les meilleurs modules"""
    candidates = {"emotion": [], "conscience": []}

    print("🔍 Scanning modules...", file=sys.stderr)

    # Borner la recherche (amélioration 7)
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
                categories = analyze_file(filepath)

                if categories:
                    for cat, score in categories.items():
                        if score > 0:
                            candidates[cat].append((filepath, score))
                            print(f"   Found {cat}: {filepath} (score: {score:.1f})", file=sys.stderr)

    result = {}
    for cat in ["emotion", "conscience"]:
        if candidates[cat]:
            best = max(candidates[cat], key=lambda x: x[1])
            result[cat] = best[0]
            print(f"\n✅ Best {cat}: {best[0]} (score: {best[1]:.1f})", file=sys.stderr)
        else:
            print(f"\n⚠️  No {cat} module found", file=sys.stderr)
            result[cat] = ""

    return result


if __name__ == "__main__":
    best = find_best_modules()

    with open("best_modules.json", "w") as f:
        json.dump(best, f, indent=2)

    print("\n✅ Results written to best_modules.json", file=sys.stderr)
