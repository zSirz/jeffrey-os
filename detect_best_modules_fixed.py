#!/usr/bin/env python3
import ast
import json
import os


def has_jeffrey_meta(filepath):
    """Check si le fichier a __jeffrey_meta__"""
    try:
        with open(filepath) as f:
            content = f.read()
            if "__jeffrey_meta__" in content:
                tree = ast.parse(content)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if hasattr(target, "id") and target.id == "__jeffrey_meta__":
                                return True
        return False
    except:
        return False


def find_best_modules():
    """Trouve les meilleurs modules par métadonnées et taille"""
    candidates = {"emotion": [], "conscience": []}

    # FIX GPT #1: Exclure venv et site-packages
    EXCLUDES = (
        "venv",
        ".venv",
        "env",
        "site-packages",
        "node_modules",
        "__pycache__",
        "test",
        ".git",
        "backup",
    )

    for root, _, files in os.walk("src/jeffrey"):
        if any(x in root for x in EXCLUDES):
            continue

        for filename in files:
            if not filename.endswith(".py"):
                continue

            filepath = os.path.join(root, filename)
            name_lower = filename.lower()

            score = 0

            # Points pour le nom
            if "emotion" in name_lower:
                score += 10
                category = "emotion"
            elif "conscience" in name_lower:
                score += 10
                category = "conscience"
            else:
                continue

            # Bonus pour métadonnées Jeffrey
            if has_jeffrey_meta(filepath):
                score += 50

            # Points pour la taille
            try:
                size = os.path.getsize(filepath)
                score += min(size / 1000, 20)
            except:
                pass

            # Pénalité pour stubs
            try:
                with open(filepath) as f:
                    content = f.read()
                    if "Stub" in content:
                        score -= 100
                    # Bonus si vraies méthodes
                    if "def process(" in content or "async def process(" in content:
                        score += 20
                    if "class EmotionEngine" in content or "class ConscienceEngine" in content:
                        score += 30
                    if "async def initialize(" in content:
                        score += 10
            except:
                pass

            candidates[category].append((filepath, score))

    result = {}

    # Pour emotion, utiliser les meilleurs candidats connus
    if not candidates["emotion"]:
        # Fallback aux paths connus
        emotion_paths = [
            "src/jeffrey/core/orchestration/emotional_core.py",
            "src/jeffrey/core/orchestration/emotion_engine_bridge.py",
            "src/jeffrey/core/emotions/emotion_orchestrator_v2.py",
        ]
        for path in emotion_paths:
            if os.path.exists(path):
                result["emotion"] = path
                print(f"✅ Module emotion (fallback): {path}")
                break
    else:
        best = max(candidates["emotion"], key=lambda x: x[1])
        result["emotion"] = best[0]
        print(f"✅ Meilleur module emotion: {best[0]} (score: {best[1]:.1f})")

    # Pour conscience
    if not candidates["conscience"]:
        # Utiliser le path connu
        conscience_path = "src/jeffrey/core/consciousness/conscience_engine.py"
        if os.path.exists(conscience_path):
            result["conscience"] = conscience_path
            print(f"✅ Module conscience (direct): {conscience_path}")
    else:
        best = max(candidates["conscience"], key=lambda x: x[1])
        result["conscience"] = best[0]
        print(f"✅ Meilleur module conscience: {best[0]} (score: {best[1]:.1f})")

    # Vérifier qu'on a bien les deux
    if not result.get("emotion"):
        result["emotion"] = "src/jeffrey/core/orchestration/emotional_core.py"
    if not result.get("conscience"):
        result["conscience"] = "src/jeffrey/core/consciousness/conscience_engine.py"

    return result


if __name__ == "__main__":
    best = find_best_modules()
    with open("best_modules.json", "w") as f:
        json.dump(best, f)
    print(json.dumps(best))
