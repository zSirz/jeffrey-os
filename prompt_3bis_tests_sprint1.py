#!/usr/bin/env python3
"""
JEFFREY OS - PROMPT 3-BIS : TESTS SPRINT 1 - VALIDATION COMPLÈTE
=================================================================

Ce script crée et lance le runner de tests Sprint 1 qui :
- Utilise EmotionDetectorV2 sur les 40 scénarios YAML
- Calcule Macro-F1, Accuracy, Confusion Matrix
- Mesure la latence (avg, p95)
- Génère un rapport JSON détaillé

OBJECTIF SPRINT 1 :
- Macro-F1 ≥ 0.70
- Accuracy ≥ 0.65
- Latence p95 ≤ 500ms

USAGE:
    python3 prompt_3bis_tests_sprint1.py

Équipe : Dream Team Jeffrey OS
"""

import os
import subprocess
import sys
from pathlib import Path

# ===============================================================================
# CONFIGURATION
# ===============================================================================

PROJECT_ROOT = Path(__file__).parent
TESTS_DIR = PROJECT_ROOT / "tests"
RESULTS_DIR = PROJECT_ROOT / "test_results"

print("=" * 80)
print("🚀 JEFFREY OS - PROMPT 3-BIS : TESTS SPRINT 1")
print("=" * 80)
print()
print("Ce script va :")
print("1. Créer le runner de tests Sprint 1")
print("2. Lancer les 40 scénarios YAML")
print("3. Calculer Macro-F1, Accuracy, Confusion Matrix")
print("4. Mesurer la latence (avg, p95)")
print("5. Générer le rapport JSON")
print()

# ===============================================================================
# ÉTAPE 1 : CRÉATION DU RUNNER SPRINT 1
# ===============================================================================

print("📝 [1/2] Création du runner de tests Sprint 1...")
print()

runner_code = '''#!/usr/bin/env python3
"""
JEFFREY OS - Runner Sprint 1 : Évaluation Détection Émotionnelle
================================================================

Ce runner évalue EmotionDetectorV2 sur les 40 scénarios YAML.

Métriques calculées :
- Macro-F1 par émotion
- Accuracy globale
- Confusion Matrix
- Latence (avg, p95)

Équipe : Dream Team Jeffrey OS (implémentation GPT/Marc)
"""

import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass
from collections import Counter, defaultdict
import yaml

# Assure imports locaux
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Modules Sprint 1
from jeffrey.nlp.emotion_detector_v2 import EmotionDetectorV2


@dataclass
class Example:
    """Exemple de prédiction pour évaluation"""
    scenario: str
    turn_idx: int
    text: str
    gold: str
    pred: str
    latency_ms: float


def load_scenarios(dirpath: Path):
    """
    Charge tous les fichiers YAML du dossier.

    Supporte 2 schémas:
    - Nouveau: {metadata, conversation: [{role, content, expected_emotion?}], validation?}
    - Ancien : {meta, session: [{user: "...", assert_* / expect_* ...}]}
    """
    files = sorted(dirpath.glob("*.yaml"))
    scenarios = []

    for f in files:
        try:
            data = yaml.safe_load(f.read_text(encoding='utf-8'))
            scenarios.append((f.name, data))
        except Exception as e:
            print(f"⚠️  YAML invalide {f.name}: {e}")

    return scenarios


def iterate_user_turns(name, data):
    """
    Itère sur les tours utilisateur annotés.

    Yields:
        (turn_idx, text, gold_emotion)
    """
    # Nouveau schéma
    if "conversation" in data:
        for i, step in enumerate(data["conversation"]):
            if step.get("role") == "user":
                text = step.get("content", "")
                gold = step.get("expected_emotion", None)
                yield i, text, gold
        return

    # Ancien schéma
    if "session" in data:
        for i, step in enumerate(data["session"]):
            if "user" in step:
                text = step.get("user", "")
                # Pas toujours d'annotation → None
                gold = step.get("expected_emotion", None) or step.get("expect_emotion", None)
                yield i, text, gold
        return


def compute_metrics(examples):
    """
    Calcule Macro-F1, Accuracy, Confusion Matrix.

    Args:
        examples: Liste d'Example avec gold et pred

    Returns:
        Dict avec métriques
    """
    # Récupérer toutes les émotions gold uniques
    labels = sorted(set(ex.gold for ex in examples if ex.gold))

    if not labels:
        return {
            "note": "Aucune étiquette de vérité terrain trouvée dans les YAML.",
            "macro_f1": 0.0,
            "accuracy": 0.0,
            "support": 0,
            "confusion": {},
        }

    # Mapping label → index
    idx = {l: i for i, l in enumerate(labels)}

    # Confusion matrix
    conf = [[0 for _ in labels] for __ in labels]
    support = Counter()
    correct = 0

    for ex in examples:
        if not ex.gold:
            continue

        support[ex.gold] += 1

        if ex.pred == ex.gold:
            correct += 1

        # Si la préd n'est pas dans labels (ex: "neutral"), on l'ignore pour confusion
        if ex.pred in idx and ex.gold in idx:
            conf[idx[ex.gold]][idx[ex.pred]] += 1

    # F1 par label
    def f1_for(label):
        i = idx[label]
        tp = conf[i][i]
        fp = sum(conf[r][i] for r in range(len(labels)) if r != i)
        fn = sum(conf[i][c] for c in range(len(labels)) if c != i)

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    per_label_f1 = {l: round(f1_for(l), 4) for l in labels}
    macro_f1 = round(sum(per_label_f1.values()) / len(labels), 4)
    acc = round(correct / sum(support.values()), 4) if support else 0.0

    # Confusion matrix lisible
    confusion = {
        "labels": labels,
        "matrix": conf,
    }

    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "support": sum(support.values()),
        "per_label_f1": per_label_f1,
        "per_label_support": dict(support),
        "confusion": confusion,
    }


def main():
    """Point d'entrée principal"""
    base = ROOT
    conv_dir = base / "tests" / "convos"

    if not conv_dir.exists():
        print("⚠️  tests/convos/ introuvable. Crée d'abord les scénarios.")
        sys.exit(1)

    # Initialiser le détecteur
    det = EmotionDetectorV2()
    scenarios = load_scenarios(conv_dir)

    all_examples = []
    latencies = []

    print("=" * 80)
    print("🧪 JEFFREY OS — Sprint 1 Emotion Eval Runner")
    print("=" * 80)
    print()
    print(f"📁 Dossier : {conv_dir}")
    print(f"📊 Scénarios détectés : {len(scenarios)}")
    print()
    print("⏳ Traitement en cours...")
    print()

    # Traiter tous les scénarios
    for name, data in scenarios:
        for turn_idx, text, gold in iterate_user_turns(name, data):
            if not text.strip():
                continue

            # Détection avec mesure de latence
            t0 = time.perf_counter()
            res = det.detect(text)
            dt = (time.perf_counter() - t0) * 1000.0

            latencies.append(dt)
            pred = res.primary

            all_examples.append(
                Example(name, turn_idx, text, gold, pred, dt)
            )

    # Garder uniquement les exemples annotés pour le calcul F1
    annotated = [ex for ex in all_examples if ex.gold]

    print(f"✅ Traitement terminé : {len(all_examples)} tours, {len(annotated)} annotés")
    print()

    # Calculer les métriques
    metrics = compute_metrics(annotated)

    # Latence
    lat_sorted = sorted(latencies)
    p95 = round(lat_sorted[int(0.95 * len(lat_sorted)) - 1], 2) if lat_sorted else 0.0
    avg = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    # Rapport complet
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_scenarios": len(scenarios),
        "num_turns": len(all_examples),
        "num_annotated": len(annotated),
        "latency_ms": {
            "avg": avg,
            "p95": p95,
            "min": round(min(latencies), 2) if latencies else 0.0,
            "max": round(max(latencies), 2) if latencies else 0.0
        },
        "emotion_metrics": metrics,
        "samples": [
            {
                "scenario": ex.scenario,
                "turn_idx": ex.turn_idx,
                "gold": ex.gold,
                "pred": ex.pred,
                "match": "✅" if ex.gold == ex.pred else "❌",
                "latency_ms": round(ex.latency_ms, 2),
                "text": ex.text[:200]
            } for ex in annotated[:25]  # Échantillon
        ]
    }

    # Sauvegarder le rapport
    out_dir = ROOT / "test_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "sprint1_emotion_eval.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    # Afficher le résumé
    print("=" * 80)
    print("📊 RÉSULTATS SPRINT 1")
    print("=" * 80)
    print()
    print(f"📈 MÉTRIQUES GLOBALES :")
    print(f"   • Tours annotés : {report['num_annotated']} / {report['num_turns']}")
    print(f"   • Macro-F1      : {metrics.get('macro_f1', 0.0):.3f}")
    print(f"   • Accuracy      : {metrics.get('accuracy', 0.0):.3f}")
    print()
    print(f"⚡ LATENCE :")
    print(f"   • Moyenne  : {avg}ms")
    print(f"   • P95      : {p95}ms")
    print(f"   • Min/Max  : {report['latency_ms']['min']}ms / {report['latency_ms']['max']}ms")
    print()

    # F1 par émotion
    if "per_label_f1" in metrics:
        print(f"🎭 F1 PAR ÉMOTION :")
        for label, f1 in sorted(metrics["per_label_f1"].items(), key=lambda x: -x[1]):
            support = metrics.get("per_label_support", {}).get(label, 0)
            print(f"   • {label:15s} : {f1:.3f}  (n={support})")
        print()

    # Confusion matrix
    if "labels" in metrics.get("confusion", {}):
        labels = metrics["confusion"]["labels"]
        matrix = metrics["confusion"]["matrix"]
        print(f"🔀 CONFUSION MATRIX :")
        print(f"   Labels : {', '.join(labels)}")
        print()

    print(f"💾 RAPPORT SAUVEGARDÉ :")
    print(f"   {out_json}")
    print()

    # Verdict Sprint 1
    macro_f1 = metrics.get("macro_f1", 0.0)
    accuracy = metrics.get("accuracy", 0.0)

    print("=" * 80)
    print("🎯 VERDICT SPRINT 1")
    print("=" * 80)
    print()

    if macro_f1 >= 0.70 and p95 <= 500:
        print("✅ OBJECTIFS ATTEINTS !")
        print(f"   • Macro-F1 : {macro_f1:.3f} (objectif ≥ 0.70) ✅")
        print(f"   • Latence  : {p95}ms (objectif ≤ 500ms) ✅")
        print()
        print("🚀 PRÊT POUR PROMPT 4 : Intégration Jeffrey OS !")
    elif macro_f1 >= 0.65:
        print("⚠️  PROCHE DE L'OBJECTIF")
        print(f"   • Macro-F1 : {macro_f1:.3f} (objectif ≥ 0.70) ⚠️")
        print(f"   • Latence  : {p95}ms (objectif ≤ 500ms) {'✅' if p95 <= 500 else '❌'}")
        print()
        print("💡 Quelques ajustements nécessaires avant intégration.")
    else:
        print("❌ OBJECTIFS NON ATTEINTS")
        print(f"   • Macro-F1 : {macro_f1:.3f} (objectif ≥ 0.70) ❌")
        print(f"   • Latence  : {p95}ms (objectif ≤ 500ms) {'✅' if p95 <= 500 else '❌'}")
        print()
        print("🔧 Besoin d'améliorer le lexique ou les patterns.")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
'''

# Créer le dossier tests si nécessaire
TESTS_DIR.mkdir(exist_ok=True)

# Écrire le fichier
runner_file = TESTS_DIR / "runner_convos_sprint1.py"
with open(runner_file, 'w', encoding='utf-8') as f:
    f.write(runner_code)

# Rendre exécutable
os.chmod(runner_file, 0o755)

print(f"✅ Fichier créé : {runner_file}")
print()

# ===============================================================================
# ÉTAPE 2 : EXÉCUTION DES TESTS
# ===============================================================================

print("🧪 [2/2] Lancement des tests Sprint 1...")
print()
print("⏳ Cela peut prendre 30-60 secondes...")
print()

# Exécuter avec PYTHONPATH correct
env = os.environ.copy()
env['PYTHONPATH'] = str(PROJECT_ROOT / "src")

try:
    result = subprocess.run(
        [sys.executable, str(runner_file)],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,  # 2 minutes max
    )

    # Afficher la sortie
    print(result.stdout)

    if result.stderr:
        print("⚠️  Warnings/Errors :")
        print(result.stderr)

    if result.returncode != 0:
        print(f"❌ Le runner a retourné un code d'erreur : {result.returncode}")
        sys.exit(1)

except subprocess.TimeoutExpired:
    print("❌ Timeout : Les tests ont pris trop de temps (>2 min)")
    sys.exit(1)
except Exception as e:
    print(f"❌ Erreur lors de l'exécution : {e}")
    sys.exit(1)

# ===============================================================================
# ÉTAPE 3 : AFFICHAGE DU RAPPORT
# ===============================================================================

print()
print("=" * 80)
print("📄 RAPPORT JSON DISPONIBLE")
print("=" * 80)
print()

report_file = RESULTS_DIR / "sprint1_emotion_eval.json"

if report_file.exists():
    print(f"✅ Rapport sauvegardé : {report_file}")
    print()
    print("📊 Pour voir les détails :")
    print(f"   cat {report_file}")
    print()
    print("   ou ouvre-le dans VS Code pour explorer les métriques.")
else:
    print("⚠️  Le rapport JSON n'a pas été généré.")

print()
print("=" * 80)
print("✅ PROMPT 3-BIS TERMINÉ !")
print("=" * 80)
print()
print("🎯 PROCHAINE ÉTAPE :")
print("   Si Macro-F1 ≥ 0.70 → PROMPT 4 (Intégration Jeffrey OS)")
print("   Si Macro-F1 < 0.70 → Améliorer le lexique et relancer")
print()
