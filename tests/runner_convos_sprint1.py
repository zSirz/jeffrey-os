#!/usr/bin/env python3
"""
JEFFREY OS - Runner Sprint 1 : Évaluation Détection Émotionnelle (PHASE 1 ML)
==============================================================================

Ce runner évalue le HybridEmotionDetector (ProtoClassifier + fallback V3) sur les 40 scénarios YAML.

Métriques calculées :
- Macro-F1 par émotion
- Accuracy globale
- Confusion Matrix
- Latence (avg, p95)
- ECE (Expected Calibration Error)
- Coverage (taux non-abstention)

Phase 1 : Transition lexical → sémantique avec apprentissage auto-adaptatif
"""

import argparse
import hashlib
import json
import logging
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Assure imports locaux
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Modules Sprint 1 + Phase 1
from jeffrey.ml.encoder import create_default_encoder
from jeffrey.ml.feedback import FeedbackEvent, FeedbackStore
from jeffrey.ml.proto import EmotionPrediction, ProtoClassifier
from jeffrey.nlp.emotion_detector_v3 import EmotionDetectorV3

# Flag pour activer/désactiver le nouveau système
USE_PROTO_CLASSIFIER = True  # Mettre False pour fallback V3 pur

logging.basicConfig(level=logging.INFO)

# Mapping 26 → 8 émotions core (Phase 1 pragmatique)
EMOTION_MAPPING_26_TO_8 = {
    # Core emotions (identité)
    "anger": "anger",
    "joy": "joy",
    "sadness": "sadness",
    "fear": "fear",
    "disgust": "disgust",
    "surprise": "surprise",
    "neutral": "neutral",
    "frustration": "frustration",
    # Mapping vers core
    "determination": "frustration",  # Proche de la frustration positive
    "relief": "joy",  # Soulagement = joie légère
    "exhaustion": "sadness",  # Épuisement = tristesse physique
    "better": "joy",  # Amélioration = joie
    "vulnerability": "fear",  # Vulnérabilité = peur
    "amusement": "joy",  # Amusement = joie
    "betrayal": "sadness",  # Trahison = tristesse intense
    "clarification": "neutral",  # Clarification = neutre
    "confusion": "frustration",  # Confusion = frustration cognitive
    "contentment": "joy",  # Contentement = joie calme
    "despair": "sadness",  # Désespoir = tristesse extrême
    "discomfort": "disgust",  # Inconfort = dégoût léger
    "motivation": "determination",  # Motivation = détermination (puis → frustration)
    "negative": "sadness",  # Négatif générique = tristesse
    "panic": "fear",  # Panique = peur extrême
    "pride": "joy",  # Fierté = joie
    "reflective": "neutral",  # Réflexif = neutre
    "tired": "exhaustion",  # Fatigué = épuisement (puis → sadness)
}

# === AJUSTEMENT 3 : CORE_EMOTIONS ET CHARGEMENT SEEDS GLOBAL ===

# Ordre canonique des 8 émotions (NE JAMAIS CHANGER L'ORDRE)
CORE_EMOTIONS = ["anger", "joy", "sadness", "fear", "disgust", "surprise", "neutral", "frustration"]

# Mapping fixe bidirectionnel
EMOTION_TO_IDX = {emo: i for i, emo in enumerate(CORE_EMOTIONS)}
IDX_TO_EMOTION = {i: emo for emo, i in EMOTION_TO_IDX.items()}


def load_bootstrap_seeds_once():
    """
    Charger les seeds bootstrap UNE SEULE FOIS (optimisation).

    Returns:
        seed_texts: List[str]
        seed_labels_encoded: List[int] (0..7)
    """
    seed_texts = []
    seed_labels_encoded = []

    seed_file = Path("tests/data/bootstrap_seed.yaml")

    if not seed_file.exists():
        logger = logging.getLogger(__name__)
        logger.warning(f"⚠️  Bootstrap seed file not found: {seed_file}")
        return [], []

    try:
        with open(seed_file) as f:
            seed_data = yaml.safe_load(f)

        for item in seed_data:
            text = item.get('text')
            emotion = item.get('emotion')

            if text and emotion:
                # Encoder immédiatement avec mapping fixe
                if emotion in EMOTION_TO_IDX:
                    seed_texts.append(text)
                    seed_labels_encoded.append(EMOTION_TO_IDX[emotion])
                else:
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Unknown emotion in seed: {emotion}")

        logger = logging.getLogger(__name__)
        logger.info(f"✅ Loaded {len(seed_texts)} bootstrap seeds globally")

        # Vérifier couverture des 8 classes
        unique_encoded = set(seed_labels_encoded)
        if len(unique_encoded) < len(CORE_EMOTIONS):
            missing_indices = set(range(len(CORE_EMOTIONS))) - unique_encoded
            missing_emotions = [CORE_EMOTIONS[i] for i in missing_indices]
            logger.warning(f"⚠️  Seeds missing classes: {missing_emotions}")

        return seed_texts, seed_labels_encoded

    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"❌ Error loading bootstrap seeds: {e}")
        return [], []


# Charger seeds UNE FOIS au démarrage du module
BOOTSTRAP_SEEDS_TEXTS, BOOTSTRAP_SEEDS_LABELS_ENCODED = load_bootstrap_seeds_once()


def map_emotion_to_core(emotion: str) -> str:
    """Mappe une émotion 26-label vers 8 core.

    Args:
        emotion: Label émotion 26

    Returns:
        Label émotion core (8)
    """
    # Mapping direct si existe
    if emotion in EMOTION_MAPPING_26_TO_8:
        mapped = EMOTION_MAPPING_26_TO_8[emotion]
        # Si le mapped n'est pas core, re-mapper
        if mapped not in ["anger", "joy", "sadness", "fear", "disgust", "surprise", "neutral", "frustration"]:
            return map_emotion_to_core(mapped)
        return mapped

    # Par défaut → neutral
    return "neutral"


@dataclass
class Example:
    """Exemple de prédiction pour évaluation"""

    scenario: str
    turn_idx: int
    text: str
    gold: str
    pred: str
    latency_ms: float
    confidence: float = 0.0
    abstention: bool = False
    margin: float = 0.0


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
    idx = {label: i for i, label in enumerate(labels)}

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

    per_label_f1 = {label: round(f1_for(label), 4) for label in labels}
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


def check_data_overlap(bootstrap_texts, eval_texts):
    """Vérifie qu'il n'y a pas de fuite entre bootstrap et évaluation.

    Args:
        bootstrap_texts: Liste de textes utilisés pour bootstrap
        eval_texts: Liste de textes utilisés pour évaluation

    Returns:
        Nombre d'exemples en overlap
    """
    # Hashes pour comparaison
    bootstrap_hashes = {
        hashlib.md5(text.strip().lower().encode()).hexdigest() for text in bootstrap_texts if text.strip()
    }
    eval_hashes = {hashlib.md5(text.strip().lower().encode()).hexdigest() for text in eval_texts if text.strip()}

    overlap = bootstrap_hashes & eval_hashes

    print("\n🔍 DATA OVERLAP CHECK:")
    print(f"   • Bootstrap examples: {len(bootstrap_hashes)}")
    print(f"   • Eval examples: {len(eval_hashes)}")
    print(f"   • Overlap: {len(overlap)}")

    if len(overlap) > 0:
        print(f"   ⚠️  WARNING: {len(overlap)} exemples en commun (FUITE DE DONNÉES)")
    else:
        print("   ✅ Aucune fuite détectée")

    return len(overlap)


def benchmark_encoder_latency(detector, n_samples=100):
    """Benchmark latence encodeur sur textes uniques (sans cache).

    Args:
        detector: HybridEmotionDetector
        n_samples: Nombre d'échantillons à tester

    Returns:
        Dict avec p50, p95, avg en ms
    """
    if not detector.use_proto:
        print("⚠️  ProtoClassifier désactivé, skip benchmark encodeur")
        return {"encoder_p50_ms": 0, "encoder_p95_ms": 0, "encoder_avg_ms": 0}

    print(f"\n🔬 BENCHMARK LATENCE ENCODEUR ({n_samples} textes uniques)...")

    # Générer textes uniques (éviter cache)
    unique_texts = [f"Texte unique benchmark latence numéro {uuid.uuid4()}" for _ in range(n_samples)]

    # Clear cache
    detector.encoder.clear_cache()

    # Warm-up
    _ = detector.encoder.encode("warm-up text")

    # Mesure
    latencies = []
    for text in unique_texts:
        start = time.time()
        _ = detector.encoder.encode(text)
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

    latencies.sort()
    p50 = latencies[int(0.50 * len(latencies))]
    p95 = latencies[int(0.95 * len(latencies))]
    avg = sum(latencies) / len(latencies)

    results = {"encoder_p50_ms": round(p50, 2), "encoder_p95_ms": round(p95, 2), "encoder_avg_ms": round(avg, 2)}

    print(f"   • p50: {results['encoder_p50_ms']}ms")
    print(f"   • p95: {results['encoder_p95_ms']}ms")
    print(f"   • avg: {results['encoder_avg_ms']}ms")

    # Sanity check
    if results['encoder_p95_ms'] < 10:
        print("   ⚠️  WARNING: Latence suspecte < 10ms (cache?)")

    return results


def tune_global_thresholds_with_coverage_constraint(detector, texts, golds, target_coverage=0.82, tolerance=0.03):
    """Tune seuils globaux pour maximiser F1 avec contrainte coverage.

    NOUVELLE APPROCHE : Grid global au lieu de per-label pour éviter
    effondrement coverage.

    Args:
        detector: HybridEmotionDetector
        texts: Liste de textes de validation
        golds: Liste d'émotions attendues (core-8)
        target_coverage: Coverage cible (défaut 82%)
        tolerance: Tolérance +/- (défaut 3%)

    Returns:
        (temperature, best_min_conf, best_min_margin, achieved_coverage)
    """
    if not detector.use_proto:
        return 1.0, 0.18, 0.06, 0.0

    print("\n🎛️  TUNING GLOBAL THRESHOLDS + TEMPERATURE")
    print(f"   Target coverage: {target_coverage:.1%} ± {tolerance:.1%}")

    # Mapper et filtrer golds vers core-8 AVANT calibration
    CORE8 = ["anger", "joy", "sadness", "fear", "disgust", "surprise", "neutral", "frustration"]
    golds_core = [map_emotion_to_core(g) if g not in CORE8 else g for g in golds]

    # Filtrer exemples invalides
    valid_idx = [i for i, g in enumerate(golds_core) if g in CORE8]
    if len(valid_idx) < len(golds_core):
        print(f"   ⚠️  WARNING: {len(golds_core) - len(valid_idx)} invalid labels filtered")
        texts = [texts[i] for i in valid_idx]
        golds_core = [golds_core[i] for i in valid_idx]
    else:
        print(f"   ✅ All {len(golds_core)} labels are valid core-8")

    golds = golds_core

    # 1. Calibrer température
    print("   Step 1/3: Calibrating temperature (NLL minimization)...")
    embeddings = np.array([detector.encoder.encode(text).flatten() for text in texts])
    optimal_temp = detector.classifier.calibrate_temperature_nll(embeddings, golds)
    detector.classifier.set_temperature(optimal_temp)

    # 2. Grid search global avec contrainte coverage
    print("   Step 2/3: Global grid search (macro-F1 max + coverage constraint)...")

    # Grilles baseline pour coverage optimal
    conf_values = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22]
    margin_values = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]

    best_score = -float('inf')
    best_conf = 0.16
    best_margin = 0.06
    best_f1 = 0.0
    best_cov = 0.0

    lambda_penalty = 0.5  # Baseline permissif pour coverage optimal

    print(
        f"      Testing {len(conf_values)} × {len(margin_values)} = {len(conf_values) * len(margin_values)} configs..."
    )

    for min_conf in conf_values:
        for min_margin in margin_values:
            # Appliquer seuils globaux uniformément
            detector.classifier.min_confidence = min_conf
            detector.classifier.min_margin = min_margin

            # Prédire sur tous les textes
            preds = []
            abstentions = 0

            for i, text in enumerate(texts):
                emb = embeddings[i]
                res = detector.classifier.predict(emb, text=text)

                if getattr(res, "abstention", False):
                    preds.append(None)  # Abstention = erreur pour recall
                    abstentions += 1
                else:
                    preds.append(res.primary)

            # Calculer coverage
            coverage = 1 - (abstentions / len(texts)) if len(texts) > 0 else 0.0

            # Calculer macro-F1 (abstentions = erreurs)
            label_f1s = []
            for label in CORE8:
                tp = sum(1 for i, p in enumerate(preds) if p == label and golds[i] == label)
                fp = sum(1 for i, p in enumerate(preds) if p == label and golds[i] != label)
                fn = sum(1 for i, g in enumerate(golds) if g == label and preds[i] != label)

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                label_f1s.append(f1)

            macro_f1 = np.mean(label_f1s) if label_f1s else 0.0

            # Score avec pénalité coverage ASYMÉTRIQUE (pénalise seulement si trop bas)
            penalty_coverage = max(0.0, target_coverage - coverage)
            score = macro_f1 - lambda_penalty * penalty_coverage

            # Mise à jour meilleur
            if score > best_score:
                best_score = score
                best_conf = min_conf
                best_margin = min_margin
                best_f1 = macro_f1
                best_cov = coverage

    print("\n   📊 BEST GLOBAL THRESHOLDS:")
    print(f"      • min_confidence: {best_conf:.2f}")
    print(f"      • min_margin: {best_margin:.2f}")
    print(f"      • macro-F1: {best_f1:.3f}")
    print(f"      • coverage: {best_cov:.2%}")
    print(f"      • score: {best_score:.3f}")

    # Appliquer meilleurs seuils globaux
    detector.classifier.min_confidence = best_conf
    detector.classifier.min_margin = best_margin

    # 3. Boucle de relaxation améliorée si coverage hors cible
    print("   Step 3/3: Coverage relaxation loop (if needed)...")

    def measure_coverage_fast_from_embeddings(embs, texts, detector):
        """Mesure rapide de coverage avec embeddings pré-calculés."""
        abst = 0
        for emb, t in zip(embs, texts):
            r = detector.classifier.predict(emb, text=t)
            if getattr(r, "abstention", False):
                abst += 1
        return 1 - abst / len(texts) if len(texts) > 0 else 0.0

    max_iters = 15  # Augmenté de 10 → 15
    step_conf = 0.04  # Augmenté de 0.03 → 0.04
    step_margin = 0.02

    coverage = best_cov
    iters = 0

    print(f"      Initial coverage: {coverage:.2%}")

    while (coverage < target_coverage - tolerance or coverage > target_coverage + tolerance) and iters < max_iters:
        if coverage < target_coverage - tolerance:
            # Trop d'abstentions → RELÂCHER
            best_conf = max(0.08, best_conf - step_conf)
            best_margin = max(0.00, best_margin - step_margin)
            direction = "relâché"
        elif coverage > target_coverage + tolerance:
            # Trop de couverture → DURCIR
            best_conf = min(0.40, best_conf + step_conf)
            best_margin = min(0.20, best_margin + step_margin)
            direction = "durci"
        else:
            # Dans la cible, arrêter
            break

        detector.classifier.min_confidence = best_conf
        detector.classifier.min_margin = best_margin
        coverage = measure_coverage_fast_from_embeddings(embeddings, texts, detector)
        iters += 1

        print(f"      Iter {iters}: coverage={coverage:.2%} ({direction} by {step_conf:.2f})")

        # Early stop si dans la cible
        if target_coverage - tolerance <= coverage <= target_coverage + tolerance:
            break

    print(f"   ✅ Final coverage: {coverage:.2%} after {iters} iterations")
    print(f"   ✅ Final thresholds: conf={best_conf:.2f}, margin={best_margin:.2f}")

    return optimal_temp, best_conf, best_margin, coverage


def evaluate_loso(
    scenarios,
    base_dir,
    temperature=0.05,
    min_confidence=0.30,
    min_margin=0.12,
    k_prototypes=1,
    learn_head=False,
    no_abstention=False,
):
    """Leave-One-Scenario-Out cross-validation.

    Args:
        scenarios: Liste de (nom, data) scénarios
        base_dir: Répertoire de base
        temperature: Température softmax
        min_confidence: Seuil confiance minimum
        min_margin: Marge minimum

    Returns:
        metrics, all_examples
    """
    # === AJUSTEMENT 1 : ORDRE CANONIQUE GLOBAL ===
    logging.info(f"\n{'=' * 60}")
    logging.info(f"🎯 CORE_EMOTIONS (canonical order): {CORE_EMOTIONS}")
    logging.info(f"{'=' * 60}\n")

    # === DEBUG PARAMS EVALUATE_LOSO ===
    logging.info("\n" + "=" * 60)
    logging.info("🔍 DEBUG EVALUATE_LOSO - PARAMS RECEIVED")
    logging.info("=" * 60)
    logging.info(f"  learn_head      : {learn_head}")
    logging.info(f"  no_abstention   : {no_abstention}")
    logging.info(f"  k_prototypes    : {k_prototypes}")
    logging.info("  learn           : not used in this context")
    logging.info("=" * 60 + "\n")
    print("\n" + "=" * 80)
    print("🔄 LOSO CROSS-VALIDATION (Leave-One-Scenario-Out)")
    print("=" * 80)
    print("Mode: Validation robuste sans fuite")
    print(f"Hyperparams: temp={temperature}, min_conf={min_confidence}, min_margin={min_margin}, k={k_prototypes}")

    all_fold_examples = []
    alignment_logged_once = False  # Flag local pour réduire les logs [ALIGN]

    for i, (held_out_name, held_out_data) in enumerate(scenarios):
        print(f"\n📊 Fold {i + 1}/{len(scenarios)} - Held out: {held_out_name}")

        # Train scenarios (tous sauf held_out)
        train_scenarios = [(name, data) for j, (name, data) in enumerate(scenarios) if j != i]

        # Créer detector pour ce fold (sans bootstrap auto)
        detector = HybridEmotionDetector(
            use_proto=True,
            bootstrap=False,  # Bootstrap manuel ci-dessous
            temperature=temperature,
            min_confidence=min_confidence,
            min_margin=min_margin,
            k_prototypes=k_prototypes,
        )

        # Bootstrap MANUEL avec train_scenarios uniquement
        if detector.use_proto:
            labeled_data = {}

            for name, data in train_scenarios:
                for turn_idx, text, emotion in iterate_user_turns(name, data):
                    if emotion and text.strip():
                        if emotion not in labeled_data:
                            labeled_data[emotion] = []

                        embedding = detector.encoder.encode(text)
                        labeled_data[emotion].append(embedding.flatten())

            # Convertir en arrays
            labeled_data = {
                emotion: np.vstack(embeddings) for emotion, embeddings in labeled_data.items() if len(embeddings) > 0
            }

            if labeled_data:
                detector.classifier.bootstrap(labeled_data)
                print(
                    f"   ✅ Bootstrapped: {sum(len(emb) for emb in labeled_data.values())} exemples, "
                    f"{len(labeled_data)} émotions"
                )

            # Collect train data pour tuning et linear head
            train_texts = []
            train_golds = []
            for name, data in train_scenarios[: int(len(train_scenarios) * 0.8)]:  # 80% train
                for turn_idx, text, emotion in iterate_user_turns(name, data):
                    if emotion and text.strip():
                        train_texts.append(text)
                        # ✅ Mapper vers core-8 dès la collecte
                        train_golds.append(map_emotion_to_core(emotion))

            # === PHASE 2 : LINEAR HEAD + CALIBRATION (FIXES COMPLETS) ===
            linear_head = None
            temperature = 1.0
            embeddings_cache = {}
            scaler_fold = None  # Scaler pour ce fold

            if learn_head and len(train_texts) >= 10:
                logging.info(f"\n{'=' * 60}")
                logging.info("🚀 Phase 2 Sprint 1: Training linear head (with all fixes)")
                logging.info(f"{'=' * 60}")

                # === FIX 1 : COMBINER FOLD + SEEDS AVEC DÉDUP ===

                # Encoder labels du fold avec mapping fixe
                train_golds_encoded = [EMOTION_TO_IDX[label] for label in train_golds]

                # Déduplication smart : éviter textes identiques fold/seeds
                train_texts_set = set(train_texts)
                seed_texts_unique = []
                seed_labels_unique = []

                for st, sl_encoded in zip(BOOTSTRAP_SEEDS_TEXTS, BOOTSTRAP_SEEDS_LABELS_ENCODED):
                    if st not in train_texts_set:  # Garde seulement seeds uniques
                        seed_texts_unique.append(st)
                        seed_labels_unique.append(sl_encoded)

                logging.info(f"   Deduplication: {len(BOOTSTRAP_SEEDS_TEXTS)} seeds → {len(seed_texts_unique)} unique")

                # Combiner fold + seeds
                train_texts_full = train_texts + seed_texts_unique
                train_golds_encoded_full = train_golds_encoded + seed_labels_unique

                logging.info(
                    f"   Total train: {len(train_texts)} fold + {len(seed_texts_unique)} seeds = {len(train_texts_full)}"
                )

                # === AJUSTEMENT 4 : VÉRIFICATION COUVERTURE CLASSES POST-DÉDUP ===
                unique_classes_present = sorted(set(train_golds_encoded_full))
                missing_classes = set(range(len(CORE_EMOTIONS))) - set(unique_classes_present)

                if missing_classes:
                    logging.warning(f"⚠️  Classes missing after dedup: {[CORE_EMOTIONS[i] for i in missing_classes]}")

                    # Backup : Ajouter 1 seed par classe manquante
                    for missing_idx in missing_classes:
                        # Trouver un seed pour cette classe dans les seeds originaux
                        backup_found = False
                        for st, sl in zip(BOOTSTRAP_SEEDS_TEXTS, BOOTSTRAP_SEEDS_LABELS_ENCODED):
                            if sl == missing_idx:
                                train_texts_full.append(st)
                                train_golds_encoded_full.append(sl)
                                logging.info(f"   ✅ Added backup seed for {CORE_EMOTIONS[missing_idx]}")
                                backup_found = True
                                break

                        if not backup_found:
                            logging.error(f"   ❌ No backup seed found for {CORE_EMOTIONS[missing_idx]}")

                # Log classes présentes
                final_classes = sorted(set(train_golds_encoded_full))
                final_emotions = [CORE_EMOTIONS[i] for i in final_classes]
                logging.info(f"   Classes in train: {final_emotions} ({len(final_classes)}/8)")

                if len(final_classes) < 8:
                    logging.warning(f"   ⚠️  Only {len(final_classes)}/8 classes present!")

                # === OPT1 : SAMPLE WEIGHTS (Fold 2x > Seeds 1x) ===
                sample_weights = np.array(
                    [2.0] * len(train_texts)  # Fold data = 2x weight
                    + [1.0] * len(seed_texts_unique)  # Seeds = 1x weight
                )

                # Si backup seeds ajoutés, ajouter leurs weights
                if missing_classes:
                    sample_weights = np.concatenate([sample_weights, np.ones(len(missing_classes))])

                logging.info("   Sample weights: fold 2.0x, seeds 1.0x")

                # ========================================
                # === FIX 1 : VAL FOLD-ONLY (CRITIQUE) ===
                # ========================================

                from sklearn.model_selection import train_test_split

                logging.info(f"\n{'=' * 60}")
                logging.info("🔧 FIX 1 : Val fold-only (seeds EXCLUDED from val)")
                logging.info(f"{'=' * 60}")

                # ÉTAPE 1 : Split FOLD SEULEMENT en train/val (80/20)
                # Val doit être PURE fold (représentatif du test LOSO)
                train_texts_fold_only, val_texts, train_golds_fold_only, val_golds = train_test_split(
                    train_texts,  # ← Fold SEULEMENT (PAS seeds)
                    train_golds_encoded,  # ← Labels fold
                    test_size=0.2,
                    random_state=42,
                    stratify=train_golds_encoded,
                )

                logging.info(
                    f"   Step 1/3: Fold split → {len(train_texts_fold_only)} train, {len(val_texts)} val (fold-only)"
                )

                # ========================================
                # === ADJ1 : OVERSAMPLE FRUSTRATION 100% + FIX CRITIQUES ===
                # ========================================

                logging.info("\n   🎯 ADJ1 : Oversample frustration 100% (fix F1=0)")

                # Séparer seeds par classe
                seed_texts_sampled = []
                seed_labels_sampled = []

                if len(seed_texts_unique) > 0:
                    # Groupe par classe
                    from collections import defaultdict

                    seeds_by_class = defaultdict(list)

                    for st, sl in zip(seed_texts_unique, seed_labels_unique):
                        seeds_by_class[sl].append((st, sl))

                    # Sampling différencié par classe
                    from sklearn.model_selection import train_test_split

                    # FIX CRITIQUE 3 : Assurer fold_idx existe
                    # fold_idx devrait venir de la boucle LOSO parente
                    # Si pas défini, utiliser hash du fold pour variance
                    try:
                        current_fold_seed = 42 + i  # i is the fold index from LOSO loop
                    except NameError:
                        # Fallback : utiliser hash des train_texts pour variance
                        import hashlib

                        fold_hash = int(hashlib.md5(''.join(train_texts_fold_only[:5]).encode()).hexdigest(), 16)
                        current_fold_seed = 42 + (fold_hash % 1000)
                        logging.warning(f"   ⚠️  fold_idx undefined, using fold_hash seed: {current_fold_seed}")

                    for class_idx, class_seeds in seeds_by_class.items():
                        class_name = CORE_EMOTIONS[class_idx]
                        texts = [s[0] for s in class_seeds]
                        labels = [s[1] for s in class_seeds]

                        # FRUSTRATION : 75% (équilibré pour limiter biais)
                        if class_name == 'frustration':
                            sampling_rate = 0.75
                            # Stratify safety
                            if len(texts) >= 2:
                                sampled_texts, _, sampled_labels, _ = train_test_split(
                                    texts,
                                    labels,
                                    test_size=1 - sampling_rate,
                                    random_state=current_fold_seed,
                                    stratify=labels,
                                )
                            else:
                                # Pas assez pour stratify
                                sampled_texts = texts
                                sampled_labels = labels
                            logging.info(
                                f"      {class_name:12s}: {len(sampled_texts)}/{len(texts)} seeds (75% - balanced)"
                            )

                        # SURPRISE : 75% (classe rare aussi)
                        elif class_name == 'surprise':
                            sampling_rate = 0.75
                            # FIX CRITIQUE 2 : Stratify safety
                            if len(texts) >= 2:
                                sampled_texts, _, sampled_labels, _ = train_test_split(
                                    texts,
                                    labels,
                                    test_size=1 - sampling_rate,
                                    random_state=current_fold_seed,
                                    stratify=labels,
                                )
                            else:
                                # Pas assez d'exemples pour stratify
                                sampled_texts = texts
                                sampled_labels = labels
                            logging.info(f"      {class_name:12s}: {len(sampled_texts)}/{len(texts)} seeds (75%)")

                        # AUTRES : 50% (standard)
                        else:
                            sampling_rate = 0.50
                            # FIX CRITIQUE 2 : Stratify safety
                            if len(texts) >= 2:
                                sampled_texts, _, sampled_labels, _ = train_test_split(
                                    texts,
                                    labels,
                                    test_size=1 - sampling_rate,
                                    random_state=current_fold_seed,
                                    stratify=labels,
                                )
                            else:
                                # Pas assez d'exemples pour stratify
                                sampled_texts = texts
                                sampled_labels = labels
                            logging.info(f"      {class_name:12s}: {len(sampled_texts)}/{len(texts)} seeds (50%)")

                        seed_texts_sampled.extend(sampled_texts)
                        seed_labels_sampled.extend(sampled_labels)

                    logging.info(f"\n   Total sampled: {len(seed_texts_sampled)}/{len(seed_texts_unique)} seeds")
                else:
                    seed_texts_sampled = []
                    seed_labels_sampled = []

                # ÉTAPE 2 : Ajouter seeds sampled au train fold 80%
                train_texts_split = list(train_texts_fold_only) + list(seed_texts_sampled)
                train_golds_split = list(train_golds_fold_only) + list(seed_labels_sampled)

                logging.info(
                    f"   Step 2/3: Adding seeds → {len(train_texts_fold_only)} fold + {len(seed_texts_sampled)} seeds"
                )

                # ÉTAPE 3 : Backup seeds pour classes manquantes (si nécessaire)
                # Vérifier couverture après sampling
                present_classes_after_sampling = set(train_golds_split)
                all_classes = set(range(len(CORE_EMOTIONS)))
                missing_classes_after_sampling = all_classes - present_classes_after_sampling

                if missing_classes_after_sampling:
                    logging.info("   Step 3/3: Adding backup seeds for missing classes after sampling...")
                    for missing_idx in missing_classes_after_sampling:
                        backup_found = False
                        for st, sl in zip(BOOTSTRAP_SEEDS_TEXTS, BOOTSTRAP_SEEDS_LABELS_ENCODED):
                            if sl == missing_idx:
                                train_texts_split.append(st)
                                train_golds_split.append(sl)
                                logging.info(f"      ✅ Backup seed added: {CORE_EMOTIONS[missing_idx]}")
                                backup_found = True
                                break

                        if not backup_found:
                            logging.error(f"      ❌ No backup seed for: {CORE_EMOTIONS[missing_idx]}")
                else:
                    logging.info("   Step 3/3: No backup needed (all classes present after sampling)")

                # Convertir en numpy arrays
                train_golds_split = np.array(train_golds_split, dtype=np.int32)
                val_golds = np.array(val_golds, dtype=np.int32)

                # Summary
                logging.info("\n   📊 Final split summary:")
                logging.info(
                    f"      Train: {len(train_texts_split)} ({len(train_texts_fold_only)} fold + {len(train_texts_split) - len(train_texts_fold_only)} seeds)"
                )
                logging.info(f"      Val:   {len(val_texts)} (fold-only, NO seeds)")
                logging.info(f"{'=' * 60}\n")

                # NOTE: Pas de sample_weights pour l'instant (simplification team)
                # Si overfitting persiste, on pourra réactiver fold 2x / seeds 1x

                # === PRECOMPUTE EMBEDDINGS AVEC CACHE ===
                logging.info("   Encoding embeddings...")

                def get_embedding(text):
                    if text not in embeddings_cache:
                        embeddings_cache[text] = detector.encoder.encode(text).flatten()
                    return embeddings_cache[text]

                X_train = np.array([get_embedding(t) for t in train_texts_split])
                X_val = np.array([get_embedding(t) for t in val_texts])

                # ========================================
                # === OPT1 : L2 NORMALIZATION (IMPORTANT) ===
                # ========================================

                from sklearn.preprocessing import Normalizer

                logging.info(f"\n{'=' * 60}")
                logging.info("⚡ OPT1 : L2 normalization (better for SBERT cosine space)")
                logging.info(f"{'=' * 60}")

                scaler = Normalizer(norm='l2')  # Unit norm for cosine similarity
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                scaler_fold = scaler

                logging.info(
                    f"   ✅ Embeddings L2-normalized → unit vectors (train: {X_train.shape}, val: {X_val.shape})"
                )
                logging.info(f"{'=' * 60}\n")

                # ========================================
                # === FIX 2 : LINEARSVC + PLATT (CRITIQUE) ===
                # ========================================

                import warnings

                from sklearn.calibration import CalibratedClassifierCV
                from sklearn.metrics import f1_score
                from sklearn.svm import LinearSVC

                logging.info(f"\n{'=' * 60}")
                logging.info("🔧 FIX 2 : LinearSVC + Platt calibration")
                logging.info(f"{'=' * 60}")

                best_C = 1.0
                best_f1_val = 0.0
                best_model = None
                best_model_type = "unknown"

                # === ADJ2 : GRID C EXTRA ÉLARGI ===
                C_values = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
                logging.info(f"   🎯 ADJ2 : Grid C extra-wide [{C_values[0]}...{C_values[-1]}]")

                logging.info(f"   Testing LinearSVC + Platt (grid: {C_values})...")

                for C in C_values:
                    try:
                        # Base LinearSVC (frontières nettes)
                        base_svc = LinearSVC(
                            C=C,
                            class_weight='balanced',
                            max_iter=5000,  # Plus d'itérations pour convergence
                            random_state=42,
                            dual=True,  # Fixed: dual must be bool, not 'auto'
                            loss='squared_hinge',
                        )

                        # Calibration Platt (3-fold interne)
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)

                            model_temp = CalibratedClassifierCV(
                                base_svc,
                                method='sigmoid',  # Platt scaling
                                cv=2,  # Réduire pour vitesse (de 3 à 2)
                                n_jobs=1,
                            )

                        # Garde-fou : Vérifier que toutes les classes sont valides (indices 0-7)
                        unique_train_indices = set(np.unique(train_golds_split))
                        valid_indices = set(range(len(CORE_EMOTIONS)))
                        if not unique_train_indices.issubset(valid_indices):
                            invalid = unique_train_indices - valid_indices
                            logging.error(f"⚠️  CRITICAL: Invalid class indices in y_train: {invalid}")
                            raise ValueError(f"Training indices contain invalid classes: {invalid}")

                        # Log classes présentes par leur nom
                        classes_present = [CORE_EMOTIONS[i] for i in sorted(unique_train_indices)]
                        logging.debug(f"Training classes: {classes_present}")

                        # Fit (sans sample_weight pour simplifier - team recommendation)
                        model_temp.fit(X_train, train_golds_split)

                        # Évaluer sur val (fold-only)
                        y_val_pred = model_temp.predict(X_val)

                        # Safeguard: Convertir en labels string si nécessaire
                        val_golds_labels = val_golds
                        y_val_pred_labels = y_val_pred
                        if np.issubdtype(np.array(val_golds_labels).dtype, np.integer):
                            val_golds_labels = [CORE_EMOTIONS[i] for i in val_golds_labels]
                        if np.issubdtype(np.array(y_val_pred_labels).dtype, np.integer):
                            y_val_pred_labels = [CORE_EMOTIONS[i] for i in y_val_pred_labels]

                        f1_val = f1_score(
                            val_golds_labels, y_val_pred_labels, labels=CORE_EMOTIONS, average='macro', zero_division=0
                        )

                        logging.info(f"      SVC C={C:5.2f} → val F1={f1_val:.4f}")

                        if f1_val > best_f1_val:
                            best_f1_val = f1_val
                            best_C = C
                            best_model = model_temp
                            best_model_type = "LinearSVC+Platt"

                    except Exception as e:
                        logging.warning(f"      SVC C={C:5.2f} → FAILED: {str(e)[:50]}")
                        continue

                # Fallback LogisticRegression si tous SVC échouent
                if best_model is None:
                    logging.warning("\n   ⚠️  All LinearSVC failed! Fallback to LogisticRegression...")

                    from sklearn.linear_model import LogisticRegression

                    for C in [0.5, 1.0, 2.0, 4.0]:
                        try:
                            model_temp = LogisticRegression(
                                C=C,
                                class_weight='balanced',
                                max_iter=1000,
                                random_state=42,
                                multi_class='multinomial',
                                solver='lbfgs',
                            )

                            # Garde-fou : Vérifier que toutes les classes sont valides (indices 0-7)
                            unique_train_indices = set(np.unique(train_golds_split))
                            valid_indices = set(range(len(CORE_EMOTIONS)))
                            if not unique_train_indices.issubset(valid_indices):
                                invalid = unique_train_indices - valid_indices
                                logging.error(f"⚠️  CRITICAL: Invalid class indices in y_train: {invalid}")
                                raise ValueError(f"Training indices contain invalid classes: {invalid}")

                            # Log classes présentes par leur nom
                            classes_present = [CORE_EMOTIONS[i] for i in sorted(unique_train_indices)]
                            logging.debug(f"Training classes: {classes_present}")

                            model_temp.fit(X_train, train_golds_split)
                            y_val_pred = model_temp.predict(X_val)

                            # Safeguard: Convertir en labels string si nécessaire
                            val_golds_labels = val_golds
                            y_val_pred_labels = y_val_pred
                            if np.issubdtype(np.array(val_golds_labels).dtype, np.integer):
                                val_golds_labels = [CORE_EMOTIONS[i] for i in val_golds_labels]
                            if np.issubdtype(np.array(y_val_pred_labels).dtype, np.integer):
                                y_val_pred_labels = [CORE_EMOTIONS[i] for i in y_val_pred_labels]

                            f1_val = f1_score(
                                val_golds_labels,
                                y_val_pred_labels,
                                labels=CORE_EMOTIONS,
                                average='macro',
                                zero_division=0,
                            )

                            logging.info(f"      LR  C={C:5.2f} → val F1={f1_val:.4f}")

                            if f1_val > best_f1_val:
                                best_f1_val = f1_val
                                best_C = C
                                best_model = model_temp
                                best_model_type = "LogisticRegression"

                        except Exception as e:
                            logging.warning(f"      LR  C={C:5.2f} → FAILED: {str(e)[:50]}")
                            continue

                logging.info(f"\n   ✅ Best model: {best_model_type}, C={best_C}, val F1={best_f1_val:.4f}")
                logging.info(f"{'=' * 60}\n")

                if best_model is None:
                    raise RuntimeError("All models failed! Check data or hyperparams.")

                # ========================================
                # === CRÉER LINEAR HEAD COMPATIBLE (NO DOUBLE CALIBRATION) ===
                # ========================================

                from jeffrey.ml.heads import LinearHead

                n_classes = len(CORE_EMOTIONS)
                linear_head = LinearHead(n_classes=n_classes, in_dim=384)

                # For CalibratedClassifierCV, we cannot set linear_head.model directly
                # Instead, we'll store the best_model separately and use predict_proba
                # No sample_weight scoring for calibrated models (would need weights_train)
                if best_model_type == "LinearSVC+Platt":
                    logging.info(
                        f"   ✅ Created LinearHead with {best_model_type} (pre-calibrated, no temperature needed)"
                    )
                    # Temperature is bypassed for CalibratedClassifierCV - already calibrated!
                    temperature = 1.0  # Identity temperature (no effect)
                    # CORRECTION 2: Set linear_head.model for consistency
                    linear_head.model = best_model
                    linear_head.classes_ = np.array(CORE_EMOTIONS)
                else:
                    # Only LogisticRegression can be assigned to linear_head.model
                    linear_head.model = best_model
                    linear_head.classes_ = np.array(CORE_EMOTIONS)  # Convert to numpy array
                    train_score = best_model.score(X_train, train_golds_split)
                    logging.info(f"   Train accuracy: {train_score:.3f}")

                    # === FIX CRITIQUE 1 : PAS DE DOUBLE CALIBRATION ===
                    # CalibratedClassifierCV fait déjà la calibration Platt
                    # On ne calibre PAS une 2ème fois avec temperature scaling

                    logging.info("\n   ⚠️  FIX CRITIQUE 1 : Skipping temperature calibration")
                    logging.info("   CalibratedClassifierCV already provides calibrated probas via Platt scaling")

                    # Temperature = 1.0 (identité, pas de scaling)
                    temperature = 1.0

                # ========================================
                # === OPT3 : MONITORING F1 PAR CLASSE (DEBUG) ===
                # ========================================

                logging.info("\n   📊 Best model F1 per class on val (fold-only):")

                # === FIX CRITIQUE 1 : UTILISER SEULEMENT predict_proba (PAS temperature) ===
                # CORRECTION : Remapper validation probas → CORE_EMOTIONS
                val_probs_raw = best_model.predict_proba(X_val)
                val_probs = np.zeros((len(X_val), len(CORE_EMOTIONS)), dtype=float)

                model_classes = getattr(best_model, "classes_", None)
                if model_classes is not None:
                    for j, cls_label in enumerate(model_classes):
                        # Gérer classes numériques (indices) ET strings
                        if isinstance(cls_label, (int, np.integer)):
                            # Classe numérique (0-7) → utiliser directement comme index
                            if 0 <= cls_label < len(CORE_EMOTIONS):
                                val_probs[:, cls_label] = val_probs_raw[:, j]
                            else:
                                logging.warning(f"⚠️  Numeric class {cls_label} out of range in validation")
                        elif cls_label in EMOTION_TO_IDX:
                            # Classe string → convertir via EMOTION_TO_IDX
                            val_probs[:, EMOTION_TO_IDX[cls_label]] = val_probs_raw[:, j]
                        else:
                            logging.warning(f"⚠️  Unknown class '{cls_label}' in validation")
                else:
                    logging.warning("⚠️  model.classes_ absent in validation → uniform fallback")
                    val_probs[:] = 1.0 / len(CORE_EMOTIONS)

                # Prédictions alignées avec CORE_EMOTIONS
                val_preds_idx = np.argmax(val_probs, axis=1)
                val_preds_labels = [CORE_EMOTIONS[i] for i in val_preds_idx]

                # S'assurer que y_val sont des labels (pas des indices)
                arr = np.asarray(val_golds)
                if arr.dtype.kind in ("i", "u"):
                    y_val_labels = [CORE_EMOTIONS[i] for i in arr]
                else:
                    y_val_labels = list(arr)

                # Utiliser les indices pour les calculs compatibles
                val_preds = val_preds_idx

                for i, emotion in enumerate(CORE_EMOTIONS):
                    # Binariser pour cette classe
                    y_true_bin = (val_golds == i).astype(int)
                    y_pred_bin = (val_preds == i).astype(int)

                    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
                    status = "✅" if f1 > 0.30 else ("⚠️" if f1 > 0.15 else "❌")
                    logging.info(f"      {emotion:12s}: {f1:.3f} {status}")

                # Check classes critiques
                if (val_golds == EMOTION_TO_IDX['frustration']).sum() > 0:
                    f1_frust = f1_score(
                        (val_golds == EMOTION_TO_IDX['frustration']).astype(int),
                        (val_preds == EMOTION_TO_IDX['frustration']).astype(int),
                        zero_division=0,
                    )
                    if f1_frust == 0.0:
                        logging.warning("      ⚠️  Frustration F1 = 0.0 on val (check if present in val)")

                # Store model info for inference compatibility
                linear_head.best_model_type = best_model_type
                linear_head.best_model = best_model
                linear_head.temperature = temperature

                logging.info(f"\n{'=' * 60}")
                logging.info("✅ Linear head ready for inference")
                logging.info(f"{'=' * 60}\n")

            elif learn_head:
                logging.warning(f"⚠️  Phase 2 skipped: only {len(train_texts)} train samples (need ≥10)")

            # Tuner les seuils sur un split du train (20% pour tuning)
            if len(train_texts) > 30:  # Au moins 30 exemples pour tuning
                tuned_temp, tuned_conf, tuned_margin, tuned_cov = tune_global_thresholds_with_coverage_constraint(
                    detector, train_texts, train_golds, target_coverage=0.82
                )
                print(
                    f"   🎛️  Tuned: T={tuned_temp:.4f}, conf={tuned_conf:.2f}, "
                    f"margin={tuned_margin:.2f}, cov={tuned_cov:.2%}"
                )

            # Clear cache avant évaluation
            detector.encoder.clear_cache()

        # Évaluer sur held_out (SANS learning)
        fold_examples = []
        latencies = []

        fold_turn_idx = 0  # Counter for debug logging
        for turn_idx, text, gold in iterate_user_turns(held_out_name, held_out_data):
            if not gold or not text.strip():
                continue

            start_time = time.perf_counter()

            # === DEBUG INFERENCE PATH (log once) ===
            if fold_turn_idx == 0:
                if linear_head is not None:
                    logging.info("🎯 PHASE2: Using LINEAR HEAD path (decision_function + temperature)")
                else:
                    logging.info("🎯 PHASE1: Using PROTOTYPES path (baseline)")
            fold_turn_idx += 1

            # Get embedding (avec cache si disponible)
            if text in embeddings_cache:
                emb = embeddings_cache[text]
            else:
                emb = detector.encoder.encode(text).flatten()
                embeddings_cache[text] = emb

            # ========================================
            # === CORRECTION RÉGRESSION : SVC SEUL (NO BLEND) ===
            # ========================================
            # PROBLÈME : Blend avec proto causait régression (0.448 → 0.394)
            #            91× "Proto probs sum to 0" → fallback uniform pollue SVC
            # SOLUTION : SVC uniquement (LinearSVC + Platt calibration)

            if linear_head is not None:
                # Log correction (première fois uniquement)
                if fold_turn_idx == 1:
                    logging.info("🧪 CORRECTION: Using SVC only (blend disabled - fix regression 0.448→0.394)")

                # === STANDARDISER EMBEDDING ===
                if scaler_fold is not None:
                    emb_scaled = scaler_fold.transform(emb.reshape(1, -1))
                else:
                    logging.warning("⚠️  Scaler not found, using raw embedding")
                    emb_scaled = emb.reshape(1, -1)

                # === SVC PROBS CALIBRÉES (Platt) ===
                # CORRECTION CRITIQUE : Remapper predict_proba → CORE_EMOTIONS
                probs_raw = linear_head.model.predict_proba(emb_scaled)[0]
                probs = np.zeros(len(CORE_EMOTIONS), dtype=float)

                # Récupérer l'ordre des classes du modèle
                model_classes = getattr(linear_head.model, "classes_", None)

                if model_classes is None:
                    logging.error("⚠️  CRITICAL: model.classes_ absent → uniform fallback")
                    probs = np.ones(len(CORE_EMOTIONS)) / len(CORE_EMOTIONS)
                else:
                    # Gérer classes numériques (indices) ET strings
                    for i, cls_label in enumerate(model_classes):
                        # Conversion flexible: indices numériques → CORE_EMOTIONS index
                        if isinstance(cls_label, (int, np.integer)):
                            # Classe numérique (0-7) → utiliser directement comme index
                            if 0 <= cls_label < len(CORE_EMOTIONS):
                                probs[cls_label] = probs_raw[i]
                            else:
                                logging.warning(
                                    f"⚠️  Numeric class {cls_label} out of range [0-{len(CORE_EMOTIONS) - 1}]"
                                )
                        elif cls_label in EMOTION_TO_IDX:
                            # Classe string → convertir via EMOTION_TO_IDX
                            probs[EMOTION_TO_IDX[cls_label]] = probs_raw[i]
                        else:
                            logging.warning(f"⚠️  Unknown class '{cls_label}' from model")

                    # Log de debug (une seule fois pour tous les folds)
                    if not alignment_logged_once:
                        logging.info(f"🔍 [ALIGN] model.classes_ = {list(getattr(linear_head.model, 'classes_', []))}")
                        logging.info(f"🔍 [ALIGN] CORE_EMOTIONS = {CORE_EMOTIONS}")
                        logging.info(f"🔍 [ALIGN] sum(probs) = {probs.sum():.6f}")
                        logging.info(f"🔍 [ALIGN] argmax = {CORE_EMOTIONS[np.argmax(probs)]}")
                        alignment_logged_once = True

                # Assertion de sécurité
                assert np.isfinite(probs).all(), "Probs contain NaN or Inf!"

                # Normalisation sécurité
                s = probs.sum()
                probs = probs / s if s > 0 else np.ones(len(CORE_EMOTIONS)) / len(CORE_EMOTIONS)

                # === PRÉDICTION AVEC CORE_EMOTIONS ===
                pred_idx = np.argmax(probs)
                pred_label = CORE_EMOTIONS[pred_idx]
                confidence = float(probs[pred_idx])

                # === ABSTENTION PERMISSIVE ===
                abstain = False
                if not no_abstention:
                    sorted_probs = np.sort(probs)[::-1]
                    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]

                    # Seuils permissifs (cibler seulement très basse confiance)
                    min_conf_permissive = 0.20
                    min_margin_permissive = 0.05

                    if confidence < min_conf_permissive or margin < min_margin_permissive:
                        abstain = True

                    if fold_turn_idx == 1 and abstain:
                        logging.info(f"   Permissive abstention (conf={confidence:.3f}, margin={margin:.3f})")

                # Create result object
                class PredResult:
                    def __init__(self, label, confidence, abstention):
                        self.primary = label  # Compatibility avec existing code
                        self.label = label
                        self.confidence = float(confidence)
                        self.abstention = abstention

                res = PredResult(pred_label, confidence, abstain)

            # === PHASE 1 PATH : Use prototypes ===
            else:
                res = detector.detect(text)

            # Track latency
            dt = (time.perf_counter() - start_time) * 1000  # ms
            latencies.append(dt)

            # Mapper vers émotions core (8 labels)
            gold_core = map_emotion_to_core(gold) if gold else None
            pred_core = map_emotion_to_core(res.primary)

            fold_examples.append(
                Example(
                    held_out_name,
                    turn_idx,
                    text,
                    gold_core,  # ✅ Utiliser gold_core au lieu de gold
                    pred_core,  # ✅ Utiliser pred_core au lieu de pred
                    dt,
                    res.confidence if hasattr(res, 'confidence') else 0.5,
                    res.abstention if hasattr(res, 'abstention') else False,
                    res.margin if hasattr(res, 'margin') else 0.0,
                )
            )

        # Log latency statistics
        if latencies:
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            p99 = np.percentile(latencies, 99)
            logging.info(f"📊 Latency: p50={p50:.2f}ms, p95={p95:.2f}ms, p99={p99:.2f}ms")

            if p95 > 120:
                logging.warning(f"⚠️  p95 latency {p95:.2f}ms exceeds target 120ms")

        all_fold_examples.extend(fold_examples)

        # Métriques fold
        if fold_examples:
            fold_metrics = compute_metrics(fold_examples)
            print(f"   Fold F1: {fold_metrics['macro_f1']:.3f}, Acc: {fold_metrics['accuracy']:.3f}")

    # Logs de vérification structure
    num_folds = len(scenarios)
    logging.info(f"✅ [LOSO] Processed {num_folds} folds")
    logging.info(f"✅ [LOSO] Aggregated {len(all_fold_examples)} examples (expected ~200+)")

    # Comptage unique des utilisateurs testés
    unique_users = len({ex.held_out_name for ex in all_fold_examples})
    logging.info(f"✅ [LOSO] Unique users in results: {unique_users}")

    if len(all_fold_examples) < 100:
        logging.error(f"⚠️  [LOSO] BUG: Only {len(all_fold_examples)} examples!")

    if unique_users < 30:
        logging.warning(f"⚠️  [LOSO] Only {unique_users} unique users processed")

    # Agrégation finale
    print("\n" + "=" * 80)
    print("📊 LOSO RÉSULTATS AGRÉGÉS")
    print("=" * 80)

    final_metrics = compute_metrics(all_fold_examples)
    advanced = compute_advanced_metrics(all_fold_examples)
    final_metrics.update(advanced)

    print("\n🎯 MÉTRIQUES FINALES (LOSO):")
    print(f"   • Macro-F1: {final_metrics['macro_f1']:.3f}")
    print(f"   • Accuracy: {final_metrics['accuracy']:.3f}")
    print(f"   • ECE: {final_metrics['ece']:.4f}")
    print(f"   • Coverage: {final_metrics['coverage']:.2%}")

    # Affichage final standardisé (grep-friendly)
    print(f"\n🎯 Macro-F1 (LOSO): {final_metrics['macro_f1']:.3f}")
    print(f"   Accuracy (LOSO): {final_metrics['accuracy']:.3f}")
    print(f"   ECE (LOSO): {final_metrics.get('ece', 0.0):.3f}")

    # Export machine-lisible
    import json

    with open("loso_metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2, default=float)
    logging.info("✅ Metrics exported to loso_metrics.json")

    return final_metrics, all_fold_examples


def export_loso_results(results, output_dir="tests/results"):
    """Exporte résultats LOSO en JSON et CSV pour documentation.

    Args:
        results: Dict des résultats LOSO
        output_dir: Répertoire de sortie
    """
    import csv
    import json
    from datetime import datetime
    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Export JSON complet
    json_path = output_path / f"loso_results_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ JSON exporté : {json_path}")

    # 2. Export CSV agrégé par fold
    csv_path = output_path / f"loso_folds_{timestamp}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['fold', 'f1', 'accuracy', 'coverage', 'ece', 'latency_p95'])
        writer.writeheader()

        for fold in results.get('folds', []):
            writer.writerow(
                {
                    'fold': fold.get('name', 'unknown'),
                    'f1': fold.get('macro_f1', 0.0),
                    'accuracy': fold.get('accuracy', 0.0),
                    'coverage': fold.get('coverage', 0.0),
                    'ece': fold.get('ece', 0.0),
                    'latency_p95': fold.get('latency_p95', 0.0),
                }
            )
    print(f"✅ CSV folds exporté : {csv_path}")

    # 3. Export CSV métriques agrégées
    summary_path = output_path / f"loso_summary_{timestamp}.csv"
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])

        overall = results.get('overall', {})
        writer.writerow(['Macro-F1 Mean', f"{overall.get('macro_f1', 0):.4f}"])
        writer.writerow(['Macro-F1 Std', f"{overall.get('macro_f1_std', 0):.4f}"])
        writer.writerow(['Coverage Mean', f"{overall.get('coverage', 0):.4f}"])
        writer.writerow(['ECE Mean', f"{overall.get('ece', 0):.4f}"])
        writer.writerow(['Accuracy Mean', f"{overall.get('accuracy', 0):.4f}"])
        writer.writerow(['Latency P95 Mean', f"{overall.get('latency_p95', 0):.2f}ms"])
    print(f"✅ CSV summary exporté : {summary_path}")

    # 4. Export thresholds finaux
    thresholds_path = output_path / f"thresholds_{timestamp}.json"
    thresholds_data = {
        "hyperparams": results.get('hyperparams', {}),
        "timestamp": timestamp,
        "note": "Seuils optimaux trouvés par grid global avec contrainte coverage",
    }
    with open(thresholds_path, 'w', encoding='utf-8') as f:
        json.dump(thresholds_data, f, indent=2)
    print(f"✅ Thresholds exportés : {thresholds_path}")

    return {
        'json': str(json_path),
        'csv_folds': str(csv_path),
        'csv_summary': str(summary_path),
        'thresholds': str(thresholds_path),
    }


class HybridEmotionDetector:
    """Détecteur hybride avec fallback V3 lexical."""

    def __init__(
        self,
        use_proto: bool = True,
        bootstrap: bool = True,  # NOUVEAU
        temperature: float = 0.05,  # NOUVEAU (ajusté)
        min_confidence: float = 0.30,  # NOUVEAU (ajusté)
        min_margin: float = 0.12,  # NOUVEAU (ajusté)
        k_prototypes: int = 1,  # NOUVEAU - multi-prototypes
    ):
        """Initialise le détecteur hybride.

        Args:
            use_proto: Si True, utilise ProtoClassifier, sinon V3
            bootstrap: Si True, bootstrap avec données YAML (peut causer fuite)
            temperature: Température softmax (plus bas = plus confiant)
            min_confidence: Seuil minimum confiance pour abstention
            min_margin: Marge minimum top1-top2 pour abstention
            k_prototypes: Nombre de prototypes par label (1=single, 3=multi)
        """
        self.use_proto = use_proto
        self.bootstrap_texts = []  # Pour tracking overlap
        self.k_prototypes = k_prototypes

        # Charger V3 (fallback toujours disponible)
        self.v3_detector = EmotionDetectorV3()

        if self.use_proto:
            try:
                # Charger encoder et classifier
                self.encoder = create_default_encoder()

                # Définir les 8 émotions core pour restriction
                CORE8 = ["anger", "joy", "sadness", "fear", "disgust", "surprise", "neutral", "frustration"]

                self.classifier = ProtoClassifier(
                    dimension=384,
                    temperature=temperature,  # Utiliser param
                    min_confidence=min_confidence,  # Utiliser param
                    min_margin=min_margin,  # Utiliser param
                    allowed_labels=CORE8,  # ✅ Restriction aux 8 core
                    k_prototypes=k_prototypes,  # Support multi-prototypes
                )

                # Bootstrap conditionnel
                if bootstrap:
                    self._bootstrap_from_yaml()
                    logging.info("✅ ProtoClassifier enabled WITH bootstrap")
                else:
                    logging.info("✅ ProtoClassifier enabled WITHOUT bootstrap (cold-start)")

            except Exception as e:
                logging.error(f"❌ ProtoClassifier failed: {e}")
                self.use_proto = False

        # Feedback store
        self.feedback_store = FeedbackStore()

    def _bootstrap_from_yaml(self):
        """Bootstrap le classifier avec le seed séparé."""
        # Charger seed bootstrap (DIFFÉRENT des scénarios de test)
        seed_path = ROOT / "tests" / "data" / "bootstrap_seed.yaml"
        if not seed_path.exists():
            logging.warning(f"No bootstrap seed: {seed_path}")
            return

        import yaml

        with open(seed_path) as f:
            seed_data = yaml.safe_load(f)

        if not seed_data:
            logging.warning("Empty bootstrap seed")
            return

        # Grouper par émotion
        labeled_data = {}
        for item in seed_data:
            text = item['text']
            emotion = item['emotion']

            if emotion not in labeled_data:
                labeled_data[emotion] = []

            # Encoder
            embedding = self.encoder.encode(text)
            labeled_data[emotion].append(embedding.flatten())
            self.bootstrap_texts.append(text)

        # Convertir en arrays
        labeled_data = {
            emotion: np.vstack(embeddings) for emotion, embeddings in labeled_data.items() if len(embeddings) > 0
        }

        # Bootstrap
        if labeled_data:
            self.classifier.bootstrap(labeled_data)
            logging.info(
                f"Bootstrapped with {sum(len(emb) for emb in labeled_data.values())} examples "
                f"across {len(labeled_data)} emotions"
            )

        # Vérif couverture des 8 core labels
        core_labels = ["anger", "joy", "sadness", "fear", "disgust", "surprise", "neutral", "frustration"]
        missing = [label for label in core_labels if label not in labeled_data]
        if missing:
            logging.warning(f"Core emotions non initialisés: {missing}")

    def detect(self, text: str) -> EmotionPrediction:
        """Détecte l'émotion avec fallback automatique.

        Args:
            text: Texte à analyser

        Returns:
            EmotionPrediction
        """
        if not self.use_proto:
            # Fallback V3
            result_v3 = self.v3_detector.detect(text)
            return EmotionPrediction(
                primary=result_v3.primary,
                confidence=result_v3.intensity,
                all_scores={result_v3.primary: result_v3.intensity},
                margin=0.0,
                abstention=False,
                reason="v3_fallback",
            )

        try:
            # ProtoClassifier
            embedding = self.encoder.encode(text)
            prediction = self.classifier.predict(embedding.flatten(), text=text)

            # Si abstention haute, fallback V3
            if prediction.abstention and prediction.confidence < 0.3:
                logging.debug(f"Abstention → V3 fallback for: {text[:50]}")
                result_v3 = self.v3_detector.detect(text)
                prediction.primary = result_v3.primary
                prediction.confidence = result_v3.intensity
                prediction.reason = "abstention_v3_fallback"

            return prediction

        except Exception as e:
            logging.error(f"ProtoClassifier error, using V3: {e}")
            result_v3 = self.v3_detector.detect(text)
            return EmotionPrediction(
                primary=result_v3.primary,
                confidence=result_v3.intensity,
                all_scores={result_v3.primary: result_v3.intensity},
                margin=0.0,
                abstention=False,
                reason=f"error_v3_fallback: {e}",
            )


def compute_advanced_metrics(examples):
    """Calcule métriques avancées (ECE, coverage)."""
    # ECE (Expected Calibration Error)
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    confidences = [ex.confidence for ex in examples]
    corrects = [1 if ex.pred == ex.gold else 0 for ex in examples]

    ece = 0.0
    total_samples = len(examples)

    for i in range(n_bins):
        mask = [(conf >= bin_edges[i] and conf < bin_edges[i + 1]) for conf in confidences]
        bin_size = sum(mask)

        if bin_size > 0:
            bin_acc = sum(corrects[j] for j in range(len(mask)) if mask[j]) / bin_size
            bin_conf = sum(confidences[j] for j in range(len(mask)) if mask[j]) / bin_size
            ece += abs(bin_acc - bin_conf) * bin_size / total_samples

    # Coverage (taux abstention)
    abstentions = sum(1 for ex in examples if ex.abstention)
    coverage = 1 - (abstentions / len(examples)) if examples else 0.0

    return {"ece": round(ece, 4), "coverage": round(coverage, 4), "abstentions": abstentions}


def main():
    """Point d'entrée principal"""

    # Parse arguments
    parser = argparse.ArgumentParser(description='Évaluation Phase 1 Détection Émotionnelle - Validation Non Biaisée')
    parser.add_argument(
        '--no-bootstrap', action='store_true', help='Désactiver bootstrap avec données de test (éval cold-start propre)'
    )
    parser.add_argument('--no-learn', action='store_true', help='Désactiver apprentissage pendant évaluation')
    parser.add_argument(
        '--cv', choices=['none', 'loso'], default='none', help='Mode cross-validation (loso = Leave-One-Scenario-Out)'
    )
    parser.add_argument(
        '--temperature', type=float, default=0.05, help='Température softmax (défaut: 0.05, plus bas = plus confiant)'
    )
    parser.add_argument(
        '--min-confidence', type=float, default=0.30, help='Seuil minimum confiance pour abstention (défaut: 0.30)'
    )
    parser.add_argument(
        '--min-margin', type=float, default=0.12, help='Marge minimum top1-top2 pour abstention (défaut: 0.12)'
    )
    parser.add_argument(
        '--benchmark-latency', action='store_true', help='Benchmarker latence encodeur séparément (100 textes uniques)'
    )
    parser.add_argument(
        '--k-prototypes',
        type=int,
        default=1,
        choices=[1, 3],
        help='Nombre de prototypes par label (1=single, 3=multi-modal)',
    )
    parser.add_argument("--learn-head", action="store_true", help="Phase 2: Train linear head on frozen embeddings")
    parser.add_argument("--no-abstention", action="store_true", help="Disable abstention (100%% coverage for Sprint 1)")

    args = parser.parse_args()

    # === DEBUG FLAGS PHASE 2 ===
    print("\n" + "=" * 60)
    print("🔍 DEBUG FLAGS PHASE 2 SPRINT 1")
    print("=" * 60)
    print(f"  --learn-head     : {args.learn_head}")
    print(f"  --no-abstention  : {args.no_abstention}")
    print(f"  --cv             : {args.cv}")
    print(f"  --no-learn       : {args.no_learn}")
    print(f"  --k-prototypes   : {args.k_prototypes}")
    print("=" * 60 + "\n")

    # Seed pour reproductibilité
    np.random.seed(42)

    base = ROOT
    conv_dir = base / "tests" / "convos"

    if not conv_dir.exists():
        print("⚠️  tests/convos/ introuvable. Crée d'abord les scénarios.")
        sys.exit(1)

    # Initialiser le détecteur
    det = HybridEmotionDetector(
        use_proto=USE_PROTO_CLASSIFIER,
        bootstrap=not args.no_bootstrap,
        temperature=args.temperature,
        min_confidence=args.min_confidence,
        min_margin=args.min_margin,
        k_prototypes=args.k_prototypes,
    )

    print("\n⚙️  CONFIGURATION:")
    print(f"   • Bootstrap: {'Désactivé (cold-start)' if args.no_bootstrap else 'Activé'}")
    print(f"   • Learning: {'Désactivé' if args.no_learn else 'Activé'}")
    print(f"   • Temperature: {args.temperature}")
    print(f"   • Min confidence: {args.min_confidence}")
    print(f"   • Min margin: {args.min_margin}")
    print(f"   • K-prototypes: {args.k_prototypes}")

    scenarios = load_scenarios(conv_dir)

    # Benchmark latence si demandé
    bench_results = {}
    if args.benchmark_latency:
        bench_results = benchmark_encoder_latency(det, n_samples=100)

    # Si mode LOSO, brancher sur fonction dédiée
    if args.cv == 'loso':
        metrics, all_examples = evaluate_loso(
            scenarios,
            base,
            temperature=args.temperature,
            min_confidence=args.min_confidence,
            min_margin=args.min_margin,
            k_prototypes=args.k_prototypes,
            learn_head=args.learn_head,
            no_abstention=args.no_abstention,
        )

        # Sauvegarder résultats LOSO
        out_json = base / "test_results" / "sprint1_loso_results.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "mode": "loso",
            "scenarios_count": len(scenarios),
            "examples_count": len(all_examples),
            "metrics": metrics,
            "hyperparams": {
                "temperature": args.temperature,
                "min_confidence": args.min_confidence,
                "min_margin": args.min_margin,
                "k_prototypes": args.k_prototypes,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        with open(out_json, "w") as f:
            json.dump(report, f, indent=2)

        print(f"\n💾 RAPPORT LOSO SAUVEGARDÉ: {out_json}")

        # Export automatique pour documentation
        # Note: Nous créons un format simple car evaluate_loso ne retourne que les métriques agrégées
        loso_results = {
            "overall": {
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "macro_f1_std": 0.0,  # Non calculé dans cette version
                "coverage": float(metrics.get("coverage", 0.0)),
                "ece": float(metrics.get("ece", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "latency_p95": 0.0,  # Non calculé dans cette version
            },
            "hyperparams": {
                "temperature": args.temperature,
                "min_confidence": args.min_confidence,
                "min_margin": args.min_margin,
                "k_prototypes": args.k_prototypes,
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "folds": [],  # Placeholder - les détails par fold ne sont pas retournés par evaluate_loso
        }

        export_loso_results(loso_results)
        print("\n📁 Résultats exportés dans tests/results/")

        # Verdict LOSO
        print("\n" + "=" * 80)
        print("🎯 VERDICT LOSO (VALIDATION ROBUSTE)")
        print("=" * 80)
        macro_f1 = metrics.get("macro_f1", 0.0)
        ece = metrics.get("ece", 1.0)
        coverage = metrics.get("coverage", 0.0)

        f1_ok = macro_f1 >= 0.50
        ece_ok = ece <= 0.10
        cov_ok = 0.75 <= coverage <= 0.90

        print(f"   • Macro-F1: {macro_f1:.3f} {'✅' if f1_ok else '❌'} (target ≥ 0.50)")
        print(f"   • ECE: {ece:.4f} {'✅' if ece_ok else '❌'} (target ≤ 0.10)")
        print(f"   • Coverage: {coverage:.2%} {'✅' if cov_ok else '❌'} (target 75-90%)")
        print()

        if f1_ok and ece_ok and cov_ok:
            print("✅ OBJECTIFS PHASE 1 ATTEINTS (validation LOSO robuste)")
        else:
            print("⚠️  Ajustements nécessaires")

        print("=" * 80)

        return  # Exit early après LOSO

    all_examples = []
    latencies = []

    print("=" * 80)
    print("🧪 JEFFREY OS — Sprint 1 Emotion Eval Runner")
    print("=" * 80)
    print()
    print(f"📁 Dossier : {conv_dir}")
    print(f"📊 Scénarios détectés : {len(scenarios)}")
    print()

    # Check overlap bootstrap/eval si bootstrap activé
    if not args.no_bootstrap and det.use_proto and hasattr(det, 'bootstrap_texts'):
        eval_texts = []
        for name, data in scenarios:
            for turn_idx, text, gold in iterate_user_turns(name, data):
                if gold and text.strip():
                    eval_texts.append(text)

        if det.bootstrap_texts:
            overlap_count = check_data_overlap(det.bootstrap_texts, eval_texts)
            if overlap_count > 0:
                print(f"⚠️  WARNING: Fuite de données détectée ({overlap_count} exemples)")

    print("⏳ Traitement en cours...")
    print()

    # Traiter tous les scénarios
    for name, data in scenarios:
        for turn_idx, text, gold in iterate_user_turns(name, data):
            if not text.strip():
                continue

            # Détection avec mesure de latence
            t0 = time.perf_counter()
            if det.use_proto:
                # Réutiliser l'embedding (amélioration suggrée)
                embedding = det.encoder.encode(text).flatten()
                res = det.classifier.predict(embedding, text=text)
                pred = res.primary
            else:
                res = det.detect(text)
                pred = res.primary
                embedding = np.zeros(384, dtype=np.float32)  # Placeholder pour V3
            dt = (time.perf_counter() - t0) * 1000.0

            latencies.append(dt)

            # Mapper vers émotions core (8 labels)
            gold_core = map_emotion_to_core(gold) if gold else None
            pred_core = map_emotion_to_core(pred)

            # Stocker feedback (même si correct, pour stats)
            if det.use_proto:
                event = FeedbackEvent(
                    text=text,
                    embedding=embedding.astype(np.float32),
                    predicted_emotion=pred,  # Original 26
                    predicted_confidence=res.confidence,
                    corrected_emotion=gold if pred != gold else None,  # Original 26
                    user_confidence=1.0,  # Test = confiance maximale
                    timestamp=datetime.now(),
                    abstention=res.abstention,
                    margin=res.margin,
                )
                det.feedback_store.add_event(event)

                # Learning avec gold MAPPÉ (8 core)
                if not args.no_learn:
                    if pred_core != gold_core and gold_core:
                        det.classifier.learn(embedding, gold_core, confidence=1.0)
                        logging.debug(f"Learned: {text[:50]}... → {gold_core}")

            all_examples.append(
                Example(
                    name,
                    turn_idx,
                    text,
                    gold_core,  # ✅ Utiliser gold_core au lieu de gold
                    pred_core,  # ✅ Utiliser pred_core au lieu de pred
                    dt,
                    res.confidence if hasattr(res, 'confidence') else 0.5,
                    res.abstention if hasattr(res, 'abstention') else False,
                    res.margin if hasattr(res, 'margin') else 0.0,
                )
            )

    # Garder uniquement les exemples annotés pour le calcul F1
    annotated = [ex for ex in all_examples if ex.gold]

    print(f"✅ Traitement terminé : {len(all_examples)} tours, {len(annotated)} annotés")
    print()

    # Calculer les métriques
    metrics = compute_metrics(annotated)

    # Calculer métriques avancées Phase 1
    advanced_metrics = compute_advanced_metrics(annotated)
    metrics.update(advanced_metrics)

    # Ajouter bench_results si disponible
    if bench_results:
        metrics.update(bench_results)

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
            "max": round(max(latencies), 2) if latencies else 0.0,
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
                "text": ex.text[:200],
            }
            for ex in annotated[:25]  # Échantillon
        ],
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
    print("📈 MÉTRIQUES GLOBALES :")
    print(f"   • Tours annotés : {report['num_annotated']} / {report['num_turns']}")
    print(f"   • Macro-F1      : {metrics.get('macro_f1', 0.0):.3f}")
    print(f"   • Accuracy      : {metrics.get('accuracy', 0.0):.3f}")
    print()
    print("⚡ LATENCE :")
    print(f"   • Moyenne  : {avg}ms")
    print(f"   • P95      : {p95}ms")
    print(f"   • Min/Max  : {report['latency_ms']['min']}ms / {report['latency_ms']['max']}ms")
    print()

    # Métriques avancées Phase 1
    if "ece" in metrics:
        print("🎯 MÉTRIQUES AVANCÉES (PHASE 1) :")
        target_ece = "✅" if metrics["ece"] <= 0.10 else "❌"
        target_cov = "✅" if 0.75 <= metrics["coverage"] <= 0.90 else "❌"
        print(f"   • ECE (Expected Calibration Error): {metrics['ece']:.4f} {target_ece} (target ≤ 0.10)")
        print(f"   • Coverage (1 - abstention rate): {metrics['coverage']:.2%} {target_cov} (target 75-90%)")
        print(f"   • Abstentions: {metrics.get('abstentions', 0)}")
        print()

    # Stats feedback
    if det.use_proto:
        feedback_stats = det.feedback_store.get_stats(days=1)
        print("📊 FEEDBACK STATS (last 24h):")
        print(f"   • Total events: {feedback_stats['total_events']}")
        print(f"   • Corrections: {feedback_stats['corrections']}")
        print(f"   • Accuracy on corrections: {feedback_stats['accuracy']:.2%}")
        print()

    # F1 par émotion
    if "per_label_f1" in metrics:
        print("🎭 F1 PAR ÉMOTION :")
        for label, f1 in sorted(metrics["per_label_f1"].items(), key=lambda x: -x[1]):
            support = metrics.get("per_label_support", {}).get(label, 0)
            print(f"   • {label:15s} : {f1:.3f}  (n={support})")
        print()

    # Confusion matrix
    if "labels" in metrics.get("confusion", {}):
        labels = metrics["confusion"]["labels"]
        # matrix = metrics["confusion"]["matrix"]  # Unused for now
        print("🔀 CONFUSION MATRIX :")
        print(f"   Labels : {', '.join(labels)}")
        print()

    print("💾 RAPPORT SAUVEGARDÉ :")
    print(f"   {out_json}")

    # Sauvegarder prototypes (UNIQUEMENT si learning activé)
    if det.use_proto and not args.no_learn and args.cv != 'loso':
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_path = ROOT / "data" / f"prototypes_{timestamp}.json"
        det.classifier.save(save_path)
        print(f"   {save_path} (prototypes)")
    print()

    # Verdict Phase 1 (adaptif selon mode)
    macro_f1 = metrics.get("macro_f1", 0.0)
    # accuracy = metrics.get("accuracy", 0.0)  # Unused for now
    ece = metrics.get("ece", 1.0)
    coverage = metrics.get("coverage", 0.0)

    # Ajouter métriques bench si disponibles
    if bench_results:
        print("🔌 LATENCE ENCODEUR RÉELLE:")
        print(f"   • p50: {bench_results['encoder_p50_ms']}ms")
        print(f"   • p95: {bench_results['encoder_p95_ms']}ms")
        print(f"   • avg: {bench_results['encoder_avg_ms']}ms")
        print()

    mode_label = "COLD-START" if args.no_bootstrap else "WITH BOOTSTRAP"
    if args.no_learn:
        mode_label += " + NO LEARNING"

    print("=" * 80)
    print(f"🎯 VERDICT PHASE 1 - {mode_label}")
    print("=" * 80)

    # Critères ajustés selon mode
    if args.no_bootstrap and args.no_learn:
        # Cold-start : critères plus permissifs
        f1_target = 0.35
        f1_ok = macro_f1 >= f1_target
        target_label = "≥ 0.35 (cold-start)"
    elif args.cv == 'loso':
        # LOSO : critères standards
        f1_target = 0.50
        f1_ok = macro_f1 >= f1_target
        target_label = "≥ 0.50 (LOSO)"
    else:
        # Avec bootstrap : critères exigeants
        f1_target = 0.60
        f1_ok = macro_f1 >= f1_target
        target_label = "≥ 0.60 (bootstrap)"

    lat_ok = p95 <= 120
    ece_ok = ece <= 0.10
    cov_ok = 0.75 <= coverage <= 0.90

    all_ok = f1_ok and lat_ok and ece_ok and cov_ok

    print(f"   • Macro-F1: {macro_f1:.3f} {'✅' if f1_ok else '❌'} (target {target_label})")
    print(f"   • Latence: {p95}ms {'✅' if lat_ok else '❌'} (target ≤ 120ms)")
    print(f"   • ECE: {ece:.4f} {'✅' if ece_ok else '❌'} (target ≤ 0.10)")
    print(f"   • Coverage: {coverage:.2%} {'✅' if cov_ok else '❌'} (target 75-90%)")
    print()

    if all_ok:
        print("✅ OBJECTIFS PHASE 1 ATTEINTS")
        print(f"   Mode: {mode_label}")
        print(f"   Validation: {'Robuste' if args.no_bootstrap and args.no_learn else 'Standard'}")
    else:
        print("⚠️  Ajustements nécessaires")
        if not f1_ok:
            print(f"   • F1 insuffisant ({macro_f1:.3f} < {f1_target})")
        if not ece_ok:
            print("   • ECE trop élevé (essayer --temperature 0.04)")
        if not cov_ok:
            print("   • Coverage hors cible (ajuster --min-confidence)")

    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
