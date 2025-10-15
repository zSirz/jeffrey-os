#!/usr/bin/env python3
"""
Optimized ProtoClassifier training with E5-base + K-medoids.

Improvements over v2.4.2:
- E5-base encoder (768-dim, emotion-optimized)
- K-medoids (k=3) for intra-class variance
- Preprocessing integration (slang + emoji)
- Temperature scaling preparation

Usage:
  python scripts/train_prototypes_e5_optimized.py \\
    --yaml-dir data/conversations_preprocessed
"""

import argparse
import logging
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CORE_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral", "frustration"]
ENCODER_MODEL = "intfloat/multilingual-e5-base"
SEED = 42
K_MEDOIDS = 3  # Number of prototypes per emotion

# Set seeds for reproducibility
np.random.seed(SEED)
random.seed(SEED)


@dataclass
class TrainingConfig:
    """Training configuration."""

    yaml_dir: str
    output_dir: str = "data"
    k_medoids: int = K_MEDOIDS
    encoder_model: str = ENCODER_MODEL
    seed: int = SEED


def _prep_for_e5(texts: list[str]) -> list[str]:
    """Prepend 'query:' prefix for E5 models (CRITICAL)."""
    return [f"query: {t.strip()}" for t in texts]


def load_encoder():
    """Load E5-base encoder."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
        sys.exit(1)

    logger.info(f"🔧 Loading encoder: {ENCODER_MODEL}")
    model = SentenceTransformer(ENCODER_MODEL)

    # Verify
    test_emb = model.encode(["test"], show_progress_bar=False)
    logger.info(f"✅ Encoder loaded: {test_emb[0].shape[0]}-dim embeddings")

    return model


def load_training_data(yaml_dir: str) -> list[dict]:
    """Load preprocessed YAML files."""
    yaml_path = Path(yaml_dir)
    yaml_files = list(yaml_path.glob("*.yaml"))

    logger.info(f"📂 Loading from: {yaml_dir}")
    logger.info(f"   Found {len(yaml_files)} YAML files")

    examples = []

    for fp in yaml_files:
        try:
            with open(fp, encoding='utf-8') as f:
                data = yaml.safe_load(f)

            text = data.get("text", "").strip()
            emotion = data.get("emotion")

            if text and emotion in CORE_EMOTIONS:
                examples.append(
                    {
                        "text": text,
                        "emotion": emotion,
                        "scenario_id": data.get("scenario_id", f"unknown_{len(examples)}"),
                    }
                )
        except Exception as e:
            logger.warning(f"⚠️  Error loading {fp.name}: {e}")
            continue

    logger.info(f"✅ Loaded {len(examples)} valid examples\n")

    # Distribution
    emotion_counts = Counter([ex["emotion"] for ex in examples])
    logger.info("Distribution by emotion:")
    for emo in CORE_EMOTIONS:
        count = emotion_counts.get(emo, 0)
        logger.info(f"  {emo:12s} : {count:5d}")

    return examples


def compute_k_medoids(embeddings: np.ndarray, k: int = 3) -> np.ndarray:
    """
    Compute k medoids using PAM (Partitioning Around Medoids).

    Algorithm:
    1. Initialize k centers using k++ (greedy max-min)
    2. Assign points to nearest center
    3. Update each center to the point that minimizes total distance
    4. Repeat until convergence

    Args:
        embeddings: L2-normalized embeddings (n, dim)
        k: Number of medoids

    Returns:
        Array of k medoid embeddings (k, dim)
    """
    n = len(embeddings)

    if n <= k:
        # Not enough points, return all
        return embeddings

    # Step 1: k++ initialization (greedy max-min for diversity)
    centers_idx = [np.random.randint(0, n)]

    for _ in range(1, k):
        # Compute min distance to existing centers
        sim = embeddings @ embeddings[centers_idx].T  # (n, len(centers))
        min_dist = 1 - sim.max(axis=1)  # Max similarity = min distance

        # Ensure non-negative probabilities (add small epsilon for numerical stability)
        min_dist = np.maximum(min_dist, 1e-8)

        # Sample proportional to distance (far points more likely)
        probs = min_dist / (min_dist.sum() + 1e-12)
        next_center = np.random.choice(np.arange(n), p=probs)
        centers_idx.append(next_center)

    centers_idx = np.array(centers_idx)

    # Step 2: PAM refinement (5 iterations, fast convergence)
    for iteration in range(5):
        # Assign: each point to nearest center
        sim = embeddings @ embeddings[centers_idx].T  # (n, k)
        assignments = sim.argmax(axis=1)  # (n,)

        # Update: for each cluster, find medoid (point minimizing total distance)
        new_centers = []

        for j in range(k):
            cluster_idx = np.where(assignments == j)[0]

            if len(cluster_idx) == 0:
                # Empty cluster, keep current center
                new_centers.append(centers_idx[j])
                continue

            # Compute pairwise similarities within cluster
            cluster_embs = embeddings[cluster_idx]
            sim_matrix = cluster_embs @ cluster_embs.T  # (n_cluster, n_cluster)

            # Sum of distances (1 - similarity) for each point
            dist_sum = (1 - sim_matrix).sum(axis=1)

            # Medoid = point with minimal total distance
            medoid_local_idx = dist_sum.argmin()
            medoid_global_idx = cluster_idx[medoid_local_idx]
            new_centers.append(medoid_global_idx)

        new_centers = np.array(new_centers)

        # Check convergence
        if np.all(new_centers == centers_idx):
            logger.debug(f"   K-medoids converged at iteration {iteration + 1}")
            break

        centers_idx = new_centers

    return embeddings[centers_idx]


def l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-12)


def train_prototypes(config: TrainingConfig) -> dict:
    """
    Main training loop with LOSO validation.

    Returns:
        Dictionary with prototypes and metadata
    """
    logger.info("🚀 Training ProtoClassifier with E5-base + K-medoids")
    logger.info(f"   Encoder: {config.encoder_model}")
    logger.info(f"   K-medoids: {config.k_medoids} per emotion")
    logger.info(f"   Seed: {config.seed}\n")

    # Load data
    examples = load_training_data(config.yaml_dir)

    if len(examples) == 0:
        raise ValueError("No valid examples found!")

    # Load encoder
    encoder = load_encoder()

    # Encode all texts (with E5 prefix!)
    logger.info("📊 Encoding texts with E5-base...")
    texts = [ex["text"] for ex in examples]
    texts_prefixed = _prep_for_e5(texts)

    embeddings = encoder.encode(texts_prefixed, batch_size=32, show_progress_bar=True, convert_to_numpy=True)

    # L2 normalize
    embeddings = l2_normalize(embeddings)

    logger.info(f"✅ Encoded {len(embeddings)} texts → {embeddings.shape}\n")

    # Group by emotion
    by_emotion = defaultdict(list)
    for i, ex in enumerate(examples):
        by_emotion[ex["emotion"]].append(i)

    # Compute K-medoids prototypes per emotion
    logger.info(f"🔧 Computing K-medoids (k={config.k_medoids}) per emotion...")

    prototypes = {}

    for emotion in CORE_EMOTIONS:
        indices = by_emotion.get(emotion, [])

        if len(indices) == 0:
            logger.warning(f"⚠️  {emotion}: No examples, skipping")
            continue

        emotion_embs = embeddings[indices]

        # Compute k medoids
        k_actual = min(config.k_medoids, len(indices))
        medoids = compute_k_medoids(emotion_embs, k=k_actual)

        # Normalize
        medoids = l2_normalize(medoids)

        prototypes[emotion] = medoids

        logger.info(f"   {emotion:12s} : {len(indices):4d} examples → {medoids.shape[0]} medoids")

    logger.info(f"✅ Prototypes computed for {len(prototypes)} emotions\n")

    # LOSO Validation (Leave-One-Scenario-Out)
    logger.info("📈 Running LOSO validation...")

    # Group by scenario
    by_scenario = defaultdict(list)
    for i, ex in enumerate(examples):
        scenario_id = ex.get("scenario_id", f"unknown_{i}")
        by_scenario[scenario_id].append(i)

    all_y_true = []
    all_y_pred = []
    all_y_probs = []

    for scenario_id, test_indices in by_scenario.items():
        # Train set = all other scenarios
        train_indices = [i for i in range(len(examples)) if i not in test_indices]

        # Recompute prototypes on train set
        train_by_emotion = defaultdict(list)
        for idx in train_indices:
            train_by_emotion[examples[idx]["emotion"]].append(idx)

        scenario_protos = {}
        for emotion in CORE_EMOTIONS:
            emo_train_idx = train_by_emotion.get(emotion, [])
            if len(emo_train_idx) == 0:
                continue

            emo_embs = embeddings[emo_train_idx]
            k_actual = min(config.k_medoids, len(emo_train_idx))
            medoids = compute_k_medoids(emo_embs, k=k_actual)
            medoids = l2_normalize(medoids)
            scenario_protos[emotion] = medoids

        # Predict on test set
        for idx in test_indices:
            test_emb = embeddings[idx]
            true_label = examples[idx]["emotion"]

            # Compute similarities to all medoids of all emotions
            scores = {}
            for emotion, medoids in scenario_protos.items():
                # Max similarity across k medoids
                sims = test_emb @ medoids.T  # (k,)
                scores[emotion] = sims.max()

            # Softmax for probabilities
            exp_scores = {k: np.exp(v) for k, v in scores.items()}
            total = sum(exp_scores.values())
            probs = {k: v / total for k, v in exp_scores.items()}

            # Predict
            pred_label = max(scores, key=scores.get)

            all_y_true.append(true_label)
            all_y_pred.append(pred_label)
            all_y_probs.append(probs)

    # Compute metrics
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
    except ImportError:
        logger.error("scikit-learn not installed. Run: pip install scikit-learn")
        sys.exit(1)

    accuracy = accuracy_score(all_y_true, all_y_pred)
    f1_macro = f1_score(all_y_true, all_y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(all_y_true, all_y_pred, average='weighted', zero_division=0)

    logger.info("\n📊 LOSO Validation Results:")
    logger.info(f"   Accuracy  : {accuracy:.3f}")
    logger.info(f"   F1 Macro  : {f1_macro:.3f}")
    logger.info(f"   F1 Weighted : {f1_weighted:.3f}")

    # Compute ECE (Expected Calibration Error)
    confidences = [max(probs.values()) for probs in all_y_probs]
    correct = [1 if t == p else 0 for t, p in zip(all_y_true, all_y_pred)]
    ece = compute_expected_calibration_error(correct, confidences, n_bins=10)

    logger.info(f"   ECE       : {ece:.3f}")

    # Confusion matrix
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import confusion_matrix
    except ImportError:
        logger.warning("⚠️  matplotlib/seaborn not installed, skipping confusion matrix plot")
        cm = None

    if 'confusion_matrix' in locals():
        cm = confusion_matrix(all_y_true, all_y_pred, labels=CORE_EMOTIONS)

        # Save confusion matrix visualization
        try:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CORE_EMOTIONS, yticklabels=CORE_EMOTIONS)
            plt.title(f'Confusion Matrix - E5-base K-medoids (F1={f1_macro:.3f})')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()

            cm_path = Path(config.output_dir) / "confusion_matrix_v2.4.2.png"
            plt.savefig(cm_path, dpi=150)
            logger.info(f"📊 Confusion matrix saved: {cm_path}")
        except Exception as e:
            logger.warning(f"⚠️  Error saving confusion matrix: {e}")

    # Save prototypes
    output_file = Path(config.output_dir) / "prototypes.npz"

    # Flatten prototypes for saving (each emotion → (k, dim))
    save_dict = {}
    for emotion, medoids in prototypes.items():
        save_dict[f"proto_{emotion}"] = medoids

    np.savez(output_file, **save_dict)

    logger.info(f"\n💾 Prototypes saved: {output_file}")

    # Save metadata
    meta = {
        "version": "2.4.0",
        "created_at": datetime.now().isoformat(),
        "encoder_model": config.encoder_model,
        "seed": config.seed,
        "k_medoids": config.k_medoids,
        "embedding_dim": embeddings.shape[1],
        "allowed_labels": CORE_EMOTIONS,
        "validation": {
            "method": "LOSO_by_scenario",
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "accuracy": float(accuracy),
            "ece": float(ece),
        },
        "training": {
            "n_examples": len(examples),
            "n_scenarios": len(by_scenario),
            "distribution": {emo: len(indices) for emo, indices in by_emotion.items()},
        },
    }

    meta_file = Path(config.output_dir) / "prototypes.meta.json"

    import json

    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)

    logger.info(f"📋 Metadata saved: {meta_file}")

    return {
        "prototypes": prototypes,
        "metadata": meta,
        "validation": {
            "y_true": all_y_true,
            "y_pred": all_y_pred,
            "y_probs": all_y_probs,
        },
    }


def compute_expected_calibration_error(correct: list[int], confidences: list[float], n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).

    Args:
        correct: Binary list (1 if correct, 0 if wrong)
        confidences: Predicted confidence scores [0, 1]
        n_bins: Number of bins

    Returns:
        ECE score
    """
    correct = np.array(correct)
    confidences = np.array(confidences)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Samples in this bin
        in_bin = np.logical_and(confidences >= bin_lower, confidences < bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(correct[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ProtoClassifier with E5-base + K-medoids")
    parser.add_argument("--yaml-dir", required=True, help="Directory with preprocessed YAML files")
    parser.add_argument("--output-dir", default="data", help="Output directory for prototypes")
    parser.add_argument("--k-medoids", type=int, default=K_MEDOIDS, help="Number of medoids per emotion")

    args = parser.parse_args()

    config = TrainingConfig(yaml_dir=args.yaml_dir, output_dir=args.output_dir, k_medoids=args.k_medoids)

    result = train_prototypes(config)

    logger.info("\n🎉 Training complete!")
    logger.info(f"   F1 Macro: {result['metadata']['validation']['f1_macro']:.3f}")
    logger.info(f"   ECE: {result['metadata']['validation']['ece']:.3f}")

    # Check if we reached target
    f1 = result['metadata']['validation']['f1_macro']
    if f1 >= 0.50:
        logger.info("🎯 Target F1 ≥ 0.50 REACHED!")
    else:
        logger.warning(f"⚠️  F1 below target: {f1:.3f} < 0.50")
