#!/usr/bin/env python3
"""
Train a calibrated linear head (LogisticRegression + isotonic calibration) on E5 embeddings.
This replaces the pure prototype classifier with a more robust decision boundary.

Expected gain: +0.10 to +0.15 F1 macro
"""

import argparse
import json
import random

# Import preprocessing
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

sys.path.append('.')
from scripts.preprocess_text import preprocess_light

# Core-8 emotions
CORE_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral", "frustration"]


def set_seeds(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    # Note: sklearn uses numpy random state, so setting np.random.seed is sufficient


def prep_e5(texts):
    """Prepend 'query:' prefix for E5 models."""
    return [f"query: {t.strip()}" for t in texts]


def load_yaml_dataset(yaml_dir):
    """Load dataset from YAML files."""
    X, y = [], []
    yaml_dir = Path(yaml_dir)

    for fp in yaml_dir.glob("*.yaml"):
        data = yaml.safe_load(fp.read_text(encoding='utf-8'))
        text = (data.get("text") or "").strip()
        emotion = data.get("emotion")

        if text and emotion in CORE_EMOTIONS:
            X.append(preprocess_light(text))
            y.append(CORE_EMOTIONS.index(emotion))

    return X, np.array(y)


def compute_config_hash(config_dict):
    """Compute a hash of the configuration for reproducibility."""
    import hashlib

    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def main(args):
    print("🚀 Training Linear Head with mE5-large + Isotonic Calibration")
    print(f"   Encoder: {args.encoder}")
    print(f"   Dataset: {args.yaml_dir}")
    print(f"   Seed: {args.seed}")
    print()

    # Set seeds for reproducibility
    set_seeds(args.seed)

    # Define configuration for hash
    config = {
        "encoder": args.encoder,
        "preprocessing": "light",
        "mapping": "surprise",  # confusion/curiosity → surprise
        "C": args.C,
        "cv": args.cv,
        "test_size": args.test_size,
        "seed": args.seed,
        "version": "2.4.2",
    }
    config_hash = compute_config_hash(config)
    print(f"   Config hash: {config_hash}")

    # Load model
    print("📥 Loading encoder...")
    model = SentenceTransformer(args.encoder)

    # Load dataset
    print("📂 Loading dataset...")
    X_text, y = load_yaml_dataset(args.yaml_dir)
    print(f"   Loaded {len(X_text)} examples")
    print(f"   Distribution: {Counter(y)}")
    print()

    # Encode texts
    print("🔄 Encoding texts...")
    X_text_prefixed = prep_e5(X_text)
    Z = model.encode(X_text_prefixed, show_progress_bar=True, convert_to_numpy=True, batch_size=32)

    # L2 normalization
    Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)
    print(f"   Embeddings shape: {Z.shape}")
    print()

    # CRITICAL: Split train/val for TRUE evaluation (stratified)
    Z_train, Z_val, y_train, y_val = train_test_split(
        Z, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    print("📊 Data split:")
    print(f"   Train: {len(Z_train)} examples")
    print(f"   Val:   {len(Z_val)} examples")
    print(f"   Train distribution: {Counter(y_train)}")
    print(f"   Val distribution:   {Counter(y_val)}")
    print()

    # Train base model (LogisticRegression with class_weight='balanced')
    print("🎯 Training LogisticRegression (base)...")
    base_clf = LogisticRegression(
        max_iter=2000,
        n_jobs=-1,
        multi_class="multinomial",
        solver="lbfgs",
        C=args.C,  # Regularization strength
        class_weight="balanced",  # Handle class imbalance
        random_state=args.seed,
    )

    # Calibrate with isotonic regression (cv=5)
    print(f"📊 Calibrating with isotonic regression (cv={args.cv})...")
    calibrated_clf = CalibratedClassifierCV(base_clf, cv=args.cv, method="isotonic")

    # CRITICAL: Fit ONLY on train data (not on full dataset)
    calibrated_clf.fit(Z_train, y_train)

    # Evaluate on train (sanity check - should be high)
    y_train_pred = calibrated_clf.predict(Z_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred, average="macro")
    print(f"   Train Accuracy: {train_acc:.3f}")
    print(f"   Train F1 Macro: {train_f1:.3f}")
    print()

    # Evaluate on validation (TRUE performance)
    print("✅ Validation Results (UNBIASED):")
    y_val_pred = calibrated_clf.predict(Z_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")
    print(f"   Val Accuracy: {val_acc:.3f}")
    print(f"   Val F1 Macro: {val_f1:.3f}")
    print()
    print("📊 Detailed classification report:")
    print(classification_report(y_val, y_val_pred, target_names=CORE_EMOTIONS, digits=3))

    # Check if we meet target performance
    target_f1 = 0.45
    target_acc = 0.60
    print("🎯 Performance targets:")
    print(f"   F1 ≥ {target_f1}: {'✅ PASS' if val_f1 >= target_f1 else '❌ FAIL'}")
    print(f"   Acc ≥ {target_acc}: {'✅ PASS' if val_acc >= target_acc else '❌ FAIL'}")
    print()

    # Save model
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import joblib

    model_path = output_dir / "linear_head.joblib"
    joblib.dump(
        {"encoder": args.encoder, "classes": CORE_EMOTIONS, "clf": calibrated_clf, "preprocessing": "light"}, model_path
    )

    # Save metadata
    meta = {
        "version": "2.4.2-linear",
        "encoder": args.encoder,
        "classes": CORE_EMOTIONS,
        "preprocessing": "light",
        "config_hash": config_hash,
        "config": config,
        "training": {
            "seed": args.seed,
            "n_total": int(len(y)),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "train_acc": float(train_acc),
            "train_f1": float(train_f1),
            "val_acc": float(val_acc),
            "val_f1": float(val_f1),
            "C": args.C,
            "test_size": args.test_size,
        },
        "calibration": {"method": "isotonic", "cv": args.cv},
        "targets": {
            "f1_target": target_f1,
            "acc_target": target_acc,
            "f1_pass": val_f1 >= target_f1,
            "acc_pass": val_acc >= target_acc,
        },
    }

    meta_path = output_dir / "linear_head.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print("💾 Saved:")
    print(f"   Model: {model_path}")
    print(f"   Metadata: {meta_path}")
    print()
    print("🎉 Linear head training complete!")

    return val_f1, val_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-dir", required=True, help="Path to preprocessed YAML dataset")
    parser.add_argument("--output-dir", default="data", help="Output directory for model")
    parser.add_argument("--encoder", default="intfloat/multilingual-e5-large", help="Encoder model name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--C", type=float, default=2.0, help="LogisticRegression regularization strength")
    parser.add_argument("--cv", type=int, default=5, help="Cross-validation folds for calibration")
    parser.add_argument("--test-size", type=float, default=0.2, help="Fraction for validation split")
    args = parser.parse_args()
    main(args)
