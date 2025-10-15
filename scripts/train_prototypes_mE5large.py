#!/usr/bin/env python3
"""
Re-train prototypes with mE5-large (1024-dim) for v2.4.2 fallback robustness.
Uses preprocessing light mode for consistency with linear head.

IMPORTANT: GPT micro-adjustment applied - uses preprocess_light + query: prefix
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

# Fix seeds for reproducibility (GPT micro-adjustment)
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Import preprocessing
import sys

sys.path.append('.')
from scripts.preprocess_text import preprocess_light

CORE_EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral", "frustration"]


def prep_e5(texts):
    """Prepend 'query:' prefix for E5 models."""
    return [f"query: {t.strip()}" for t in texts]


def load_yaml_dataset(yaml_dir):
    """Load dataset from YAML files with light preprocessing."""
    X_by_emotion = {emotion: [] for emotion in CORE_EMOTIONS}

    yaml_dir = Path(yaml_dir)
    for fp in yaml_dir.glob("*.yaml"):
        data = yaml.safe_load(fp.read_text(encoding='utf-8'))
        text = (data.get("text") or "").strip()
        emotion = data.get("emotion")

        if text and emotion in CORE_EMOTIONS:
            # Apply light preprocessing (GPT micro-adjustment: consistent with linear head)
            text_preprocessed = preprocess_light(text)
            X_by_emotion[emotion].append(text_preprocessed)

    return X_by_emotion


def kmedoids_plusplus(embeddings, k, seed=42):
    """
    K-medoids++ initialization (similar to k-means++).
    Returns indices of k medoids.
    """
    np.random.seed(seed)
    n = len(embeddings)

    if k >= n:
        return list(range(n))

    # First center: random
    centers_idx = [np.random.randint(0, n)]

    # Remaining centers: weighted by distance to nearest existing center
    for _ in range(k - 1):
        # Compute similarities to existing centers
        sim = embeddings @ embeddings[centers_idx].T  # (n, len(centers))
        min_dist = 1 - sim.max(axis=1)  # Max similarity = min distance

        # Ensure non-negative probabilities
        min_dist = np.maximum(min_dist, 1e-8)

        # Sample proportional to distance (far points more likely)
        probs = min_dist / (min_dist.sum() + 1e-12)
        next_center = np.random.choice(np.arange(n), p=probs)
        centers_idx.append(next_center)

    return centers_idx


def compute_prototypes(embeddings, k=3, seed=42):
    """
    Compute k prototypes using k-medoids++ initialization.
    Returns array of shape (k, embedding_dim).
    """
    if len(embeddings) < k:
        # Not enough samples, pad with copies
        prototypes = np.array(embeddings)
        while len(prototypes) < k:
            prototypes = np.vstack([prototypes, embeddings[0]])
        return prototypes[:k]

    # K-medoids++ initialization
    medoid_indices = kmedoids_plusplus(embeddings, k, seed)
    prototypes = embeddings[medoid_indices]

    # L2 normalize each medoid
    prototypes = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-12)

    return prototypes


def compute_config_hash(config_dict):
    """Compute a hash of the configuration for reproducibility."""
    import hashlib

    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def main(args):
    print("🚀 Re-training Prototypes with mE5-large (1024-dim) for v2.4.2")
    print(f"   Encoder: {args.encoder}")
    print(f"   K-medoids: {args.k}")
    print(f"   Dataset: {args.yaml_dir}")
    print(f"   Seed: {args.seed}")
    print()

    # Set seeds for reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load encoder
    print("📥 Loading encoder...")
    encoder = SentenceTransformer(args.encoder)
    embedding_dim = encoder.get_sentence_embedding_dimension()
    print(f"   Embedding dimension: {embedding_dim}")
    print()

    # Load dataset
    print("📂 Loading dataset...")
    X_by_emotion = load_yaml_dataset(args.yaml_dir)

    print("   Distribution by emotion:")
    for emotion in CORE_EMOTIONS:
        count = len(X_by_emotion[emotion])
        print(f"      {emotion:12s}: {count:4d}")
    print()

    # Encode and compute prototypes for each emotion
    print("🔄 Encoding texts and computing prototypes...")
    prototypes_dict = {}

    for emotion in CORE_EMOTIONS:
        texts = X_by_emotion[emotion]
        if not texts:
            print(f"   ⚠️ No samples for {emotion}, skipping")
            continue

        # GPT micro-adjustment: consistent preprocessing + query: prefix
        texts_prefixed = prep_e5(texts)
        embeddings = encoder.encode(texts_prefixed, show_progress_bar=False, batch_size=32)

        # Convert to numpy if needed
        if hasattr(embeddings, 'numpy'):
            embeddings = embeddings.numpy()

        # L2 normalize
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)

        # Compute k prototypes
        prototypes = compute_prototypes(embeddings, k=args.k, seed=args.seed)
        prototypes_dict[emotion] = prototypes

        print(f"   ✅ {emotion:12s}: {len(texts):4d} samples → {prototypes.shape[0]} prototypes")

    print()

    # Save prototypes
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    proto_path = output_dir / "prototypes_mE5large.npz"

    # Convert dict to npz format (GPT micro-adjustment: use proto_ prefix)
    proto_arrays = {}
    for emotion, protos in prototypes_dict.items():
        proto_arrays[f"proto_{emotion}"] = protos

    np.savez(proto_path, **proto_arrays)

    # Save metadata with config hash (GPT micro-adjustment)
    config = {
        "encoder": args.encoder,
        "preprocessing": "light",
        "mapping": "surprise",  # confusion/curiosity → surprise
        "k_medoids": args.k,
        "seed": args.seed,
        "version": "2.4.2",
    }

    config_hash = compute_config_hash(config)

    meta = {
        "version": "2.4.2-prototypes-mE5large",
        "encoder": args.encoder,
        "embedding_dim": int(embedding_dim),
        "classes": CORE_EMOTIONS,
        "k_medoids": args.k,
        "preprocessing": "light",
        "seed": args.seed,
        "config_hash": config_hash,
        "config": config,
        "training": {
            "n_total": sum(len(texts) for texts in X_by_emotion.values()),
            "distribution": {emotion: len(texts) for emotion, texts in X_by_emotion.items()},
        },
    }

    meta_path = output_dir / "prototypes_mE5large.meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    print("💾 Saved:")
    print(f"   Model: {proto_path}")
    print(f"   Metadata: {meta_path}")
    print(f"   Config hash: {config_hash}")
    print()
    print("🎉 Prototype re-training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-dir", required=True, help="Path to preprocessed YAML dataset")
    parser.add_argument("--output-dir", default="data", help="Output directory for prototypes")
    parser.add_argument("--encoder", default="intfloat/multilingual-e5-large", help="Encoder model name")
    parser.add_argument("--k", type=int, default=3, help="Number of medoids per emotion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    main(args)
