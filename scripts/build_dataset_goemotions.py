#!/usr/bin/env python3
"""
Build real training dataset from GoEmotions (Google, EN).
Map 27 emotions → core-8, export as YAML compatible with Jeffrey OS.

Features:
- Single-label filtering (quality > multi-label ambiguity)
- Class-balanced sampling (300 examples/emotion)
- Deduplication
- Compatible YAML format

Requirements:
  pip install datasets pyyaml --break-system-packages

Usage:
  python scripts/build_dataset_goemotions.py
"""

import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

try:
    from datasets import load_dataset
except ImportError:
    print("❌ ERROR: 'datasets' not installed")
    print("Run: pip install datasets --break-system-packages")
    exit(1)

# Core-8 emotions Jeffrey OS
CORE8 = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral", "frustration"]

# Mapping GoEmotions (27 labels) → core-8
# Source: https://github.com/google-research/google-research/tree/master/goemotions
GO2CORE = {
    # JOY cluster
    "joy": "joy",
    "amusement": "joy",
    "excitement": "joy",
    "admiration": "joy",
    "approval": "joy",
    "gratitude": "joy",
    "love": "joy",
    "optimism": "joy",
    "pride": "joy",
    "relief": "joy",
    "caring": "joy",
    "desire": "joy",
    # SADNESS cluster
    "sadness": "sadness",
    "remorse": "sadness",
    "grief": "sadness",
    # ANGER cluster
    "anger": "anger",
    "disapproval": "anger",  # AJOUTÉ : disapproval → anger
    # FRUSTRATION cluster (new in core-8)
    "annoyance": "frustration",
    "disappointment": "frustration",
    # DISGUST cluster
    "disgust": "disgust",
    # FEAR cluster
    "fear": "fear",
    "nervousness": "fear",
    "embarrassment": "fear",
    # SURPRISE cluster
    "surprise": "surprise",
    "realization": "surprise",  # CHANGÉ : neutral → surprise
    "confusion": "surprise",  # CHANGÉ : neutral → surprise
    "curiosity": "surprise",  # CHANGÉ : neutral → surprise
    # NEUTRAL cluster
    "neutral": "neutral",
}


def main(
    out_dir="data/conversations_goemotions",
    per_class_quota=300,  # Samples per emotion
    seed=42,
):
    """Main pipeline: download, map, filter, balance, export."""

    print("🚀 BUILDING REAL DATASET FROM GOEMOTIONS")
    print("=" * 60)
    print(f"Target: {per_class_quota} examples per emotion")
    print(f"Output: {out_dir}")
    print(f"Random seed: {seed}\n")

    random.seed(seed)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Step 1: Download GoEmotions
    print("📥 Downloading GoEmotions dataset (58k examples)...")
    try:
        ds = load_dataset("go_emotions", "simplified")
    except Exception as e:
        print(f"❌ ERROR downloading dataset: {e}")
        print("💡 Check internet connection or try again later")
        exit(1)

    print(f"✅ Downloaded: {len(ds['train'])} train + {len(ds['validation'])} val + {len(ds['test'])} test\n")

    # Step 2: Combine splits for diversity
    print("🔀 Combining train/val/test splits...")
    combined = []
    for split_name in ["train", "validation", "test"]:
        for ex in ds[split_name]:
            combined.append(ex)

    random.shuffle(combined)
    print(f"✅ Combined: {len(combined)} total examples\n")

    # Step 3: Filter single-label & map to core-8
    print("🔍 Filtering single-label examples + mapping to core-8...")
    label_names = ds["train"].features["labels"].feature.names

    pool = []
    multi_label_skipped = 0
    unmapped_skipped = 0

    for ex in combined:
        labels = ex["labels"]

        # Handle both int and list formats
        if isinstance(labels, int):
            labels = [labels]

        # Only keep single-label (quality filter)
        if len(labels) != 1:
            multi_label_skipped += 1
            continue

        # Get emotion name
        label_idx = labels[0]
        emotion_name = label_names[label_idx]

        # Map to core-8
        core_emotion = GO2CORE.get(emotion_name)

        if core_emotion not in CORE8:
            unmapped_skipped += 1
            continue

        # Length filter (reasonable text)
        text = ex["text"].strip()
        if not (8 <= len(text) <= 280):  # Twitter-like range
            continue

        pool.append(
            {
                "text": text,
                "emotion": core_emotion,
                "language": "en",
                "source": "go_emotions",
                "original_label": emotion_name,
            }
        )

    print(f"✅ Filtered: {len(pool)} single-label examples")
    print(f"   Skipped: {multi_label_skipped} multi-label, {unmapped_skipped} unmapped\n")

    # Step 4: Deduplication
    print("🧹 Deduplicating by text...")
    seen = set()
    dedup = []

    for ex in pool:
        key = ex["text"].lower().strip()
        if key not in seen:
            seen.add(key)
            dedup.append(ex)

    duplicates_removed = len(pool) - len(dedup)
    print(f"✅ Deduplicated: {len(dedup)} unique examples ({duplicates_removed} duplicates removed)\n")

    # Step 5: Class-balanced sampling
    print(f"⚖️  Balancing classes ({per_class_quota} samples/emotion)...")
    by_emotion = defaultdict(list)

    for ex in dedup:
        by_emotion[ex["emotion"]].append(ex)

    # Show distribution before sampling
    print("Distribution BEFORE sampling:")
    for emo in CORE8:
        count = len(by_emotion.get(emo, []))
        print(f"  {emo:12s} : {count:5d} examples")

    # Sample per class
    selected = []
    for emo in CORE8:
        bucket = by_emotion.get(emo, [])
        random.shuffle(bucket)
        take = bucket[:per_class_quota]
        selected.extend(take)

        if len(take) < per_class_quota:
            print(f"⚠️  {emo}: only {len(take)}/{per_class_quota} available (class undersampled)")

    random.shuffle(selected)

    # Show final distribution
    final_counts = Counter([x["emotion"] for x in selected])
    print("\nDistribution AFTER sampling:")
    for emo in CORE8:
        count = final_counts.get(emo, 0)
        print(f"  {emo:12s} : {count:5d} examples")

    print(f"\n✅ Final dataset: {len(selected)} balanced examples\n")

    # Step 6: Export to YAML
    print(f"💾 Exporting to YAML ({out_dir})...")

    for i, ex in enumerate(selected, 1):
        item = {
            "scenario_id": f"goemo_{i:05d}",
            "emotion": ex["emotion"],
            "language": ex["language"],
            "text": ex["text"],
            "created_at": datetime.utcnow().isoformat(),
            "source": ex["source"],
            "quality": "public_dataset_single_label",
            "original_goemotions_label": ex["original_label"],
        }

        fp = Path(out_dir) / f"scenario_{i:05d}_{ex['emotion']}_en.yaml"

        with fp.open("w", encoding="utf-8") as f:
            yaml.dump(item, f, allow_unicode=True, default_flow_style=False)

    print(f"✅ Exported {len(selected)} YAML files\n")

    # Summary
    print("=" * 60)
    print("🎉 GOEMOTIONS DATASET READY!")
    print(f"📂 Location: {out_dir}")
    print(f"📊 Total files: {len(selected)}")
    print("📋 Distribution: balanced across core-8")
    print("🌍 Language: English")
    print("✨ Quality: Single-label, deduplicated, real Reddit comments")
    print("\n👉 Next step: python scripts/train_prototypes_optimized.py --yaml-dir", out_dir)
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build GoEmotions dataset for Jeffrey OS")
    parser.add_argument("--out-dir", default="data/conversations_goemotions", help="Output directory")
    parser.add_argument("--quota", type=int, default=300, help="Samples per emotion")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    main(out_dir=args.out_dir, per_class_quota=args.quota, seed=args.seed)
