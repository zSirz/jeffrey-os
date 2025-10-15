#!/usr/bin/env python3
"""
Merge multiple YAML directories into one, with deduplication.

Usage:
  python scripts/merge_yaml_dirs.py --out data/conversations_real \\
    data/conversations_goemotions data/conversations_annotated
"""

import argparse
import hashlib
from pathlib import Path

import yaml


def main(out_dir, *src_dirs):
    """Merge YAML files from multiple directories."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    seen = set()  # For deduplication
    idx = 1
    total = 0

    print(f"🔀 Merging YAML directories → {out_dir}")
    print("=" * 60)

    for src_dir in src_dirs:
        src = Path(src_dir)
        if not src.exists():
            print(f"⚠️  Skipping non-existent: {src_dir}")
            continue

        files = list(src.glob("*.yaml"))
        print(f"\n📂 Processing: {src_dir} ({len(files)} files)")

        for fp in files:
            try:
                data = yaml.safe_load(fp.read_text(encoding="utf-8"))

                # Extract fields
                text = (data.get("text") or "").strip()
                emotion = data.get("emotion")
                lang = (data.get("language") or "en").lower()

                if not text or emotion is None:
                    continue

                # Deduplication by (text, emotion)
                key = hashlib.sha1((text.lower() + "|" + emotion).encode("utf-8")).hexdigest()

                if key in seen:
                    continue  # Skip duplicate

                seen.add(key)

                # Write to output
                out_fp = out / f"scenario_{idx:05d}_{emotion}_{lang}.yaml"
                out_fp.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")

                idx += 1
                total += 1

            except Exception as e:
                print(f"⚠️  Error processing {fp.name}: {e}")
                continue

    print("\n" + "=" * 60)
    print(f"✅ Merge complete: {total} unique files → {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge YAML directories")
    parser.add_argument("--out", default="data/conversations_real", help="Output directory")
    parser.add_argument("src", nargs="+", help="Source directories to merge")

    args = parser.parse_args()

    main(args.out, *args.src)
