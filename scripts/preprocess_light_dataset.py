#!/usr/bin/env python3
"""
Lightweight preprocessing for dataset (preserves emojis and slang).
"""

import argparse
from pathlib import Path

import yaml
from preprocess_text import preprocess_light
from tqdm import tqdm


def main(args):
    input_dir = Path(args.yaml_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yaml_files = list(input_dir.glob("*.yaml"))
    print(f"🔧 Preprocessing {len(yaml_files)} YAML files (LIGHT MODE)...")
    print(f"   Input:  {input_dir}")
    print(f"   Output: {output_dir}")

    for yaml_file in tqdm(yaml_files, desc="Processing"):
        with open(yaml_file, encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # Apply light preprocessing
        if 'text' in data and data['text']:
            data['text'] = preprocess_light(data['text'])

        # Save to output directory
        output_file = output_dir / yaml_file.name
        with open(output_file, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

    print(f"✅ Done! Processed {len(yaml_files)} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(args)
