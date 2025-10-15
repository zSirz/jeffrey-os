#!/usr/bin/env python3
"""
Universal text preprocessing module for Jeffrey OS.
Handles slang normalization, emoji demojization, and cleaning.

Features:
- Dictionary-based slang → standard English/French
- Emoji → textual description (e.g., 👍 → :thumbs_up:)
- URL/HTML removal
- Whitespace normalization

Usage:
  from scripts.preprocess_text import preprocess_for_ml
  clean_text = preprocess_for_ml("omw! 😊 gonna be late lol")
"""

import json
import re
from pathlib import Path

try:
    import emoji
except ImportError:
    print("⚠️  emoji library not installed. Run: pip install emoji --break-system-packages")
    emoji = None

# Load slang dictionary
SLANG_DICT_PATH = Path("data/slang_dictionary.json")


def load_slang_dictionary() -> dict[str, str]:
    """Load slang dictionary from JSON file."""
    if not SLANG_DICT_PATH.exists():
        print(f"⚠️  Slang dictionary not found at {SLANG_DICT_PATH}")
        return {}

    with open(SLANG_DICT_PATH, encoding='utf-8') as f:
        data = json.load(f)

    # Merge EN and FR slang
    combined = {}
    combined.update(data.get("slang", {}))
    combined.update(data.get("french_slang", {}))

    return combined


SLANG_DICT = load_slang_dictionary()


def normalize_slang(text: str, slang_dict: dict[str, str]) -> str:
    """
    Replace slang/abbreviations with standard equivalents.

    Args:
        text: Input text
        slang_dict: Dictionary of slang → standard mappings

    Returns:
        Text with normalized slang

    Example:
        >>> normalize_slang("idk btw omw", SLANG_DICT)
        "I do not know by the way on my way"
    """
    if not slang_dict:
        return text

    # Split by whitespace, preserving punctuation
    words = text.split()
    normalized = []

    for word in words:
        # Check lowercase version in dictionary
        lower_word = word.lower()

        # Handle punctuation (e.g., "idk," → "idk")
        base_word = re.sub(r'[^\w]', '', lower_word)

        if base_word in slang_dict:
            # Replace with standard form, preserving punctuation
            replacement = slang_dict[base_word]
            # Preserve punctuation at end
            trailing_punct = word[len(base_word) :] if len(word) > len(base_word) else ''
            normalized.append(replacement + trailing_punct)
        else:
            normalized.append(word)

    return " ".join(normalized)


def demojize_text(text: str) -> str:
    """
    Convert emojis to textual descriptions.

    Args:
        text: Input text with emojis

    Returns:
        Text with emojis replaced by descriptions

    Example:
        >>> demojize_text("Great work! 👍🚀")
        "Great work! :thumbs_up::rocket:"
    """
    if emoji is None:
        print("⚠️  emoji library not available, skipping demojization")
        return text

    return emoji.demojize(text, delimiters=(":", ":"))


def clean_text(text: str) -> str:
    """
    Basic text cleaning: URLs, HTML tags, excess whitespace.

    Args:
        text: Input text

    Returns:
        Cleaned text
    """
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_for_ml(text: str, slang_dict: dict[str, str] = None) -> str:
    """
    Full preprocessing pipeline for ML.

    Pipeline:
    1. Emoji demojization (preserve emotional signals)
    2. Slang normalization (standardize informal language)
    3. Basic cleaning (URLs, HTML, whitespace)

    Args:
        text: Raw input text
        slang_dict: Optional custom slang dictionary (uses default if None)

    Returns:
        Preprocessed text ready for encoding

    Example:
        >>> preprocess_for_ml("omw! 😊 https://example.com gonna be late lol")
        "on my way! :smiling_face_with_smiling_eyes: [URL] going to be late laugh out loud"
    """
    if slang_dict is None:
        slang_dict = SLANG_DICT

    # Step 1: Demojize (emojis → text)
    text = demojize_text(text)

    # Step 2: Normalize slang
    text = normalize_slang(text, slang_dict)

    # Step 3: Clean
    text = clean_text(text)

    return text


def preprocess_light(text: str) -> str:
    """
    Lightweight preprocessing for E5 models (preserves emojis and slang).

    E5 models are trained on raw text with native emojis, so we avoid
    aggressive normalization that destroys emotional signal.

    Args:
        text: Raw input text

    Returns:
        Lightly cleaned text (URLs removed, whitespace normalized)
    """
    import re

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_dataset_yaml(yaml_dir: str, output_dir: str) -> int:
    """
    Preprocess all YAML files in a directory.

    Args:
        yaml_dir: Input directory with YAML files
        output_dir: Output directory for preprocessed YAML

    Returns:
        Number of files processed
    """
    from pathlib import Path

    import yaml

    input_path = Path(yaml_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    yaml_files = list(input_path.glob("*.yaml"))

    print(f"🔧 Preprocessing {len(yaml_files)} YAML files...")
    print(f"   Input:  {yaml_dir}")
    print(f"   Output: {output_dir}\n")

    processed = 0

    for yaml_file in yaml_files:
        try:
            with open(yaml_file, encoding='utf-8') as f:
                data = yaml.safe_load(f)

            # Preprocess text field
            if 'text' in data:
                original = data['text']
                data['text'] = preprocess_for_ml(original)

                # Add preprocessing metadata
                if 'preprocessing' not in data:
                    data['preprocessing'] = {}
                data['preprocessing']['slang_normalized'] = True
                data['preprocessing']['emoji_demojized'] = True
                data['preprocessing']['original_length'] = len(original)
                data['preprocessing']['processed_length'] = len(data['text'])

            # Write to output
            output_file = output_path / yaml_file.name
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

            processed += 1

            if processed % 500 == 0:
                print(f"   Processed: {processed}/{len(yaml_files)}")

        except Exception as e:
            print(f"⚠️  Error processing {yaml_file.name}: {e}")
            continue

    print(f"\n✅ Preprocessing complete: {processed}/{len(yaml_files)} files")
    return processed


# ========== TESTS UNITAIRES ==========


def test_preprocessing():
    """Test suite for preprocessing functions."""

    print("🧪 Running preprocessing tests...\n")

    # Test 1: Slang normalization
    test_slang = "idk btw omw lol"
    result = normalize_slang(test_slang, SLANG_DICT)
    expected = "I do not know by the way on my way laugh out loud"
    assert result == expected, f"Slang test failed: {result} != {expected}"
    print(f"✅ Slang normalization: '{test_slang}' → '{result}'")

    # Test 2: Emoji demojization
    test_emoji = "Great! 👍🚀"
    result = demojize_text(test_emoji)
    assert ":thumbs_up:" in result and ":rocket:" in result
    print(f"✅ Emoji demojization: '{test_emoji}' → '{result}'")

    # Test 3: Full pipeline
    test_full = "omw! 😊 https://example.com gonna be late lol"
    result = preprocess_for_ml(test_full)
    assert "on my way" in result
    assert ":smiling" in result  # Emoji converted
    assert "[URL]" in result  # URL removed
    assert "going to" in result  # "gonna" normalized
    assert "laugh out loud" in result  # "lol" normalized
    print(f"✅ Full pipeline: '{test_full}' → '{result}'")

    # Test 4: French slang
    test_fr = "mdr stp jsp"
    result = normalize_slang(test_fr, SLANG_DICT)
    assert "mort de rire" in result
    assert "s'il te plaît" in result
    print(f"✅ French slang: '{test_fr}' → '{result}'")

    print("\n🎉 All preprocessing tests passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess text for Jeffrey OS ML")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--yaml-dir", help="Input YAML directory")
    parser.add_argument("--output-dir", help="Output directory for preprocessed YAML")

    args = parser.parse_args()

    if args.test:
        test_preprocessing()
    elif args.yaml_dir and args.output_dir:
        preprocess_dataset_yaml(args.yaml_dir, args.output_dir)
    else:
        # Interactive demo
        print("🔧 Jeffrey OS Text Preprocessing Demo\n")

        examples = [
            "idk btw omw lol 😊",
            "Super, encore un bug... génial. 🙄",
            "mdr tkt jsp pcq",
            "Great work! 👍🚀 gonna crush it asap",
        ]

        for ex in examples:
            result = preprocess_for_ml(ex)
            print(f"Original:  {ex}")
            print(f"Processed: {result}\n")
