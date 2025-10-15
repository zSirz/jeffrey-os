#!/usr/bin/env python3
"""
Nettoie les doublons avec mode dry-run
"""

import argparse
import hashlib
from pathlib import Path


def get_file_hash(filepath):
    """Calculate SHA256 hash"""
    h = hashlib.sha256()
    h.update(filepath.read_bytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description="Clean P0 duplicates")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    args = parser.parse_args()

    mode = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"🧹 Cleaning duplicates [{mode} MODE]")
    print("=" * 50)

    base_dir = Path.cwd()

    # Known duplicates from the project
    duplicates = [
        (
            base_dir / "src/jeffrey/core/memory/cortex_memoriel.py",
            base_dir / "src/jeffrey/core/memory/cortex/cortex_memoriel.py",
        ),
        # Add any other known duplicates here
    ]

    files_processed = 0
    files_moved = 0

    for original, duplicate in duplicates:
        if original.exists() and duplicate.exists():
            files_processed += 1
            try:
                hash1 = get_file_hash(original)
                hash2 = get_file_hash(duplicate)

                if hash1 == hash2:
                    print("🔄 Identical files:")
                    print(f"   Keep: {original.relative_to(base_dir)}")
                    print(f"   Remove: {duplicate.relative_to(base_dir)}")

                    if not args.dry_run:
                        backup = duplicate.with_suffix(".py.backup")
                        duplicate.rename(backup)
                        print(f"   ✅ Moved to {backup.name}")
                        files_moved += 1
                    else:
                        print("   → Would move to .backup")
                else:
                    print("⚠️ Different content (manual review needed):")
                    print(f"   {original.relative_to(base_dir)}: {hash1[:8]}...")
                    print(f"   {duplicate.relative_to(base_dir)}: {hash2[:8]}...")
            except Exception as e:
                print(f"❌ Error processing {duplicate.name}: {e}")
        elif duplicate.exists() and not original.exists():
            print("🔍 Orphan duplicate found:")
            print(f"   {duplicate.relative_to(base_dir)}")
            print(f"   Original not found: {original.relative_to(base_dir)}")

    if files_processed == 0:
        print("ℹ️ No duplicates found to process")

    print("=" * 50)
    print(f"✅ {mode} complete!")
    if files_processed > 0:
        print(f"   Files processed: {files_processed}")
        print(f"   Files moved: {files_moved}")

    if args.dry_run and files_processed > 0:
        print("\n💡 Run without --dry-run to apply changes")


if __name__ == "__main__":
    main()
