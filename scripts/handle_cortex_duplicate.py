#!/usr/bin/env python3
"""
Gère le doublon cortex_memoriel de manière sûre
"""

import hashlib
import shutil
from pathlib import Path


def get_file_info(filepath: Path):
    """Get file size and hash"""
    if not filepath.exists():
        return None, None

    size = filepath.stat().st_size
    h = hashlib.sha256()
    h.update(filepath.read_bytes())
    return size, h.hexdigest()


def main():
    print("🔍 Analyzing cortex_memoriel duplicates")
    print("=" * 50)

    base_dir = Path.cwd()

    file1 = base_dir / "src/jeffrey/core/memory/cortex_memoriel.py"
    file2 = base_dir / "src/jeffrey/core/memory/cortex/cortex_memoriel.py"

    size1, hash1 = get_file_info(file1)
    size2, hash2 = get_file_info(file2)

    print(f"📄 File 1: {file1.relative_to(base_dir) if file1.exists() else 'NOT FOUND'}")
    if size1:
        print(f"   Size: {size1} bytes")
        print(f"   Hash: {hash1[:16]}...")

    print(f"\n📄 File 2: {file2.relative_to(base_dir) if file2.exists() else 'NOT FOUND'}")
    if size2:
        print(f"   Size: {size2} bytes")
        print(f"   Hash: {hash2[:16]}...")

    if not file1.exists() and not file2.exists():
        print("\n❌ Neither file exists!")
        return False

    if not file2.exists():
        print("\n✅ No duplicate found (only file1 exists)")
        return True

    if not file1.exists() and file2.exists():
        print("\n⚠️ Only file2 exists - moving to correct location")
        file1.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(file2), str(file1))
        print(f"   ✅ Moved to {file1.relative_to(base_dir)}")
        # Clean empty dir if exists
        if file2.parent.exists() and not list(file2.parent.iterdir()):
            file2.parent.rmdir()
            print(f"   🗑️ Removed empty directory {file2.parent.name}")
        return True

    # Both exist - compare
    if hash1 == hash2:
        print("\n✅ Files are identical - keeping file1, backing up file2")
        backup = file2.with_suffix(".py.backup")
        shutil.move(str(file2), str(backup))
        print(f"   ✅ Backed up duplicate to {backup.name}")
        # Clean empty dir if exists
        if file2.parent.exists() and len(list(file2.parent.iterdir())) <= 1:  # Only backup file
            file2.parent.rmdir()
            print(f"   🗑️ Removed empty directory {file2.parent.name}")
    else:
        print("\n⚠️ Files are DIFFERENT!")
        print("\n   Analyzing to determine which to keep...")

        # Analyser les imports pour déterminer la plus complète
        content1 = file1.read_text()
        content2 = file2.read_text()

        imports1 = len(
            [l for l in content1.splitlines() if l.strip().startswith("import ") or l.strip().startswith("from ")]
        )
        imports2 = len(
            [l for l in content2.splitlines() if l.strip().startswith("import ") or l.strip().startswith("from ")]
        )

        classes1 = len([l for l in content1.splitlines() if l.strip().startswith("class ")])
        classes2 = len([l for l in content2.splitlines() if l.strip().startswith("class ")])

        print("\n   File1 stats:")
        print(f"      Size: {size1} bytes")
        print(f"      Imports: {imports1}")
        print(f"      Classes: {classes1}")

        print("\n   File2 stats:")
        print(f"      Size: {size2} bytes")
        print(f"      Imports: {imports2}")
        print(f"      Classes: {classes2}")

        # Décision basée sur la taille et la complexité
        score1 = size1 + (imports1 * 100) + (classes1 * 500)
        score2 = size2 + (imports2 * 100) + (classes2 * 500)

        if score1 >= score2:
            print(f"\n   → Keeping file1 (score: {score1} vs {score2})")
            backup = file2.with_suffix(".py.backup")
            shutil.move(str(file2), str(backup))
            print(f"   ✅ Backed up file2 to {backup.name}")
        else:
            print(f"\n   → Keeping file2 (score: {score2} vs {score1})")
            backup = file1.with_suffix(".py.backup")
            shutil.move(str(file1), str(backup))
            shutil.move(str(file2), str(file1))
            print(f"   ✅ Moved file2 to file1 location, backed up old file1 to {backup.name}")

        # Clean empty dir if exists
        if file2.parent.exists() and len(list(file2.parent.iterdir())) <= 1:
            try:
                file2.parent.rmdir()
                print(f"   🗑️ Removed empty directory {file2.parent.name}")
            except OSError:
                pass  # Directory not empty or other issue

    print("\n✅ Cortex duplicate handling complete!")
    return True


if __name__ == "__main__":
    import sys

    success = main()
    sys.exit(0 if success else 1)
