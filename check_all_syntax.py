#!/usr/bin/env python3
"""
Script to check syntax of all Python files in Jeffrey OS and report errors.
"""

import os
import subprocess
import sys
from pathlib import Path


def check_python_syntax(file_path):
    """Check syntax of a Python file using py_compile."""
    try:
        result = subprocess.run([sys.executable, '-m', 'py_compile', str(file_path)], capture_output=True, text=True)
        if result.returncode == 0:
            return True, None
        else:
            return False, result.stderr.strip()
    except Exception as e:
        return False, str(e)


def find_python_files(root_dir):
    """Find all Python files in the directory tree."""
    python_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    return python_files


def main():
    """Main function to check all Python files."""
    jeffrey_os_dir = Path("/Users/davidproz/Desktop/Jeffrey_OS")

    print("🔍 Checking syntax of all Python files in Jeffrey OS...")
    print("=" * 60)

    python_files = find_python_files(jeffrey_os_dir)
    total_files = len(python_files)
    error_files = []

    for i, py_file in enumerate(python_files, 1):
        print(f"[{i:3d}/{total_files}] Checking {py_file.relative_to(jeffrey_os_dir)}", end=" ... ")

        is_valid, error_msg = check_python_syntax(py_file)

        if is_valid:
            print("✅ OK")
        else:
            print("❌ ERROR")
            error_files.append((py_file, error_msg))

    print("\n" + "=" * 60)
    print("📊 Summary:")
    print(f"   Total files checked: {total_files}")
    print(f"   Files with errors: {len(error_files)}")
    print(f"   Files OK: {total_files - len(error_files)}")

    if error_files:
        print("\n🚨 Files with syntax errors:")
        print("-" * 40)
        for py_file, error_msg in error_files:
            rel_path = py_file.relative_to(jeffrey_os_dir)
            print(f"\n📄 {rel_path}")
            print(f"   Error: {error_msg}")
    else:
        print("\n🎉 All Python files have valid syntax!")


if __name__ == "__main__":
    main()
