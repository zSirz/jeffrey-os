#!/usr/bin/env python3
"""Fix MemoryMoment using AST - cross-platform and robust"""

import ast
import logging
import os
import sys
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryMomentFixer(ast.NodeTransformer):
    """AST transformer to fix MemoryMoment calls"""

    def __init__(self):
        self.changes_made = 0
        self.files_affected: set[str] = set()

    def visit_Call(self, node):
        self.generic_visit(node)

        # Check if this is a MemoryMoment call
        is_memory_moment = False

        if isinstance(node.func, ast.Name) and node.func.id == "MemoryMoment":
            is_memory_moment = True
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "MemoryMoment":
            is_memory_moment = True

        if is_memory_moment:
            # Fix keyword arguments
            for keyword in node.keywords:
                if keyword.arg == "human_message":
                    keyword.arg = "message"
                    self.changes_made += 1

            # Ensure 'source' exists
            has_source = any(k.arg == "source" for k in node.keywords)
            if not has_source:
                # Create the source argument
                if sys.version_info >= (3, 8):
                    source_value = ast.Constant(value="human")
                else:
                    source_value = ast.Str(s="human")

                node.keywords.append(ast.keyword(arg="source", value=source_value))
                self.changes_made += 1

        return node


def to_source(tree):
    """Convert AST to source code - compatible with Python 3.9+ and older"""
    if sys.version_info >= (3, 9):
        return ast.unparse(tree)
    else:
        # Fallback for older Python
        try:
            import astor

            return astor.to_source(tree)
        except ImportError:
            logger.error("Python < 3.9 requires 'astor' package: pip install astor")
            sys.exit(1)


def atomic_write(path: Path, content: str):
    """Atomically write file to avoid corruption"""
    # Create temp file in same directory (for atomic rename)
    fd, temp_path = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")

    try:
        # Write content
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)

        # Atomic replace
        if os.name == "nt":  # Windows
            # Windows doesn't support atomic rename if target exists
            if path.exists():
                path.unlink()
        os.rename(temp_path, str(path))

    except Exception:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise


def fix_file(file_path: Path) -> bool:
    """Fix a single Python file"""
    try:
        content = file_path.read_text(encoding="utf-8")

        # Quick check if file needs processing
        if "MemoryMoment" not in content:
            return False

        tree = ast.parse(content, filename=str(file_path))

        fixer = MemoryMomentFixer()
        new_tree = fixer.visit(tree)

        if fixer.changes_made > 0:
            new_content = to_source(new_tree)
            atomic_write(file_path, new_content)
            logger.info(f"✅ Fixed {file_path}: {fixer.changes_made} changes")
            return True

        return False

    except SyntaxError as e:
        logger.warning(f"⚠️ Skipping {file_path}: Syntax error - {e}")
        return False
    except Exception as e:
        logger.error(f"❌ Error processing {file_path}: {e}")
        return False


def main():
    """Main execution"""
    src_path = Path("src")

    if not src_path.exists():
        logger.error("src/ directory not found")
        return 1

    fixed_files = []
    checked_files = 0

    for py_file in src_path.rglob("*.py"):
        checked_files += 1
        if fix_file(py_file):
            fixed_files.append(py_file)

    print("\n📊 Summary:")
    print(f"  Files checked: {checked_files}")
    print(f"  Files fixed: {len(fixed_files)}")

    if fixed_files:
        print("\n✅ Fixed files:")
        for f in fixed_files:
            print(f"  - {f}")
    else:
        print("\n✅ No fixes needed - all files are clean!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
