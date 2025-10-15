#!/usr/bin/env python3
"""
Fix all syntax errors in Python files automatically
"""

import ast
import sys
from pathlib import Path


def fix_file(filepath):
    """Attempt to fix common syntax errors in a Python file"""
    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()

        # Try to parse first
        try:
            ast.parse(content)
            return True  # Already valid
        except SyntaxError as e:
            print(f"Fixing {filepath}: Line {e.lineno}: {e.msg}")

        lines = content.split('\n')
        fixed = False

        # Common fixes
        for i in range(len(lines)):
            line = lines[i]

            # Fix functions/methods with missing body
            if i > 0 and (
                lines[i - 1].strip().endswith(':')
                and lines[i - 1]
                .strip()
                .startswith(
                    ('def ', 'class ', 'if ', 'elif ', 'else:', 'try:', 'except', 'finally:', 'for ', 'while ', 'with ')
                )
            ):
                # Check if next line is not indented
                if i < len(lines) and lines[i].strip() and not lines[i].startswith((' ', '\t')):
                    # Add pass statement
                    lines.insert(i, '    pass')
                    fixed = True
                elif i == len(lines) - 1 or (i < len(lines) - 1 and not lines[i].strip()):
                    # Empty body - add pass
                    lines.insert(i, '    pass')
                    fixed = True

            # Fix unexpected indentation
            if 'unexpected indent' in str(e.msg):
                # Try to align with previous non-empty line
                if i > 0:
                    prev_indent = len(lines[i - 1]) - len(lines[i - 1].lstrip())
                    curr_indent = len(line) - len(line.lstrip())
                    if curr_indent > prev_indent + 4:
                        # Reduce indentation
                        lines[i] = ' ' * prev_indent + line.lstrip()
                        fixed = True

        # Try to reparse
        new_content = '\n'.join(lines)
        try:
            ast.parse(new_content)
            # Success! Write back
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        except:
            # If still failing, try autopep8 or black as fallback
            try:
                import black

                formatted = black.format_str(content, mode=black.Mode())
                ast.parse(formatted)  # Verify it's valid
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(formatted)
                return True
            except:
                pass

            try:
                import autopep8

                fixed_content = autopep8.fix_code(content, options={'aggressive': 2})
                ast.parse(fixed_content)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                return True
            except:
                pass

        return False

    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return False


def main():
    """Fix all Python files with syntax errors"""
    errors = []
    fixed = []

    for py_file in Path('src/jeffrey').rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue

        try:
            with open(py_file) as f:
                ast.parse(f.read())
        except SyntaxError:
            # Try to fix
            if fix_file(py_file):
                fixed.append(str(py_file))
            else:
                errors.append(str(py_file))

    print(f"\n✅ Fixed {len(fixed)} files")
    print(f"❌ Still have errors: {len(errors)} files")

    if errors:
        print("\nFiles still with errors:")
        for err in errors[:10]:
            print(f"  - {err}")

    return len(errors) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
