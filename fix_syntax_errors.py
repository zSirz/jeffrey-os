#!/usr/bin/env python3
"""
Robust syntax error fixer for Jeffrey OS Python files.
Fixes common syntax issues to make files parseable.
"""

import ast
import os
import re
import shutil


class SyntaxErrorFixer:
    def __init__(self):
        self.fixes_applied = []

    def get_files_with_errors(self, directory: str) -> list[tuple[str, str]]:
        """Get all Python files with syntax errors and their error details."""
        errors = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    error = self.check_syntax(file_path)
                    if error:
                        errors.append((file_path, error))
        return errors

    def check_syntax(self, file_path: str) -> str | None:
        """Check if a file has syntax errors."""
        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()
            ast.parse(content)
            return None
        except SyntaxError as e:
            return f'Line {e.lineno}: {e.msg}'
        except Exception as e:
            return f'Error: {str(e)}'

    def backup_file(self, file_path: str) -> str:
        """Create a backup of the file."""
        backup_path = file_path + '.backup'
        shutil.copy2(file_path, backup_path)
        return backup_path

    def fix_file(self, file_path: str) -> bool:
        """Fix syntax errors in a file."""
        print(f"\nFixing {file_path}...")

        # Create backup
        backup_path = self.backup_file(file_path)

        try:
            with open(file_path, encoding='utf-8') as f:
                lines = f.readlines()

            original_lines = lines.copy()
            fixes_applied = []

            # Apply fixes iteratively
            max_iterations = 5
            for iteration in range(max_iterations):
                # Check current syntax
                temp_content = ''.join(lines)
                try:
                    ast.parse(temp_content)
                    # File is now syntactically correct
                    break
                except SyntaxError as e:
                    print(f"  Iteration {iteration + 1}: {e.msg} at line {e.lineno}")

                    # Apply appropriate fix based on error type
                    if self.fix_syntax_error(lines, e, fixes_applied):
                        continue
                    else:
                        print(f"  Could not fix error: {e.msg}")
                        break

            # Write the fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            # Verify the fix worked
            final_error = self.check_syntax(file_path)
            if final_error is None:
                print("  ✓ File fixed successfully!")
                print(f"    Applied fixes: {', '.join(fixes_applied)}")
                self.fixes_applied.extend(fixes_applied)
                return True
            else:
                print(f"  ✗ Still has errors: {final_error}")
                # Restore from backup if we couldn't fix it
                shutil.copy2(backup_path, file_path)
                return False

        except Exception as e:
            print(f"  ✗ Error processing file: {e}")
            # Restore from backup
            shutil.copy2(backup_path, file_path)
            return False

    def fix_syntax_error(self, lines: list[str], error: SyntaxError, fixes_applied: list[str]) -> bool:
        """Fix a specific syntax error."""
        line_num = error.lineno - 1  # Convert to 0-based indexing

        if line_num >= len(lines):
            return False

        line = lines[line_num]
        error_msg = error.msg.lower()

        # Fix 1: Missing function/class body (add 'pass')
        if "expected an indented block" in error_msg:
            return self.fix_missing_body(lines, line_num, fixes_applied)

        # Fix 2: Unexpected indent
        elif "unexpected indent" in error_msg:
            return self.fix_unexpected_indent(lines, line_num, fixes_applied)

        # Fix 3: Unmatched braces/brackets
        elif "unmatched" in error_msg or "unexpected character" in error_msg:
            return self.fix_unmatched_characters(lines, line_num, fixes_applied)

        # Fix 4: Invalid await syntax
        elif "await" in error_msg or "'await' outside" in error_msg:
            return self.fix_await_syntax(lines, line_num, fixes_applied)

        # Fix 5: Missing colons
        elif "invalid syntax" in error_msg and ":" in line:
            return self.fix_missing_colon(lines, line_num, fixes_applied)

        return False

    def fix_missing_body(self, lines: list[str], line_num: int, fixes_applied: list[str]) -> bool:
        """Add 'pass' to functions/classes with missing bodies."""
        if line_num >= len(lines):
            return False

        # Look for the line that needs a body (function, class, if, for, etc.)
        target_line = line_num
        while target_line >= 0:
            line = lines[target_line].strip()
            if line.endswith(':') and any(
                line.startswith(keyword)
                for keyword in [
                    'def ',
                    'class ',
                    'if ',
                    'for ',
                    'while ',
                    'try:',
                    'except',
                    'else:',
                    'elif ',
                    'with ',
                    'async def ',
                ]
            ):
                # Get the indentation of the definition line
                indent = len(lines[target_line]) - len(lines[target_line].lstrip())
                # Add pass with proper indentation
                pass_line = ' ' * (indent + 4) + 'pass\n'
                lines.insert(target_line + 1, pass_line)
                fixes_applied.append("added 'pass' statement")
                return True
            target_line -= 1

        return False

    def fix_unexpected_indent(self, lines: list[str], line_num: int, fixes_applied: list[str]) -> bool:
        """Fix unexpected indentation."""
        if line_num >= len(lines) or line_num == 0:
            return False

        current_line = lines[line_num]

        # Get indentation of previous non-empty line
        prev_line_num = line_num - 1
        while prev_line_num >= 0 and lines[prev_line_num].strip() == '':
            prev_line_num -= 1

        if prev_line_num < 0:
            # Remove all leading whitespace if it's the first line
            lines[line_num] = current_line.lstrip()
            fixes_applied.append("removed unexpected indentation")
            return True

        prev_line = lines[prev_line_num]
        prev_indent = len(prev_line) - len(prev_line.lstrip())

        # If previous line ends with colon, indent properly
        if prev_line.strip().endswith(':'):
            expected_indent = prev_indent + 4
        else:
            expected_indent = prev_indent

        # Fix the indentation
        content = current_line.lstrip()
        lines[line_num] = ' ' * expected_indent + content
        fixes_applied.append("fixed indentation")
        return True

    def fix_unmatched_characters(self, lines: list[str], line_num: int, fixes_applied: list[str]) -> bool:
        """Fix unmatched braces, brackets, or quotes."""
        if line_num >= len(lines):
            return False

        line = lines[line_num]

        # Remove unmatched closing braces/brackets
        cleaned_line = re.sub(r'[}\])](?!["\'])', '', line)
        if cleaned_line != line:
            lines[line_num] = cleaned_line
            fixes_applied.append("removed unmatched closing characters")
            return True

        # Try to balance quotes
        if line.count('"') % 2 == 1:
            lines[line_num] = line.rstrip() + '"\n'
            fixes_applied.append("balanced quotes")
            return True

        if line.count("'") % 2 == 1:
            lines[line_num] = line.rstrip() + "'\n"
            fixes_applied.append("balanced quotes")
            return True

        return False

    def fix_await_syntax(self, lines: list[str], line_num: int, fixes_applied: list[str]) -> bool:
        """Fix invalid await syntax by commenting out problematic lines."""
        if line_num >= len(lines):
            return False

        line = lines[line_num]

        # Check if we're inside an async function
        in_async_function = False
        for i in range(line_num - 1, -1, -1):
            prev_line = lines[i].strip()
            if prev_line.startswith('async def '):
                in_async_function = True
                break
            elif prev_line.startswith('def ') and not prev_line.startswith('async def '):
                break

        if not in_async_function and 'await' in line:
            # Comment out the line with await
            indent = len(line) - len(line.lstrip())
            lines[line_num] = ' ' * indent + '# ' + line.lstrip()
            fixes_applied.append("commented out invalid await")
            return True

        return False

    def fix_missing_colon(self, lines: list[str], line_num: int, fixes_applied: list[str]) -> bool:
        """Fix missing colons in control structures."""
        if line_num >= len(lines):
            return False

        line = lines[line_num].rstrip()

        # Check if this looks like a control structure missing a colon
        keywords = ['if ', 'elif ', 'else', 'for ', 'while ', 'def ', 'class ', 'try', 'except', 'finally', 'with ']

        for keyword in keywords:
            if line.strip().startswith(keyword) and not line.endswith(':'):
                lines[line_num] = line + ':\n'
                fixes_applied.append("added missing colon")
                return True

        return False


def main():
    fixer = SyntaxErrorFixer()
    jeffrey_dir = "src/jeffrey"

    if not os.path.exists(jeffrey_dir):
        print(f"Directory {jeffrey_dir} not found!")
        return

    print("🔍 Scanning for Python files with syntax errors...")
    files_with_errors = fixer.get_files_with_errors(jeffrey_dir)

    print(f"\n📊 Found {len(files_with_errors)} files with syntax errors")

    if not files_with_errors:
        print("✅ No syntax errors found!")
        return

    print("\n🔧 Starting syntax error fixing process...")

    fixed_count = 0
    failed_count = 0

    for file_path, error_desc in files_with_errors:
        print(f"\n{'=' * 60}")
        print(f"File: {file_path}")
        print(f"Error: {error_desc}")

        if fixer.fix_file(file_path):
            fixed_count += 1
        else:
            failed_count += 1

    print(f"\n{'=' * 60}")
    print("🎯 SUMMARY")
    print(f"✅ Fixed: {fixed_count} files")
    print(f"❌ Failed: {failed_count} files")
    print(f"🛠️  Total fixes applied: {len(fixer.fixes_applied)}")

    if fixer.fixes_applied:
        print("\n📝 Types of fixes applied:")
        fix_types = {}
        for fix in fixer.fixes_applied:
            fix_types[fix] = fix_types.get(fix, 0) + 1

        for fix_type, count in fix_types.items():
            print(f"  • {fix_type}: {count} times")

    # Final verification
    print("\n🔍 Final verification...")
    remaining_errors = fixer.get_files_with_errors(jeffrey_dir)

    if not remaining_errors:
        print("🎉 ALL FILES ARE NOW SYNTACTICALLY VALID!")
    else:
        print(f"⚠️  {len(remaining_errors)} files still have errors:")
        for file_path, error_desc in remaining_errors[:10]:  # Show first 10
            print(f"  • {file_path}: {error_desc}")
        if len(remaining_errors) > 10:
            print(f"  ... and {len(remaining_errors) - 10} more")


if __name__ == "__main__":
    main()
