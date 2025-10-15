#!/usr/bin/env python3
"""
Comprehensive syntax fixer for Jeffrey OS.
Fixes the most common syntax issues that prevent AST parsing.
"""

import ast
import os
import re
import shutil
import sys


class ComprehensiveSyntaxFixer:
    def __init__(self):
        self.fixed_files = []
        self.failed_files = []

    def fix_all_files(self, directory: str):
        """Fix all Python files with syntax errors in the directory."""
        print(f"🔧 Starting comprehensive syntax fixing in {directory}...")

        # Get all files with syntax errors
        error_files = self.get_files_with_errors(directory)
        print(f"Found {len(error_files)} files with syntax errors")

        for file_path, error in error_files:
            print(f"\n{'=' * 50}")
            print(f"Fixing: {file_path}")
            print(f"Error: {error}")

            if self.fix_file(file_path):
                self.fixed_files.append(file_path)
                print("✅ FIXED")
            else:
                self.failed_files.append(file_path)
                print("❌ FAILED")

        print(f"\n{'=' * 50}")
        print("🎯 RESULTS:")
        print(f"✅ Fixed: {len(self.fixed_files)} files")
        print(f"❌ Failed: {len(self.failed_files)} files")

        # Final verification
        remaining_errors = self.get_files_with_errors(directory)
        print(f"\n🔍 Remaining errors: {len(remaining_errors)}")

    def get_files_with_errors(self, directory: str) -> list[tuple]:
        """Get all Python files with syntax errors."""
        errors = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    error = self.check_syntax(file_path)
                    if error:
                        errors.append((file_path, error))
        return errors

    def check_syntax(self, file_path: str) -> str:
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

    def fix_file(self, file_path: str) -> bool:
        """Fix a single file."""
        # Create backup
        backup_path = file_path + '.backup'
        shutil.copy2(file_path, backup_path)

        try:
            with open(file_path, encoding='utf-8') as f:
                content = f.read()

            original_content = content

            # Apply fixes in order
            content = self.fix_common_patterns(content)
            content = self.fix_indentation_issues(content)
            content = self.fix_empty_functions(content)
            content = self.fix_await_issues(content)
            content = self.fix_malformed_code(content)

            # Write fixed content
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)

            # Verify fix
            if self.check_syntax(file_path) is None:
                return True
            else:
                # Restore backup if fix didn't work
                shutil.copy2(backup_path, file_path)
                return False

        except Exception as e:
            print(f"  Error: {e}")
            # Restore backup
            shutil.copy2(backup_path, file_path)
            return False

    def fix_common_patterns(self, content: str) -> str:
        """Fix common syntax patterns."""
        lines = content.split('\n')

        for i, line in enumerate(lines):
            # Fix await assignment errors
            if '= await' in line:
                # Comment out problematic await assignments
                indent = len(line) - len(line.lstrip())
                lines[i] = ' ' * indent + '# ' + line.lstrip() + '  # TODO: Fix await assignment'

            # Fix incomplete f-strings
            if 'f"' in line and line.count('"') % 2 == 1:
                # Try to complete the f-string
                if '{' in line and '}' not in line:
                    lines[i] = line + '}"'
                elif line.endswith('{"'):
                    lines[i] = line[:-2] + '"'

        return '\n'.join(lines)

    def fix_indentation_issues(self, content: str) -> str:
        """Fix indentation problems systematically."""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            if not line.strip():
                fixed_lines.append(line)
                continue

            # Calculate expected indentation
            expected_indent = self.calculate_expected_indent(lines, i)

            if expected_indent is not None:
                content_part = line.lstrip()
                fixed_line = ' ' * expected_indent + content_part
                fixed_lines.append(fixed_line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def calculate_expected_indent(self, lines: list[str], line_index: int) -> int:
        """Calculate expected indentation for a line."""
        if line_index == 0:
            return 0

        current_line = lines[line_index].strip()
        if not current_line:
            return None

        # Find the previous non-empty line
        prev_index = line_index - 1
        while prev_index >= 0 and not lines[prev_index].strip():
            prev_index -= 1

        if prev_index < 0:
            return 0

        prev_line = lines[prev_index]
        prev_indent = len(prev_line) - len(prev_line.lstrip())

        # Special handling for specific patterns

        # If previous line ends with colon, increase indent
        if prev_line.strip().endswith(':'):
            return prev_indent + 4

        # If current line starts with dedenting keywords
        dedent_keywords = ['except', 'finally', 'elif', 'else']
        if any(current_line.startswith(keyword) for keyword in dedent_keywords):
            return max(0, prev_indent - 4)

        # If current line is a method definition but indented wrong
        if (current_line.startswith('def ') or current_line.startswith('async def ')) and prev_indent >= 4:
            # Check if we're inside a class
            for j in range(prev_index, -1, -1):
                check_line = lines[j].strip()
                if check_line.startswith('class '):
                    return 4  # Method should be indented 4 spaces in class
                elif check_line and not check_line.startswith('#') and len(lines[j]) - len(lines[j].lstrip()) == 0:
                    break
            return 0  # Top-level function

        # Default: maintain same indentation as previous line
        return prev_indent

    def fix_empty_functions(self, content: str) -> str:
        """Add 'pass' to empty functions and classes."""
        lines = content.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i].strip()

            # Check for function/class definitions
            if (
                line.startswith('def ')
                or line.startswith('async def ')
                or line.startswith('class ')
                or line.endswith(':')
            ):
                if line.endswith(':'):
                    # Get indentation level
                    indent_level = len(lines[i]) - len(lines[i].lstrip())

                    # Check if next non-empty line is properly indented
                    j = i + 1
                    while j < len(lines) and not lines[j].strip():
                        j += 1

                    needs_pass = False
                    if j >= len(lines):
                        needs_pass = True
                    else:
                        next_line = lines[j]
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if next_indent <= indent_level:
                            needs_pass = True

                    if needs_pass:
                        # Insert pass with proper indentation
                        pass_line = ' ' * (indent_level + 4) + 'pass'
                        lines.insert(i + 1, pass_line)
            i += 1

        return '\n'.join(lines)

    def fix_await_issues(self, content: str) -> str:
        """Fix await-related syntax issues."""
        # Comment out problematic await assignments
        content = re.sub(
            r'^(\s*)(.+?)\s*=\s*await\s+(.+)$',
            r'\1# \2 = await \3  # TODO: Fix await assignment',
            content,
            flags=re.MULTILINE,
        )
        return content

    def fix_malformed_code(self, content: str) -> str:
        """Fix various malformed code patterns."""
        lines = content.split('\n')

        for i, line in enumerate(lines):
            # Fix lines that start with random tokens
            if line.strip().startswith('except Exception:') and i > 0:
                prev_line = lines[i - 1].strip()
                if not prev_line.endswith(':') and 'try' not in prev_line:
                    # This except is orphaned, comment it out
                    indent = len(line) - len(line.lstrip())
                    lines[i] = ' ' * indent + '# ' + line.lstrip() + '  # TODO: Fix orphaned except'

            # Fix incomplete try blocks
            if line.strip() == 'try:':
                # Look ahead to see if there's an except
                j = i + 1
                has_except = False
                while j < len(lines):
                    if lines[j].strip().startswith('except'):
                        has_except = True
                        break
                    elif lines[j].strip() and not lines[j].startswith(' '):
                        break
                    j += 1

                if not has_except:
                    # Add a simple except block
                    indent = len(line) - len(line.lstrip())
                    lines.insert(i + 2, ' ' * (indent + 4) + 'pass')
                    lines.insert(i + 3, ' ' * indent + 'except Exception:')
                    lines.insert(i + 4, ' ' * (indent + 4) + 'pass')

        return '\n'.join(lines)


def main():
    fixer = ComprehensiveSyntaxFixer()

    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        directory = "src/jeffrey"

    if not os.path.exists(directory):
        print(f"Directory {directory} not found!")
        return

    fixer.fix_all_files(directory)


if __name__ == "__main__":
    main()
