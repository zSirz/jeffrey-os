#!/usr/bin/env python3
"""
Systematic Python syntax fixer for Jeffrey OS.
Fixes common indentation issues and missing pass statements.
"""

import re
import subprocess
import sys
from pathlib import Path


class SyntaxFixer:
    def __init__(self):
        self.fixes_applied = []

    def fix_indentation_issues(self, content: str) -> str:
        """Fix common indentation patterns."""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            original_line = line

            # Fix method definitions that are over-indented
            if re.match(r'^[ ]{8,}def ', line):
                line = re.sub(r'^[ ]{8,}def ', '    def ', line)

            # Fix class definitions that are over-indented
            if re.match(r'^[ ]{4,}class ', line):
                line = re.sub(r'^[ ]{4,}class ', 'class ', line)

            # Fix misplaced pass statements after docstrings
            if 'pass' in line and i > 0:
                prev_line = lines[i - 1].strip()
                if prev_line.endswith('"""') or prev_line.endswith("'''"):
                    # Remove standalone pass after docstring
                    if line.strip() == 'pass':
                        continue
                    # Remove pass from beginning of line
                    line = re.sub(r'^(\s*)pass\s*', r'\1', line)

            # Fix return statements that are over-indented
            if re.match(r'^[ ]{12,}return ', line):
                line = re.sub(r'^[ ]{12,}return ', '        return ', line)

            # Fix if statements that are over-indented
            if re.match(r'^[ ]{12,}if ', line):
                line = re.sub(r'^[ ]{12,}if ', '        if ', line)

            # Fix for statements that are over-indented
            if re.match(r'^[ ]{12,}for ', line):
                line = re.sub(r'^[ ]{12,}for ', '        for ', line)

            # Fix try statements that are over-indented
            if re.match(r'^[ ]{12,}try:', line):
                line = re.sub(r'^[ ]{12,}try:', '        try:', line)

            # Fix except statements that are over-indented
            if re.match(r'^[ ]{12,}except', line):
                line = re.sub(r'^[ ]{12,}except', '        except', line)

            if line != original_line:
                self.fixes_applied.append(f"Line {i + 1}: Indentation fix")

            fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def fix_empty_functions(self, content: str) -> str:
        """Add pass statements to empty function bodies."""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            fixed_lines.append(line)

            # Check if this is a function/method definition
            if re.match(r'^\s*def ', line) and line.strip().endswith(':'):
                # Check if next line is not indented properly or is another definition
                if i + 1 < len(lines):
                    next_line = lines[i + 1]
                    # If next line doesn't have proper indentation or is empty/comment
                    if (
                        not next_line.strip()
                        or not next_line.startswith(' ' * 8)
                        or next_line.strip().startswith(('def ', 'class ', '#'))
                    ):
                        # Get the indentation of the function
                        func_indent = len(line) - len(line.lstrip())
                        body_indent = ' ' * (func_indent + 4)

                        # Add pass statement
                        fixed_lines.append(f'{body_indent}pass')
                        self.fixes_applied.append(f"Line {i + 1}: Added pass statement")

        return '\n'.join(fixed_lines)

    def fix_try_except_syntax(self, content: str) -> str:
        """Fix invalid try/except syntax."""
        lines = content.split('\n')
        fixed_lines = []

        for i, line in enumerate(lines):
            # Fix bare try statements
            if re.match(r'^\s*try:\s*$', line) and i + 1 < len(lines):
                next_line = lines[i + 1]
                if next_line.strip().startswith('except'):
                    # Add pass to try block
                    indent = len(line) - len(line.lstrip())
                    fixed_lines.append(line)
                    fixed_lines.append(' ' * (indent + 4) + 'pass')
                    self.fixes_applied.append(f"Line {i + 1}: Added pass to try block")
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)

        return '\n'.join(fixed_lines)

    def fix_file(self, file_path: Path) -> bool:
        """Fix a single Python file."""
        try:
            with open(file_path, encoding='utf-8') as f:
                original_content = f.read()

            content = original_content

            # Apply all fixes
            content = self.fix_indentation_issues(content)
            content = self.fix_empty_functions(content)
            content = self.fix_try_except_syntax(content)

            if content != original_content:
                # Write the fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                return True

            return False

        except Exception as e:
            print(f"Error fixing {file_path}: {e}")
            return False

    def check_syntax(self, file_path: Path) -> bool:
        """Check if a Python file has valid syntax."""
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'py_compile', str(file_path)], capture_output=True, text=True
            )
            return result.returncode == 0
        except:
            return False


def get_error_files() -> list[Path]:
    """Get list of files with syntax errors from the previous check."""
    error_files = [
        "context_analyzer.py",
        "inner_weather_analyzer.py",
        "empathic_trigger_analyzer.py",
        "silence_sense_analyzer.py",
        "src/jeffrey/core/memory/jeffrey_human_memory.py",
        "src/jeffrey/core/memory/sensory/jeffrey_sensory_memory.py",
        "src/jeffrey/core/memory/sensory/sensorial_memory.py",
        "src/jeffrey/core/memory/sync/jeffrey_memory_sync.py",
        "src/jeffrey/core/learning/jeffrey_deep_learning.py",
        "src/jeffrey/core/personality/personality_profile.py",
        "src/jeffrey/core/personality/conversation_personality.py",
        "src/jeffrey/core/personality/adaptive_personality_engine.py",
        "src/jeffrey/core/consciousness/jeffrey_chat_integration.py",
        "src/jeffrey/core/consciousness/jeffrey_dream_system.py",
        "src/jeffrey/core/consciousness/jeffrey_living_consciousness.py",
        "src/jeffrey/core/consciousness/jeffrey_work_interface.py",
        "src/jeffrey/core/consciousness/jeffrey_living_memory.py",
        "src/jeffrey/core/consciousness/jeffrey_secret_diary.py",
        "src/jeffrey/core/consciousness/jeffrey_living_expressions.py",
        "src/jeffrey/core/orchestration/jeffrey_system_health.py",
        "src/jeffrey/core/orchestration/jeffrey_optimizer.py",
        "src/jeffrey/core/orchestration/jeffrey_continuel.py",
        "src/jeffrey/core/orchestration/enhanced_orchestrator.py",
        "src/jeffrey/core/emotions/core/jeffrey_curiosity_engine.py",
        "src/jeffrey/core/emotions/core/jeffrey_intimate_mode.py",
        "src/jeffrey/core/emotions/visuals/jeffrey_emotional_display.py",
        "src/jeffrey/core/emotions/visuals/jeffrey_visual_emotions.py",
        "src/jeffrey/infrastructure/monitoring/auto_scaler.py",
        "src/jeffrey/infrastructure/monitoring/event_logger.py",
        "src/jeffrey/infrastructure/monitoring/health_checker.py",
        "src/jeffrey/infrastructure/monitoring/benchmarking/collectors/stability_collector.py",
        "src/jeffrey/services/voice/adapters/voice_emotion_adapter.py",
        "src/jeffrey/services/voice/effects/voice_effects.py",
        "src/jeffrey/services/voice/engine/elevenlabs_client.py",
        "src/jeffrey/services/voice/engine/voice_recognition_error_recovery.py",
        "src/jeffrey/services/voice/engine/jeffrey_voice_system.py",
        "src/jeffrey/services/voice/engine/elevenlabs_v3_engine.py",
        "src/jeffrey/services/voice/engine/streaming_audio_pipeline.py",
        "src/jeffrey/services/sync/emotional_prosody_synchronizer.py",
        "src/jeffrey/services/sync/face_sync_manager.py",
        "src/jeffrey/services/sync/interpersonal_rhythm_synchronizer.py",
        "src/jeffrey/interfaces/ui/chat/chat_screen.py",
        "src/jeffrey/interfaces/ui/dashboard/dashboard.py",
        "src/jeffrey/interfaces/ui/dashboard/dashboard_premium.py",
        "src/jeffrey/interfaces/ui/widgets/LienAffectifWidget.py",
        "src/jeffrey/interfaces/ui/widgets/JournalEntryCard.py",
        "src/jeffrey/interfaces/ui/console/console_ui.py",
    ]

    jeffrey_os_dir = Path("/Users/davidproz/Desktop/Jeffrey_OS")
    return [jeffrey_os_dir / f for f in error_files if (jeffrey_os_dir / f).exists()]


def main():
    """Main function to fix syntax errors."""
    print("🔧 Starting systematic syntax fixing...")
    print("=" * 60)

    fixer = SyntaxFixer()
    error_files = get_error_files()

    print(f"Found {len(error_files)} files to fix")

    fixed_count = 0
    for i, file_path in enumerate(error_files, 1):
        print(f"[{i:2d}/{len(error_files)}] Fixing {file_path.name}", end=" ... ")

        # Check if it has syntax errors before fixing
        had_errors = not fixer.check_syntax(file_path)

        if had_errors:
            # Try to fix the file
            was_fixed = fixer.fix_file(file_path)

            if was_fixed:
                # Check if syntax is now valid
                is_valid = fixer.check_syntax(file_path)
                if is_valid:
                    print("✅ FIXED")
                    fixed_count += 1
                else:
                    print("⚠️  PARTIAL")
            else:
                print("❌ NO CHANGES")
        else:
            print("✅ ALREADY OK")

    print("\n" + "=" * 60)
    print(f"📊 Fixed {fixed_count} files successfully")

    if fixer.fixes_applied:
        print(f"\n🔧 Fixes applied: {len(fixer.fixes_applied)}")


if __name__ == "__main__":
    main()
