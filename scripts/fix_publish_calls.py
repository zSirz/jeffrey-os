#!/usr/bin/env python3
"""
Fix publish calls in loops to use correct signature
"""

from pathlib import Path


def fix_publish_calls():
    """Fix publish calls to use (event, data) signature"""
    fixes = 0

    # Loop files that need fixing
    loop_files = [
        "src/jeffrey/core/loops/awareness.py",
        "src/jeffrey/core/loops/curiosity.py",
        "src/jeffrey/core/loops/memory_consolidation.py",
        "src/jeffrey/core/loops/emotional_decay.py",
        "src/jeffrey/core/loops/loop_manager.py",
    ]

    for filepath in loop_files:
        file = Path(filepath)
        if not file.exists():
            continue

        content = file.read_text()
        original = content

        # Fix pattern: await self.bus.publish({ -> await self.bus.publish('loop.event', {
        if "await self.bus.publish({" in content:
            # Extract loop name from filename
            loop_name = file.stem.replace("_", ".")
            if loop_name == "loop.manager":
                loop_name = "system"

            # Replace single-arg publish with two-arg
            content = content.replace("await self.bus.publish({", f"await self.bus.publish('{loop_name}.event', {{")
            fixes += 1

        if content != original:
            file.write_text(content)
            print(f"✅ Fixed {file.name}")

    return fixes


if __name__ == "__main__":
    fixes = fix_publish_calls()
    print(f"\nFixed {fixes} files")
