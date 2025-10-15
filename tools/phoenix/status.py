#!/usr/bin/env python3
"""Dashboard de progression Phoenix"""

import ast
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class PhoenixStatus:
    def __init__(self):
        self.stats = {
            'files_total': 0,
            'ast_valid': 0,
            'legacy_imports': 0,
            'plugins_present': 0,
            'plugins_valid': 0,
            'cycles_detected': 0,
        }

    def scan_all(self):
        """Scan complet de src/jeffrey"""
        for py_file in Path('src/jeffrey').rglob('*.py'):
            if '__pycache__' in str(py_file):
                continue

            self.stats['files_total'] += 1

            # Check AST
            try:
                with open(py_file) as f:
                    ast.parse(f.read())
                self.stats['ast_valid'] += 1
            except:
                pass

            # Check imports
            content = py_file.read_text()
            if any(old in content for old in ['Orchestrateur_IA', 'Jeffrey_DEV', 'guardian_communication']):
                self.stats['legacy_imports'] += 1

            # Check plugins
            if 'JEFFREY_PLUGIN' in content:
                self.stats['plugins_present'] += 1
                if all(key in content for key in ["'topics_in'", "'topics_out'", "'handler'"]):
                    self.stats['plugins_valid'] += 1

        # Check cycles avec timeout
        try:
            result = subprocess.run(
                ['python', 'tools/check_import_cycles.py'],
                capture_output=True,
                timeout=30,  # Timeout 30 secondes
            )
            if result.returncode == 1:
                # Parse JSON output au lieu de split fragile
                output = json.loads(result.stdout.decode())
                self.stats['cycles_detected'] = output.get('critical_cycles', 0)
        except subprocess.TimeoutExpired:
            self.stats['cycles_detected'] = -1  # Indique timeout
        except:
            pass

    def generate_report(self):
        """Génère PHOENIX_STATUS.md"""
        valid_pct = (self.stats['ast_valid'] / max(1, self.stats['files_total'])) * 100
        plugin_pct = (self.stats['plugins_valid'] / max(1, self.stats['files_total'])) * 100

        report = f"""# PHOENIX STATUS REPORT
Generated: {datetime.now().isoformat()}

## 📊 Metrics

| Metric | Value | Status |
|--------|-------|--------|
| Total Files | {self.stats['files_total']} | - |
| AST Valid | {self.stats['ast_valid']} ({valid_pct:.1f}%) | {'✅' if valid_pct > 95 else '⚠️'} |
| Legacy Imports | {self.stats['legacy_imports']} | {'❌' if self.stats['legacy_imports'] > 0 else '✅'} |
| Plugins Present | {self.stats['plugins_present']} | - |
| Plugins Valid | {self.stats['plugins_valid']} ({plugin_pct:.1f}%) | {'✅' if plugin_pct > 70 else '⚠️'} |
| Import Cycles | {self.stats['cycles_detected']} | {'❌' if self.stats['cycles_detected'] > 0 else '✅'} |

## 📈 Progress Trend

```
AST:     [{'=' * int(valid_pct / 5)}>{' ' * (20 - int(valid_pct / 5))}] {valid_pct:.0f}%
Plugins: [{'=' * int(plugin_pct / 5)}>{' ' * (20 - int(plugin_pct / 5))}] {plugin_pct:.0f}%
```

## Next Actions

{'- Fix remaining syntax errors' if valid_pct < 100 else ''}
{'- Remove legacy imports' if self.stats['legacy_imports'] > 0 else ''}
{'- Add missing plugins' if plugin_pct < 80 else ''}
{'- Resolve import cycles' if self.stats['cycles_detected'] > 0 else ''}
{'✅ All checks passed!' if valid_pct == 100 and self.stats['legacy_imports'] == 0 and plugin_pct > 80 and self.stats['cycles_detected'] == 0 else ''}
"""

        with open('PHOENIX_STATUS.md', 'w') as f:
            f.write(report)

        return self.stats


if __name__ == "__main__":
    status = PhoenixStatus()
    status.scan_all()
    stats = status.generate_report()
    print("Report generated: PHOENIX_STATUS.md")

    # Exit code 1 si régression
    if stats['ast_valid'] < stats['files_total'] * 0.95:
        sys.exit(1)
