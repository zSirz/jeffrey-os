"""Audit final avec détection de dead code"""

import ast
import importlib
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "src")

print("🔍 Audit final Jeffrey OS\n")

SRC = Path("src")
skip = ("__pycache__", "vendors", "venv", "tests", "backup", ".backup")

# Scan imports
seen_imports = set()
usage_count = Counter()

for py in SRC.rglob("*.py"):
    if any(s in str(py) for s in skip):
        continue

    try:
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(("core", "jeffrey")):
                    seen_imports.add(node.module)
                    usage_count[node.module] += 1
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(("core", "jeffrey")):
                        seen_imports.add(alias.name)
                        usage_count[alias.name] += 1
    except:
        pass

print(f"📦 {len(seen_imports)} imports uniques\n")

# Test imports
ok_list = []
fail_list = []

for mod in sorted(seen_imports):
    try:
        importlib.import_module(mod)
        ok_list.append(mod)
    except Exception as e:
        fail_list.append((mod, str(e).split("\n")[0], usage_count[mod]))

# Rapport
print("=" * 80)
print("📊 RÉSULTATS AUDIT FINAL")
print("=" * 80)
print(f"\n✅ OK: {len(ok_list)}")
print(f"❌ FAIL: {len(fail_list)}")
taux = (len(ok_list) * 100 // len(seen_imports)) if seen_imports else 0
print(f"📈 Taux: {taux}%")

if fail_list:
    # Trier par usage (dead code = usage faible)
    fail_list.sort(key=lambda x: x[2], reverse=True)

    print("\n❌ TOP 20 FAIL (trié par usage):\n")
    for i, (mod, err, count) in enumerate(fail_list[:20], 1):
        dead_code = " [DEAD CODE ?]" if count <= 1 else ""
        print(f"{i:2d}. {mod} (usage: {count}){dead_code}")
        print(f"    └─ {err[:100]}")

# Sauvegarder
with open("audit_final.txt", "w") as f:
    f.write(f"OK: {len(ok_list)}\n")
    f.write(f"FAIL: {len(fail_list)}\n")
    f.write(f"Taux: {taux}%\n\n")
    f.write("FAIL détaillé:\n\n")
    for mod, err, count in fail_list:
        f.write(f"{mod} (usage: {count})\n  → {err}\n\n")

print("\n💾 Rapport: audit_final.txt")
