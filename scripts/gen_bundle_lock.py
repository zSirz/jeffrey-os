#!/usr/bin/env python3
"""Générer un bundle lock pour garantir l'immutabilité"""

import hashlib
import json
from datetime import datetime
from pathlib import Path

print("🔒 Generating Bundle 1 lock file...")

inv = json.loads(Path("artifacts/inventory_ultimate.json").read_text())
mods = inv["bundle1_recommendations"]["modules"]
# FIX 3: Éviter le double /8
regions_str = str(inv["bundle1_recommendations"]["regions_covered"])

lock_data = {
    "version": "1.0.0",
    "timestamp": datetime.now().isoformat(),
    "bundle": "bundle1",
    "regions_covered": regions_str,
    "modules": [],
}

for m in mods:
    p = Path(m["path"])
    if p.exists():
        content = p.read_bytes()
        sha256 = hashlib.sha256(content).hexdigest()
        size = len(content)
        lines = content.decode("utf-8", errors="ignore").count("\n")
    else:
        sha256 = None
        size = 0
        lines = 0

    lock_data["modules"].append(
        {
            "name": m["name"],
            "path": m["path"],
            "sha256": sha256,
            "size_bytes": size,
            "lines": lines,
            "region": m.get("region", "unknown"),
        }
    )

    status = "✅" if sha256 else "⚠️"
    print(f"  {status} {m['name']:<30} {lines:>5} lines, {size:>8} bytes")

Path("artifacts/bundle1.lock.json").write_text(json.dumps(lock_data, indent=2), encoding="utf-8")

print("\n✅ Bundle lock created: artifacts/bundle1.lock.json")
print(f"   {len(lock_data['modules'])} modules locked")
print(f"   {sum(1 for m in lock_data['modules'] if m['sha256'])} with valid checksums")
