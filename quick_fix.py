#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path.cwd()
DIAGS = [
    Path("diagnostics/analyze_compilation.py"),
    Path("diagnostics/analyze_async.py"),
    Path("diagnostics/analyze_imports.py"),
]

SAFE_HELPER = """
# --- injected helper: safe relative path ---
from pathlib import Path as _Path
__CWD = _Path.cwd().resolve()
def _safe_rel(p) -> str:
    try:
        p = p if isinstance(p, _Path) else _Path(p)
        p = p.resolve()
        return str(p.relative_to(__CWD))
    except Exception:
        return str(p)
# --- end helper ---
"""

REPL_PATTERNS = [
    # str(VAR.relative_to(Path.cwd()))
    (
        r"str\(\s*([A-Za-z_][A-Za-z0-9_\.]*)\.relative_to\(\s*(?:Path|pathlib\.Path)\.cwd\(\)\s*\)\s*\)",
        r"_safe_rel(\1)",
    ),
    # VAR.relative_to(Path.cwd())
    (r"([A-Za-z_][A-Za-z0-9_\.]*)\.relative_to\(\s*(?:Path|pathlib\.Path)\.cwd\(\)\s*\)", r"_safe_rel(\1)"),
]


def patch_diagnostic(fp: Path):
    if not fp.exists():
        print(f"⚠️ {fp} introuvable")
        return
    txt = fp.read_text(encoding="utf-8")
    orig = txt

    # inject helper après les imports
    if "_safe_rel(" not in txt:
        lines = txt.splitlines()
        insert_at = 0
        for i, line in enumerate(lines):
            if line.startswith("import ") or line.startswith("from "):
                insert_at = i + 1
        lines.insert(insert_at, SAFE_HELPER.strip("\n"))
        txt = "\n".join(lines)

    # remplace les usages fragiles
    for pat, repl in REPL_PATTERNS:
        txt = re.sub(pat, repl, txt)

    if txt != orig:
        fp.write_text(txt, encoding="utf-8")
        print(f"✅ Patch chemins: {fp}")
    else:
        print(f"ℹ️ Aucun changement: {fp}")


def main():
    print("🔧 Hotfix Jeffrey OS - Chemins seulement")
    for fp in DIAGS:
        patch_diagnostic(fp)
    print("🏁 Hotfix terminé.")


if __name__ == "__main__":
    main()
