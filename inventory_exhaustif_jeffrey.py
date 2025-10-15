#!/usr/bin/env python3
"""
INVENTAIRE EXHAUSTIF JEFFREY OS - CLAUDE CODE

Script d'analyse complète pour identifier l'état des modules Python :
- GREEN : Compile + Importable depuis le repo
- AMBER : Erreurs d'indentation (réparables facilement)
- AMBER_IMPORT : Compile mais pas importable depuis src/
- RED : Erreurs de syntaxe profondes (nécessite réécriture)

Usage:
    export INVENTORY_TAG="claude"
    python3 inventory_exhaustif_jeffrey.py
"""

import hashlib
import importlib.util
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ============ CONFIGURATION ============
ROOT = Path.cwd()
SRC = ROOT / "src"
REPORTS = ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)

TAG = os.environ.get("INVENTORY_TAG", "solo")
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT_PREFIX = f"inventory_max_{TAG}_{STAMP}"

# Chemins à scanner
HOME = Path.home()
ICLOUD = HOME / "Library/Mobile Documents/com~apple~CloudDocs"

SCAN_PATHS = [ROOT]
if ICLOUD.exists():
    SCAN_PATHS.append(ICLOUD)


# Patterns à ignorer (amélioré pour éviter les faux positifs)
def is_skipped(path: Path):
    """Test amélioré pour éviter d'exclure des fichiers légitimes"""
    s = path.as_posix().lower()
    parts = [p.lower() for p in path.parts]

    # Dossiers à ignorer par segment
    skip_dirs = {"__pycache__", ".git", "venv", ".venv", "node_modules", "reports"}
    if any(p in skip_dirs for p in parts):
        return True

    # Exclure uniquement dossiers backup* complets, pas les noms de fichier
    if any(seg.startswith(("backup", "backups_")) for seg in parts):
        return True

    # Exclure archives (rare en .py, mais on garde par sécurité)
    if any(x in s for x in (".tar", ".gz", ".zip")):
        return True

    return False


# ============ HELPERS ============
def safe_read_text(p: Path):
    """Lecture sécurisée avec plusieurs encodages"""
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            return p.read_text(encoding=enc)
        except Exception:
            pass
    return None


def safe_read_bytes(p: Path):
    """Lecture sécurisée en bytes"""
    try:
        return p.read_bytes()
    except Exception:
        return None


def sha1_file(p: Path):
    """Calcul du hash SHA1 du fichier"""
    data = safe_read_bytes(p)
    if data:
        return hashlib.sha1(data).hexdigest()
    return ""


def count_loc(text):
    """Compte les lignes de code (sans commentaires ni lignes vides)"""
    if not text:
        return 0
    return sum(1 for line in text.splitlines() if line.strip() and not line.strip().startswith("#"))


def test_compile(text, path):
    """Test de compilation Python"""
    try:
        compile(text, str(path), "exec")
        return True, ""
    except IndentationError as e:
        return False, f"IndentationError L{e.lineno}: {e.msg}"
    except SyntaxError as e:
        return False, f"SyntaxError L{e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:100]}"


def repo_importable(py_path: Path) -> tuple[bool, str]:
    """Teste l'importabilité du fichier *s'il est sous SRC* en reconstruisant son module_name."""
    try:
        rel = py_path.relative_to(SRC)
    except ValueError:
        return False, "outside_repo"

    module_name = ".".join(rel.with_suffix("").parts)
    try:
        spec = importlib.util.find_spec(module_name)
        return (spec is not None), ("" if spec else "Module spec not found")
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:100]}"


def path_score(p):
    """Score de priorité du chemin (plus haut = meilleur)"""
    s = str(p).lower()
    score = 100

    # Pénalités
    bad = ["backup", "backups_", "archive", "archives", ".tar", ".gz", ".zip"]
    for b in bad:
        if b in s:
            score -= 50

    # Bonus
    if "/src/jeffrey/core/" in s:
        score += 30
    elif "/src/jeffrey/" in s:
        score += 20
    elif "/src/" in s:
        score += 10

    if ICLOUD in p.parents and "backup" not in s:
        score += 15  # iCloud original (non backup)

    return score


def categorize(compiles, error, importable, import_err):
    """Catégorisation raffinée avec test d'import"""
    if compiles and importable:
        return "GREEN"
    elif (not compiles) and "IndentationError" in (error or ""):
        return "AMBER"  # facile à corriger (indent)
    elif compiles and not importable:
        return "AMBER_IMPORT"  # compile mais pas importable depuis src/
    else:
        return "RED"


# ============ SCAN ============
print("🔍 Inventaire exhaustif Jeffrey OS")
print(f"📁 Scan : {len(SCAN_PATHS)} répertoires")
print("-" * 70)

all_files = []
hash_to_paths = defaultdict(list)

for base in SCAN_PATHS:
    if not base.exists():
        continue
    print(f"⏳ Scan de {base}...")

    count = 0
    for py_file in base.rglob("*.py"):
        if is_skipped(py_file):
            continue

        count += 1
        if count % 50 == 0:
            print(f"   {count} fichiers analysés...")

        text = safe_read_text(py_file)
        file_hash = sha1_file(py_file)

        if file_hash:
            hash_to_paths[file_hash].append(str(py_file))

        loc = count_loc(text)
        compiles, error = test_compile(text or "", py_file)

        # Test d'import uniquement pour les fichiers du repo
        importable, import_err = (False, "")
        if compiles:
            importable, import_err = repo_importable(py_file)

        category = categorize(compiles, error, importable, import_err)
        score = path_score(py_file)

        all_files.append(
            {
                "path": str(py_file),
                "filename": py_file.name,
                "hash": file_hash,
                "category": category,
                "loc": loc,
                "compiles": compiles,
                "importable_repo": importable,
                "error": error,
                "import_error": import_err,
                "path_score": score,
            }
        )

    print(f"   ✅ {count} fichiers dans {base}")

print(f"\n📊 Total : {len(all_files)} fichiers Python analysés")
print("-" * 70)

# ============ TRAITEMENT ============
# Grouper par nom de fichier
by_filename = defaultdict(list)
for f in all_files:
    by_filename[f["filename"]].append(f)

# Choisir la meilleure version de chaque fichier
chosen = {}
for filename, versions in by_filename.items():
    # Trier : GREEN > AMBER > AMBER_IMPORT > RED, puis par score
    best = max(
        versions,
        key=lambda x: (
            3
            if x["category"] == "GREEN"
            else (2 if x["category"] == "AMBER" else (1 if x["category"] == "AMBER_IMPORT" else 0)),
            x["path_score"],
            x["loc"],
        ),
    )
    chosen[filename] = best

# Catégoriser
greens = [f for f in chosen.values() if f["category"] == "GREEN"]
ambers = [f for f in chosen.values() if f["category"] == "AMBER"]
amber_imports = [f for f in chosen.values() if f["category"] == "AMBER_IMPORT"]
reds = [f for f in chosen.values() if f["category"] == "RED"]

# Identifier doublons
duplicates = {name: paths for name, paths in by_filename.items() if len(paths) > 1}

# Stats
summary = {
    "timestamp": STAMP,
    "tag": TAG,
    "total_files_scanned": len(all_files),
    "unique_filenames": len(chosen),
    "green": len(greens),
    "amber": len(ambers),
    "amber_import": len(amber_imports),
    "red": len(reds),
    "duplicates": len(duplicates),
}


# ============ RAPPORTS ============
def write_json(name, data):
    """Écriture JSON avec encodage UTF-8"""
    path = REPORTS / f"{OUT_PREFIX}__{name}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def write_csv(name, rows, cols):
    """Écriture CSV avec gestion des caractères spéciaux"""
    lines = [",".join(cols)]
    for r in rows:
        vals = []
        for c in cols:
            v = str(r.get(c, ""))
            if "," in v or '"' in v:
                v = '"' + v.replace('"', '""') + '"'
            vals.append(v)
        lines.append(",".join(vals))

    path = REPORTS / f"{OUT_PREFIX}__{name}.csv"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# Sauvegarde
write_json("summary", summary)
write_json("all_files", all_files)
write_json("duplicates", duplicates)

cols = [
    "filename",
    "path",
    "category",
    "loc",
    "path_score",
    "compiles",
    "importable_repo",
    "error",
    "import_error",
    "hash",
]
write_csv("master_mapping", list(chosen.values()), cols)
write_csv("all_candidates", all_files, cols)  # Inventaire exhaustif
write_csv("greens", greens, cols)
write_csv("ambers", ambers, cols)
write_csv("amber_imports", amber_imports, cols)
write_csv("reds", reds, cols)

# Rapport Markdown
md = [
    f"# 📋 INVENTAIRE JEFFREY OS - {TAG}",
    f"\n**Date** : {STAMP}",
    "\n---\n## 📊 RÉSUMÉ GLOBAL\n",
    f"- **Fichiers scannés** : {summary['total_files_scanned']}",
    f"- **Fichiers uniques** : {summary['unique_filenames']}",
    f"- 🟢 **GREEN (Parfaits)** : {summary['green']} ({summary['green'] / summary['unique_filenames'] * 100:.1f}%)",
    f"- 🟠 **AMBER (Réparables)** : {summary['amber']} ({summary['amber'] / summary['unique_filenames'] * 100:.1f}%)",
    f"- 🟡 **AMBER_IMPORT (Compile mais pas importable)** : {summary['amber_import']} ({summary['amber_import'] / summary['unique_filenames'] * 100:.1f}%)",
    f"- 🔴 **RED (Problématiques)** : {summary['red']} ({summary['red'] / summary['unique_filenames'] * 100:.1f}%)",
    f"- 🔄 **Doublons** : {summary['duplicates']}",
    "\n---\n## 🎯 TOP 20 MODULES CRITIQUES À RÉPARER (AMBER)\n",
]

critical_keywords = ["consciousness", "memory", "brain", "orchestr", "decision", "emotion", "core"]
critical_ambers = [f for f in ambers if any(kw in f["path"].lower() for kw in critical_keywords)]
for f in sorted(critical_ambers, key=lambda x: x["path_score"], reverse=True)[:20]:
    md.append(f"- `{f['filename']}` (score: {f['path_score']}, {f['loc']} LOC)")
    md.append(f"  - Path: {f['path']}")
    md.append(f"  - Erreur: *{f['error']}*\n")

md.append("\n---\n## 🟡 TOP 20 MODULES CRITIQUES À CORRIGER IMPORT (AMBER_IMPORT)\n")
critical_amber_imports = [f for f in amber_imports if any(kw in f["path"].lower() for kw in critical_keywords)]
for f in sorted(critical_amber_imports, key=lambda x: x["path_score"], reverse=True)[:20]:
    md.append(f"- `{f['filename']}` (score: {f['path_score']}, {f['loc']} LOC)")
    md.append(f"  - Path: {f['path']}")
    md.append(f"  - Erreur import: *{f['import_error']}*\n")

md.append("\n---\n## 🔥 TOP 20 MODULES CRITIQUES À RÉÉCRIRE (RED)\n")
critical_reds = [f for f in reds if any(kw in f["path"].lower() for kw in critical_keywords)]
for f in sorted(critical_reds, key=lambda x: x["path_score"], reverse=True)[:20]:
    md.append(f"- `{f['filename']}` (score: {f['path_score']}, {f['loc']} LOC)")
    md.append(f"  - Path: {f['path']}")
    md.append(f"  - Erreur: *{f['error']}*\n")

md.append("\n---\n## 🔄 DOUBLONS À RÉSOUDRE\n")
for name, versions in sorted(duplicates.items())[:30]:
    md.append(f"\n### {name} ({len(versions)} versions)")
    for v in versions:
        cat_emoji = (
            "🟢"
            if v["category"] == "GREEN"
            else "🟠"
            if v["category"] == "AMBER"
            else "🟡"
            if v["category"] == "AMBER_IMPORT"
            else "🔴"
        )
        md.append(f"- {cat_emoji} {v['path']} (score: {v['path_score']})")

write_json("report_markdown", "\n".join(md))
(REPORTS / f"{OUT_PREFIX}__REPORT.md").write_text("\n".join(md), encoding="utf-8")

# ============ RÉSUMÉ CONSOLE ============
print("\n" + "=" * 70)
print("✅ INVENTAIRE TERMINÉ")
print("=" * 70)
print(json.dumps(summary, indent=2))
print("\n📄 Fichiers générés dans reports/:")
print(f"   - {OUT_PREFIX}__REPORT.md (à lire en priorité)")
print(f"   - {OUT_PREFIX}__summary.json")
print(f"   - {OUT_PREFIX}__master_mapping.csv (SOURCE DE VÉRITÉ)")
print(f"   - {OUT_PREFIX}__all_candidates.csv (INVENTAIRE EXHAUSTIF)")
print(f"   - {OUT_PREFIX}__greens.csv")
print(f"   - {OUT_PREFIX}__ambers.csv")
print(f"   - {OUT_PREFIX}__amber_imports.csv")
print(f"   - {OUT_PREFIX}__reds.csv")
print("\n🎯 PROCHAINE ÉTAPE :")
print("Envoie-moi ces 5 fichiers pour analyse :")
print(f"   1. {OUT_PREFIX}__summary.json")
print(f"   2. 100 premières lignes de {OUT_PREFIX}__master_mapping.csv")
print(f"   3. 50 premières lignes de {OUT_PREFIX}__ambers.csv")
print(f"   4. 50 premières lignes de {OUT_PREFIX}__amber_imports.csv")
print(f"   5. 50 premières lignes de {OUT_PREFIX}__reds.csv")
