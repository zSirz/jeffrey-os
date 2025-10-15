"""
Liste tous les fichiers RED (SyntaxError/IndentationError) dans src/jeffrey/
avec fichier:ligne:message pour todolist de corrections.
"""

import ast
import sys
from pathlib import Path

ROOT = Path("src/jeffrey")


def main():
    red = []

    for py_file in ROOT.rglob("*.py"):
        try:
            code = py_file.read_text(encoding="utf-8")
            ast.parse(code)
        except (SyntaxError, IndentationError) as e:
            lineno = getattr(e, "lineno", "?")
            msg = getattr(e, "msg", str(e))
            red.append((py_file, lineno, msg))
        except Exception:
            # On ignore les autres erreurs (import, etc.)
            pass

    # Trier par fichier et ligne
    red.sort(key=lambda x: (str(x[0]), x[1] if isinstance(x[1], int) else 0))

    print("\n" + "=" * 70)
    print(f"RED (Syntax/Indent) détectés dans src/jeffrey/ : {len(red)}")
    print("=" * 70 + "\n")

    if red:
        for path, lineno, msg in red:
            rel_path = str(path).replace('src/', '')
            print(f"  ❌ {rel_path}:{lineno}")
            print(f"     → {msg}\n")
    else:
        print("✅ Aucun RED détecté !")

    sys.exit(1 if red else 0)


if __name__ == "__main__":
    main()
