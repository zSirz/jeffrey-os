"""
Fix imports 'from ui.*' → 'from jeffrey.interfaces.ui.*'
Utilise AST pour éviter de modifier strings/commentaires.
"""

import ast
import pathlib


def fix_ui_imports():
    base = pathlib.Path("src")
    fixed_count = 0
    error_count = 0

    for py_file in base.rglob("*.py"):
        try:
            code = py_file.read_text(encoding="utf-8")
            tree = ast.parse(code)
        except SyntaxError:
            # Fichier cassé, on skip (sera corrigé après)
            error_count += 1
            continue

        changed = False

        class ImportRewriter(ast.NodeTransformer):
            def visit_ImportFrom(self, node):
                nonlocal changed
                if node.module and node.module.split(".")[0] == "ui":
                    node.module = "jeffrey.interfaces." + node.module
                    changed = True
                return node

            def visit_Import(self, node):
                nonlocal changed
                for alias in node.names:
                    if alias.name.split(".")[0] == "ui":
                        alias.name = "jeffrey.interfaces." + alias.name
                        changed = True
                return node

        new_tree = ImportRewriter().visit(tree)

        if changed:
            try:
                new_code = ast.unparse(new_tree)
                py_file.write_text(new_code, encoding="utf-8")
                fixed_count += 1
                print(f"✓ Fixed: {py_file.relative_to(base)}")
            except Exception as e:
                print(f"✗ Error writing {py_file}: {e}")
                error_count += 1

    print(f"\n{'=' * 60}")
    print(f"Imports UI normalisés : {fixed_count} fichiers")
    print(f"Erreurs rencontrées : {error_count} fichiers")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    fix_ui_imports()
