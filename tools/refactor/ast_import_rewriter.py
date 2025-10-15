# tools/refactor/ast_import_rewriter.py
import hashlib

import libcst as cst


class ImportRewriter(cst.CSTTransformer):
    """Réécrit les imports via libcst en préservant les commentaires/formatage."""

    JEFFREY_MAPPINGS = {
        "Orchestrateur_IA": "src.jeffrey",
        "Jeffrey_DEV": "src.jeffrey",
        "Jeffrey_Phoenix": "src.jeffrey",
        "guardian_communication": "src.jeffrey.core.security.guardian",
    }

    CONDITIONAL_MAPPINGS = {
        "core": "src.jeffrey.core",
        "emotions": "src.jeffrey.emotions",
        "memory": "src.jeffrey.memory",
        "consciousness": "src.jeffrey.consciousness",
        "orchestration": "src.jeffrey.orchestration",
    }

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.is_jeffrey_file = "src/jeffrey" in file_path or "/jeffrey/" in file_path

    def _name_to_str(self, name_expr: cst.BaseExpression) -> str:
        return name_expr.code  # robuste pour QualifiedName, Attribute, etc.

    def leave_ImportFrom(self, original: cst.ImportFrom, updated: cst.ImportFrom) -> cst.ImportFrom:
        if updated.relative or not updated.module:
            return updated
        module_str = updated.module.code

        for old, new in self.JEFFREY_MAPPINGS.items():
            if module_str.startswith(old):
                return updated.with_changes(module=cst.parse_expression(module_str.replace(old, new, 1)))

        if self.is_jeffrey_file:
            for old, new in self.CONDITIONAL_MAPPINGS.items():
                if module_str == old or module_str.startswith(old + "."):
                    return updated.with_changes(module=cst.parse_expression(module_str.replace(old, new, 1)))
        return updated

    def leave_Import(self, original: cst.Import, updated: cst.Import) -> cst.Import:
        new_aliases = []
        changed = False
        for alias in updated.names:
            if isinstance(alias, cst.ImportAlias):
                name_str = alias.name.code
                for old, new in self.JEFFREY_MAPPINGS.items():
                    if name_str.startswith(old):
                        new_aliases.append(alias.with_changes(name=cst.parse_expression(name_str.replace(old, new, 1))))
                        changed = True
                        break
                else:
                    if self.is_jeffrey_file:
                        for old, new in self.CONDITIONAL_MAPPINGS.items():
                            if name_str == old or name_str.startswith(old + "."):
                                new_aliases.append(
                                    alias.with_changes(name=cst.parse_expression(name_str.replace(old, new, 1)))
                                )
                                changed = True
                                break
                    if not changed:
                        new_aliases.append(alias)
            else:
                new_aliases.append(alias)
        return updated.with_changes(names=new_aliases) if changed else updated


def rewrite_imports(filepath: str) -> bool:
    """Réécrit les imports d'un fichier via libcst"""
    try:
        with open(filepath, encoding='utf-8') as f:
            content = f.read()

        # Hash avant modification
        original_hash = hashlib.md5(content.encode()).hexdigest()

        # Parse avec libcst
        module = cst.parse_module(content)

        # Transformer
        transformer = ImportRewriter(filepath)
        new_module = module.visit(transformer)

        # Reconvertir en code
        new_content = new_module.code

        # Hash après
        new_hash = hashlib.md5(new_content.encode()).hexdigest()

        # Écrire seulement si changé
        if original_hash != new_hash:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(new_content)
            return True
        return False
    except Exception as e:
        print(f"Error rewriting {filepath}: {e}")
        return False
