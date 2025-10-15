# tools/refactor/plugin_inserter.py
from pathlib import Path

import libcst as cst


class ModuleLevelPluginInserter(cst.CSTTransformer):
    """Insère JEFFREY_PLUGIN au niveau MODULE (pas classe) - Préserve tout"""

    def __init__(self, plugin_config: dict):
        self.plugin_config = plugin_config
        self.inserted = False

    def leave_Module(self, original: cst.Module, updated: cst.Module) -> cst.Module:
        """Insère le plugin au niveau module après imports/docstring"""

        if self.inserted:
            return updated

        # Vérifier si déjà présent
        code = updated.code if hasattr(updated, 'code') else str(original)
        if 'JEFFREY_PLUGIN' in code:
            return updated

        # Créer le code du plugin
        plugin_code = self._create_plugin_assignment()

        # Trouver où insérer (après imports et docstring module)
        body = list(updated.body)
        insert_position = self._find_insert_position(body)

        # Insérer le plugin
        plugin_node = cst.parse_statement(plugin_code)
        body.insert(insert_position, plugin_node)

        # Ajouter une ligne vide pour la lisibilité
        body.insert(insert_position + 1, cst.EmptyLine())

        self.inserted = True
        return updated.with_changes(body=body)

    def _find_insert_position(self, body: list) -> int:
        """Trouve la position d'insertion (après imports/docstring)"""
        position = 0

        for i, node in enumerate(body):
            # Skip les imports
            if isinstance(node, (cst.Import, cst.ImportFrom)):
                position = i + 1
                continue

            # Skip le docstring module (première string littérale)
            if i == 0 and isinstance(node, cst.SimpleStatementLine):
                if node.body and isinstance(node.body[0], cst.Expr):
                    if isinstance(node.body[0].value, cst.SimpleString):
                        position = i + 1
                        continue

            # Skip les lignes vides
            if isinstance(node, cst.EmptyLine):
                continue

            # On a trouvé le premier "vrai" code
            break

        return position

    def _create_plugin_assignment(self) -> str:
        """Crée le code d'assignation du plugin"""

        topics_in = self.plugin_config.get('topics_in', ['*'])
        topics_out = self.plugin_config.get('topics_out', ['processed'])
        handler = self.plugin_config.get('handler', 'process')
        dependencies = self.plugin_config.get('dependencies', [])

        plugin_code = f"""
JEFFREY_PLUGIN = {{
    'topics_in': {topics_in},
    'topics_out': {topics_out},
    'handler': '{handler}',
    'dependencies': {dependencies},
    'ml_config': {{
        'adaptive': True,
        'strength': 1.0,
        'evolution_rate': 0.1
    }}
}}"""
        return plugin_code


def add_plugin_to_file(filepath: Path, plugin_config: dict) -> bool:
    """Ajoute JEFFREY_PLUGIN à un fichier en préservant tout"""

    try:
        # Lire le fichier
        with open(filepath, encoding='utf-8') as f:
            source_code = f.read()

        # Vérifier si plugin déjà présent
        if 'JEFFREY_PLUGIN' in source_code:
            return False

        # Parser avec libcst
        module = cst.parse_module(source_code)

        # Appliquer la transformation
        transformer = ModuleLevelPluginInserter(plugin_config)
        modified = module.visit(transformer)

        # Écrire seulement si modifié
        if transformer.inserted:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified.code)
            return True

    except Exception as e:
        print(f"Error adding plugin to {filepath}: {e}")

    return False
