#!/usr/bin/env python3
"""
Extraction automatique des contrats d'interface (Gemini + Grok)
Pour chaque module manquant, analyse comment il est utilisé
et génère un contrat d'interface documenté
"""

import ast
import json
from collections import defaultdict
from pathlib import Path


class UsageAnalyzer(ast.NodeVisitor):
    """Analyse comment un module est utilisé."""

    def __init__(self, target_module: str):
        self.target_module = target_module
        self.usages = []
        self.current_context = []

    def visit_Call(self, node):
        """Détecte les appels de fonctions/méthodes."""
        # module.function()
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == self.target_module.split('.')[-1]:
                    self.usages.append(
                        {
                            'type': 'function_call',
                            'method': node.func.attr,
                            'args_count': len(node.args),
                            'has_kwargs': len(node.keywords) > 0,
                            'context': '.'.join(self.current_context),
                        }
                    )

        self.generic_visit(node)

    def visit_Attribute(self, node):
        """Détecte les accès aux attributs."""
        if isinstance(node.value, ast.Name):
            if node.value.id == self.target_module.split('.')[-1]:
                self.usages.append(
                    {'type': 'attribute_access', 'attribute': node.attr, 'context': '.'.join(self.current_context)}
                )

        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        """Track le contexte de fonction."""
        self.current_context.append(node.name)
        self.generic_visit(node)
        self.current_context.pop()

    def visit_ClassDef(self, node):
        """Track le contexte de classe."""
        self.current_context.append(node.name)
        self.generic_visit(node)
        self.current_context.pop()


def find_usage_files(module_name: str) -> list[Path]:
    """Trouve tous les fichiers qui utilisent ce module."""
    usage_files = []

    for search_dir in [Path('src'), Path('services'), Path('core'), Path('unified')]:
        if not search_dir.exists():
            continue

        for py_file in search_dir.rglob('*.py'):
            try:
                with open(py_file, encoding='utf-8') as f:
                    content = f.read()
                    if module_name in content:
                        usage_files.append(py_file)
            except Exception:
                pass

    return usage_files


def analyze_module_usage(module_name: str, usage_files: list[Path]) -> dict:
    """Analyse l'usage d'un module dans plusieurs fichiers."""
    all_usages = []
    methods = defaultdict(int)
    attributes = defaultdict(int)

    for filepath in usage_files:
        try:
            with open(filepath, encoding='utf-8') as f:
                tree = ast.parse(f.read())

            analyzer = UsageAnalyzer(module_name)
            analyzer.visit(tree)

            all_usages.extend(analyzer.usages)

            for usage in analyzer.usages:
                if usage['type'] == 'function_call':
                    methods[usage['method']] += 1
                elif usage['type'] == 'attribute_access':
                    attributes[usage['attribute']] += 1

        except Exception:
            pass

    return {
        'total_usages': len(all_usages),
        'methods': dict(methods),
        'attributes': dict(attributes),
        'usage_files': [str(f) for f in usage_files],
        'detailed_usages': all_usages,
    }


def generate_interface_contract(module_name: str, usage_analysis: dict) -> str:
    """Génère un contrat d'interface documenté."""
    contract = f"""# Interface Contract: {module_name}

## 📊 Analyse d'Usage

- **Fichiers utilisant ce module** : {len(usage_analysis['usage_files'])}
- **Usages totaux** : {usage_analysis['total_usages']}

## 🔧 Méthodes Attendues

"""

    if usage_analysis['methods']:
        for method, count in sorted(usage_analysis['methods'].items(), key=lambda x: x[1], reverse=True):
            contract += f"### `{method}()`\n\n"
            contract += f"- **Appelé** : {count} fois\n"
            contract += "- **Signature attendue** : À déterminer depuis l'usage\n"
            contract += "- **Retour attendu** : À déterminer depuis l'usage\n\n"
    else:
        contract += "_Aucune méthode détectée_\n\n"

    contract += "## 📦 Attributs Attendus\n\n"

    if usage_analysis['attributes']:
        for attr, count in sorted(usage_analysis['attributes'].items(), key=lambda x: x[1], reverse=True):
            contract += f"- `{attr}` (accédé {count} fois)\n"
    else:
        contract += "_Aucun attribut détecté_\n"

    contract += f"""

## 💡 Recommandations d'Implémentation

### Squelette de Base

```python
#!/usr/bin/env python3
\"\"\"
{module_name}
Implémentation basée sur l'analyse des usages
\"\"\"

class {module_name.split('.')[-1].title().replace('_', '')}:
    \"\"\"Classe principale du module.\"\"\"

    def __init__(self):
        \"\"\"Initialisation.\"\"\"
        # TODO: Ajouter les attributs nécessaires
        pass
"""

    for method in usage_analysis['methods'].keys():
        contract += f"""
    def {method}(self, *args, **kwargs):
        \"\"\"
        TODO: Implémenter {method}
        Analysez les fichiers d'usage pour déterminer la signature exacte
        \"\"\"
        raise NotImplementedError("À implémenter")
```

### Étapes de Développement

1. **Analyser les fichiers d'usage** :
"""

    for usage_file in usage_analysis['usage_files'][:5]:
        contract += f"   - `{usage_file}`\n"

    contract += """
2. **Déterminer les signatures exactes** : Regarder les paramètres passés
3. **Implémenter la logique minimale** : Version simple mais fonctionnelle
4. **Tester** : Créer un test basique
5. **Documenter** : Ajouter docstrings claires

### Checklist de Validation

- [ ] Toutes les méthodes attendues sont implémentées
- [ ] Tous les attributs attendus sont définis
- [ ] Un test basique passe
- [ ] La documentation est claire
- [ ] Aucun `NotImplementedError` dans les chemins d'exécution réels
"""

    return contract


def main():
    print("🔍 EXTRACTION DES CONTRATS D'INTERFACE")
    print("=" * 60)
    print()

    # Charger le diagnostic
    try:
        with open('COMPREHENSIVE_DIAGNOSTIC_V2.json') as f:
            report = json.load(f)
    except FileNotFoundError:
        print("❌ Lancez d'abord comprehensive_diagnostic_v2.py")
        return

    missing_without_candidates = report.get('missing_without_candidates', {})

    if not missing_without_candidates:
        print("✅ Aucun module manquant à analyser")
        return

    print(f"📦 {len(missing_without_candidates)} modules à analyser")
    print()

    contracts_dir = Path('interface_contracts')
    contracts_dir.mkdir(exist_ok=True)

    for module_name, info in list(missing_without_candidates.items())[:20]:  # Top 20
        print(f"📝 Analyse : {module_name}...")

        # Trouver les fichiers qui utilisent ce module
        usage_files = find_usage_files(module_name)

        if not usage_files:
            print("   ⚠️  Aucun usage trouvé (peut-être obsolète)")
            continue

        # Analyser l'usage
        usage_analysis = analyze_module_usage(module_name, usage_files)

        # Générer le contrat
        contract = generate_interface_contract(module_name, usage_analysis)

        # Sauvegarder
        contract_file = contracts_dir / f"{module_name.replace('.', '_')}.md"
        contract_file.write_text(contract)

        print(f"   ✅ Contrat généré : {contract_file}")
        print(f"      {usage_analysis['total_usages']} usages, {len(usage_analysis['methods'])} méthodes")

    print()
    print("=" * 60)
    print("✅ CONTRATS GÉNÉRÉS")
    print("=" * 60)
    print("📁 Dossier : interface_contracts/")
    print()
    print("💡 Pour chaque module à recréer :")
    print("   1. Consultez son contrat dans interface_contracts/")
    print("   2. Copiez le squelette de code")
    print("   3. Implémentez les méthodes selon l'usage réel")


if __name__ == "__main__":
    main()
