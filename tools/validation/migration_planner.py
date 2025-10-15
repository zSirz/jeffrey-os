#!/usr/bin/env python3
"""
Migration Planner - Analyse l'inventaire et crée un plan de migration par vagues
"""

import json
from pathlib import Path
from typing import Any


def load_inventory(path: Path) -> dict[str, Any]:
    """Charge l'inventaire JSON"""
    with open(path) as f:
        return json.load(f)


def categorize_by_priority(modules: list[dict]) -> dict[str, list[dict]]:
    """Catégorise les modules par priorité de migration"""
    priorities = {
        "wave_1_critical": [],  # Quantum + IA
        "wave_2_core": [],  # Consciousness + Emotional
        "wave_3_orchestration": [],  # Orchestration + Memory
        "wave_4_standard": [],  # Reste
    }

    for module in modules:
        # Wave 1: Technologies critiques
        if module.get("imports_quantum"):
            priorities["wave_1_critical"].append(module)
        elif module.get("imports_ai"):
            priorities["wave_1_critical"].append(module)
        # Wave 2: Cœur du système
        elif module["type"] in ["consciousness", "emotional", "ethics"]:
            priorities["wave_2_core"].append(module)
        # Wave 3: Infrastructure
        elif module["type"] in ["orchestration", "memory", "learning"]:
            priorities["wave_3_orchestration"].append(module)
        # Wave 4: Standard
        else:
            priorities["wave_4_standard"].append(module)

    return priorities


def generate_migration_plan(inventory: dict[str, Any]) -> str:
    """Génère le plan de migration détaillé"""
    unique_modules = inventory.get("unique_modules", [])
    priorities = categorize_by_priority(unique_modules)

    lines = [
        "# 📋 Plan de Migration Jeffrey Phoenix",
        "\n*Généré automatiquement à partir de l'inventaire*\n",
        "## 🎯 Stratégie de Migration en 4 Vagues\n",
    ]

    # Wave 1 - Critical
    if priorities["wave_1_critical"]:
        lines.extend(
            [
                "### 🔴 Vague 1 - Technologies Critiques (IMMÉDIAT)",
                f"**{len(priorities['wave_1_critical'])} modules avec IA/Quantum**",
                "",
                "Ces modules représentent les capacités les plus avancées de Jeffrey.",
                "Ils doivent être migrés en priorité absolue.",
                "",
                "**Modules à migrer:**",
            ]
        )

        for module in priorities["wave_1_critical"][:10]:
            tech = "⚛️ Quantum" if module.get("imports_quantum") else "🤖 IA"
            lines.append(f"- {tech} `{module['env']}/{module['path']}`")
            lines.append(f"  - Type: {module['type']}")
            lines.append(f"  - Taille: {module['lines']} lignes")

        if len(priorities["wave_1_critical"]) > 10:
            lines.append(f"\n*... et {len(priorities['wave_1_critical']) - 10} autres*")
        lines.append("")

    # Wave 2 - Core
    if priorities["wave_2_core"]:
        lines.extend(
            [
                "### 🟠 Vague 2 - Cœur du Système (Cette semaine)",
                f"**{len(priorities['wave_2_core'])} modules émotionnels/conscience**",
                "",
                "Le cœur émotionnel et conscient de Jeffrey.",
                "",
                "**Modules à migrer:**",
            ]
        )

        for module in priorities["wave_2_core"][:10]:
            emoji = {"consciousness": "🧠", "emotional": "❤️", "ethics": "⚖️"}.get(module['type'], "📦")
            lines.append(f"- {emoji} `{module['env']}/{module['path']}`")
        lines.append("")

    # Wave 3 - Orchestration
    if priorities["wave_3_orchestration"]:
        lines.extend(
            [
                "### 🟡 Vague 3 - Infrastructure (Semaine prochaine)",
                f"**{len(priorities['wave_3_orchestration'])} modules d'orchestration**",
                "",
                "**Aperçu:**",
            ]
        )

        for module in priorities["wave_3_orchestration"][:5]:
            lines.append(f"- `{module['path'][:60]}...`")
        lines.append("")

    # Wave 4 - Standard
    if priorities["wave_4_standard"]:
        lines.extend(
            [
                "### 🟢 Vague 4 - Modules Standard (Plus tard)",
                f"**{len(priorities['wave_4_standard'])} modules restants**",
                "",
            ]
        )

    # Instructions de migration
    lines.extend(
        [
            "## 📝 Instructions de Migration",
            "",
            "Pour chaque module à migrer:",
            "",
            "1. **Copier** le module depuis son environnement source",
            "2. **Placer** dans `unified/modules/[type]/`",
            "3. **Ajouter** au registre `unified/config/module_registry.json`",
            "4. **Tester** via la Champions Facade",
            "5. **Committer** avec message descriptif",
            "",
            "## 📊 Résumé",
            "",
            f"- **Total à migrer:** {len(unique_modules)} modules",
            f"- **Vague 1 (Critique):** {len(priorities['wave_1_critical'])} modules",
            f"- **Vague 2 (Cœur):** {len(priorities['wave_2_core'])} modules",
            f"- **Vague 3 (Infrastructure):** {len(priorities['wave_3_orchestration'])} modules",
            f"- **Vague 4 (Standard):** {len(priorities['wave_4_standard'])} modules",
            "",
            "## 🚀 Prochaine Étape",
            "",
            "Commencer immédiatement par les modules de la Vague 1.",
            "Chaque module migré renforce les capacités de Jeffrey Phoenix.",
        ]
    )

    return "\n".join(lines)


def main():
    # Chemins
    inventory_path = Path("unified/reports/complete_inventory.json")
    plan_path = Path("unified/reports/migration_plan.md")

    # Vérifier que l'inventaire existe
    if not inventory_path.exists():
        print("❌ Inventaire non trouvé. Lancez d'abord le scanner.")
        return

    # Charger et analyser
    print("📊 Chargement de l'inventaire...")
    inventory = load_inventory(inventory_path)

    # Générer le plan
    print("📝 Génération du plan de migration...")
    plan = generate_migration_plan(inventory)

    # Sauvegarder
    plan_path.write_text(plan, encoding='utf-8')

    print(f"✅ Plan de migration créé: {plan_path}")
    print("\nAperçu du plan:")
    print("-" * 40)
    print(plan[:1000])
    print("...")
    print(f"\n📁 Consultez le plan complet dans: {plan_path}")


if __name__ == "__main__":
    main()
