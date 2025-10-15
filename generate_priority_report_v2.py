#!/usr/bin/env python3
"""
Rapport de priorisation V2 avec centralité, obsolescence et estimation
"""

import json


def estimate_complexity(import_name: str) -> tuple:
    """Retourne (complexité_str, heures_estimées)."""
    if any(k in import_name.lower() for k in ['util', 'helper', 'formatter']):
        return "🟢 FAIBLE", 1
    if any(k in import_name.lower() for k in ['adapter', 'client', 'connector']):
        return "🟡 MOYENNE", 2
    if any(k in import_name.lower() for k in ['core', 'engine', 'orchestrat', 'emotional']):
        return "🔴 ÉLEVÉE", 4
    return "🟡 MOYENNE", 2


def determine_priority(refs: int, centrality: int, has_candidates: bool, complexity_hours: int) -> tuple:
    """Retourne (priorité, action)."""
    score = refs + (centrality * 2)

    if has_candidates:
        return "1️⃣ IMMÉDIATE", "Créer Shim"

    if score >= 20 and complexity_hours >= 4:
        return "2️⃣ URGENTE", "Retrouver (iCloud/backups)"
    elif score >= 20:
        return "3️⃣ HAUTE", "Recréer rapidement"
    elif score >= 10:
        return "4️⃣ NORMALE", "Recréer"
    else:
        return "5️⃣ BASSE", "Analyser (obsolète?)"


def main():
    print("📊 RAPPORT DE PRIORISATION V2")
    print("=" * 60)
    print()

    try:
        with open('COMPREHENSIVE_DIAGNOSTIC_V2.json') as f:
            report = json.load(f)
    except FileNotFoundError:
        print("❌ Lancez comprehensive_diagnostic_v2.py d'abord")
        return

    missing_with = report.get('missing_with_candidates', {})
    missing_without = report.get('missing_without_candidates', {})
    centrality = report.get('centrality', {})
    stub_scores = report.get('stubs_with_scores', [])

    # Build priority table
    priority_table = []

    for imp, info in missing_with.items():
        complexity, hours = estimate_complexity(imp)
        cent = centrality.get(imp, 0)
        priority, action = determine_priority(info['references'], cent, True, hours)

        priority_table.append(
            {
                'module': imp,
                'references': info['references'],
                'centrality': cent,
                'score': info['references'] + (cent * 2),
                'complexity': complexity,
                'hours_estimate': hours,
                'priority': priority,
                'action': action,
                'has_source': True,
                'source': info['candidates'][0],
            }
        )

    for imp, info in missing_without.items():
        complexity, hours = estimate_complexity(imp)
        cent = centrality.get(imp, 0)
        priority, action = determine_priority(info['references'], cent, False, hours)

        priority_table.append(
            {
                'module': imp,
                'references': info['references'],
                'centrality': cent,
                'score': info['references'] + (cent * 2),
                'complexity': complexity,
                'hours_estimate': hours,
                'priority': priority,
                'action': action,
                'has_source': False,
                'source': None,
            }
        )

    # Sort by score
    priority_table.sort(key=lambda x: x['score'], reverse=True)

    # Calculate sprint estimates
    total_hours = sum(item['hours_estimate'] for item in priority_table if not item['has_source'])
    sprints = []
    current_sprint = []
    current_hours = 0

    for item in priority_table:
        if item['has_source']:
            continue

        if current_hours + item['hours_estimate'] > 40:  # 1 semaine
            sprints.append(current_sprint)
            current_sprint = [item]
            current_hours = item['hours_estimate']
        else:
            current_sprint.append(item)
            current_hours += item['hours_estimate']

    if current_sprint:
        sprints.append(current_sprint)

    # Generate Markdown report
    with open('PRIORITIZATION_REPORT_V2.md', 'w') as f:
        f.write("# 🎯 RAPPORT DE PRIORISATION V2 - JEFFREY OS\n\n")
        f.write(f"**Généré** : {report['timestamp']}\n\n")
        f.write("---\n\n")

        f.write("## 📊 Vue d'Ensemble\n\n")
        f.write(f"- **Total modules** : {len(priority_table)}\n")
        f.write(f"- **Avec sources** : {len(missing_with)}\n")
        f.write(f"- **À recréer** : {len(missing_without)}\n")
        f.write(f"- **Stubs à nettoyer** : {len(stub_scores)}\n")
        f.write(f"- **Effort estimé** : {total_hours}h (~{len(sprints)} sprints d'une semaine)\n\n")

        f.write("---\n\n")

        f.write("## 🔥 TOP 20 PRIORITÉS\n\n")
        f.write("| # | Module | Score | Refs | Cent. | Complexité | Heures | Priorité | Action |\n")
        f.write("|---|--------|-------|------|-------|------------|--------|----------|--------|\n")

        for i, item in enumerate(priority_table[:20], 1):
            f.write(
                f"| {i} | `{item['module']}` | {item['score']} | {item['references']} | "
                f"{item['centrality']} | {item['complexity']} | {item['hours_estimate']}h | "
                f"{item['priority']} | {item['action']} |\n"
            )

        f.write("\n---\n\n")

        f.write("## 📅 PLANIFICATION PAR SPRINTS\n\n")
        for i, sprint in enumerate(sprints, 1):
            sprint_hours = sum(item['hours_estimate'] for item in sprint)
            f.write(f"### Sprint {i} ({sprint_hours}h)\n\n")
            for item in sprint:
                f.write(f"- **{item['module']}** ({item['hours_estimate']}h)\n")
                f.write(f"  - Priorité: {item['priority']}\n")
                f.write(f"  - Action: {item['action']}\n\n")

        f.write("---\n\n")

        f.write("## 🗑️  MODULES OBSOLÈTES À SUPPRIMER\n\n")
        obsolete = [s for s in stub_scores if s['obsolescence_score'] > 70]
        if obsolete:
            f.write(f"**{len(obsolete)} modules probablement obsolètes** :\n\n")
            for stub in obsolete:
                f.write(f"- `{stub['file']}` (score: {stub['obsolescence_score']}, refs: {stub['references']})\n")
        else:
            f.write("_Aucun module obsolète détecté_\n")

        f.write("\n---\n\n")

        f.write("## 💡 WORKFLOW RECOMMANDÉ (Boucle de Restauration Unitaire)\n\n")
        f.write("Pour chaque module du TOP 20, suivez ces 5 étapes :\n\n")
        f.write("### 1. **Choisir** (1 min)\n")
        f.write("Prenez le premier module non traité de la liste ci-dessus.\n\n")

        f.write("### 2. **Chercher** (15 min chrono)\n")
        f.write("```bash\n")
        f.write("# Exemple pour emotional_core\n")
        f.write("find ~/iCloud -name \"*emotional_core*\"\n")
        f.write("grep -r \"emotional_core\" ~/backups/\n")
        f.write("```\n")
        f.write("Si rien après 15 min → considéré perdu\n\n")

        f.write("### 3. **Décider** (5 min)\n")
        f.write("- **Trouvé** : Copier au bon endroit + relancer validation\n")
        f.write("- **Introuvable** : Recréer (voir étape 4)\n")
        f.write("- **Obsolète** : Supprimer les imports\n\n")

        f.write("### 4. **Implémenter** (1-4h selon complexité)\n")
        f.write("```bash\n")
        f.write("# a. Consulter le contrat d'interface\n")
        f.write("cat interface_contracts/jeffrey_core_emotions_emotional_core.md\n\n")
        f.write("# b. Copier le squelette généré\n")
        f.write("# c. Analyser l'usage réel\n")
        f.write("grep -r \"emotional_core\" src/ services/\n\n")
        f.write("# d. Implémenter la logique minimale\n")
        f.write("# e. Documenter avec TODO si incomplet\n")
        f.write("```\n\n")

        f.write("### 5. **Valider** (5 min)\n")
        f.write("```bash\n")
        f.write("# Test d'import\n")
        f.write("python3 -c \"import jeffrey.core.emotions.emotional_core; print('OK')\"\n\n")
        f.write("# Validation complète\n")
        f.write("bash validate_complete.sh\n\n")
        f.write("# Commit\n")
        f.write("git add src/jeffrey/core/emotions/emotional_core.py\n")
        f.write("git commit -m \"feat(core): Recréation emotional_core (IMV)\"\n")
        f.write("```\n\n")

        f.write("---\n\n")
        f.write("## 📈 Légende\n\n")
        f.write("- **Score** : Références + (Centralité × 2)\n")
        f.write("- **Centralité** : Nombre de modules qui dépendent de celui-ci\n")
        f.write("- **Priorités** :\n")
        f.write("  - 1️⃣ IMMÉDIATE : Shim disponible\n")
        f.write("  - 2️⃣ URGENTE : Critique + complexe\n")
        f.write("  - 3️⃣ HAUTE : Critique + simple\n")
        f.write("  - 4️⃣ NORMALE : Important\n")
        f.write("  - 5️⃣ BASSE : Peu utilisé (peut-être obsolète)\n")

    print("✅ Rapport généré : PRIORITIZATION_REPORT_V2.md")
    print()
    print(f"📊 {len(priority_table)} modules au total")
    print(f"⏱️  Effort estimé : {total_hours}h (~{len(sprints)} sprints)")
    print()
    print("💡 Consultez le rapport pour planifier votre travail")


if __name__ == "__main__":
    main()
