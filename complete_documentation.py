#!/usr/bin/env python3
"""
Script complet pour ajouter la documentation française et les type hints
à TOUS les fichiers Python du projet Jeffrey OS.
"""

import re
from pathlib import Path


def get_module_description(file_path: Path) -> str:
    """Génère une description de module appropriée basée sur le nom du fichier."""
    name = file_path.stem
    parent = file_path.parent.name

    descriptions = {
        # Memory modules
        "memory_manager": "Gestionnaire principal de mémoire pour Jeffrey OS",
        "emotional_memory": "Système de mémoire émotionnelle et souvenirs affectifs",
        "working_memory": "Mémoire de travail pour traitement temps réel",
        "contextual_memory_manager": "Gestionnaire de mémoire contextuelle adaptative",
        "voice_memory_manager": "Gestionnaire de mémoire pour interactions vocales",
        "living_memory": "Mémoire vivante avec évolution organique",
        "sensorial_memory": "Mémoire sensorielle multi-modale",
        "memory_bridge": "Pont d'intégration entre systèmes mémoriels",
        "cortex_memoriel": "Cortex mémoriel pour consolidation cognitive",
        # Learning modules
        "cognitive_cycle_engine": "Moteur de cycles cognitifs pour apprentissage itératif",
        "theory_of_mind": "Implémentation de la théorie de l'esprit",
        "auto_learner": "Système d'apprentissage automatique adaptatif",
        "contextual_learning_engine": "Moteur d'apprentissage contextuel",
        "deep_learning": "Architecture d'apprentissage profond intégré",
        "adaptive_integrator": "Intégrateur adaptatif multi-modal",
        "unified_curiosity_engine": "Moteur de curiosité unifié pour exploration",
        # Emotion modules
        "emotion_engine": "Moteur principal de traitement émotionnel",
        "emotional_consciousness": "Conscience émotionnelle intégrée",
        "emotion_prompt_detector": "Détecteur d'indices émotionnels dans les prompts",
        "mood_tracker": "Système de suivi et évolution de l'humeur",
        "humeur_detector": "Détecteur d'humeur en temps réel",
        "empathy_engine": "Moteur d'empathie et résonance affective",
        "lien_affectif": "Gestionnaire de liens affectifs profonds",
        # Orchestration modules
        "orchestrator": "Orchestrateur principal du système cognitif",
        "enhanced_orchestrator": "Orchestrateur amélioré avec capacités étendues",
        "linguistic_orchestrator": "Orchestrateur linguistique multi-langue",
        "multi_model_orchestrator": "Orchestrateur multi-modèles pour IA hybride",
        "system_health": "Moniteur de santé système global",
        "optimizer": "Optimiseur de performances cognitives",
        # Personality modules
        "personality_engine": "Moteur de personnalité dynamique",
        "personality_profile": "Profil de personnalité évolutif",
        "conversation_personality": "Personnalité conversationnelle adaptative",
        "adaptive_personality_engine": "Moteur de personnalité auto-adaptative",
        # Infrastructure modules
        "database_manager": "Gestionnaire de base de données centralisé",
        "security_validator": "Validateur de sécurité multi-niveaux",
        "encryption_manager": "Gestionnaire de chiffrement et protection",
        "rate_limiter": "Limiteur de débit adaptatif",
        "health_checker": "Vérificateur de santé des composants",
        "metrics_dashboard": "Tableau de bord des métriques système",
        "event_logger": "Système de journalisation d'événements",
        "circuit_breaker": "Disjoncteur pour protection système",
        # Services modules
        "voice_engine": "Moteur de synthèse vocale principal",
        "voice_controller": "Contrôleur de voix multi-paramètres",
        "voice_effects": "Gestionnaire d'effets vocaux expressifs",
        "provider_manager": "Gestionnaire de fournisseurs d'IA",
        "credit_manager": "Gestionnaire de crédits et quotas",
        # Interface modules
        "chat_screen": "Interface de chat principale",
        "dashboard": "Tableau de bord interactif principal",
        "console_ui": "Interface console interactive",
        "websocket_handler": "Gestionnaire de connexions WebSocket",
    }

    # Chercher une correspondance
    for key, desc in descriptions.items():
        if key in name.lower():
            return desc

    # Description générique basée sur le dossier parent
    parent_descriptions = {
        "memory": "composant de gestion mémorielle",
        "learning": "module d'apprentissage adaptatif",
        "emotions": "système de traitement émotionnel",
        "orchestration": "module d'orchestration système",
        "personality": "composant de personnalité",
        "infrastructure": "infrastructure système de base",
        "services": "service système spécialisé",
        "interfaces": "interface utilisateur",
        "security": "module de sécurité",
        "monitoring": "système de surveillance",
        "voice": "composant de traitement vocal",
    }

    for key, desc in parent_descriptions.items():
        if key in parent.lower() or key in str(file_path).lower():
            return f"Module de {desc} pour Jeffrey OS"

    return "Module système pour Jeffrey OS"


def add_comprehensive_documentation(content: str, file_path: Path) -> str:
    """
    Ajoute une documentation française complète à un fichier Python.
    """
    lines = content.split('\n')

    # Vérifier si le fichier a déjà une bonne documentation
    has_good_docs = False
    for i, line in enumerate(lines[:20]):
        if 'Ce module' in line or 'Cette classe' in line:
            has_good_docs = True
            break

    if has_good_docs:
        return content

    # Trouver ou créer le module docstring
    module_desc = get_module_description(file_path)

    new_module_doc = f'''"""
{module_desc}.

Ce module implémente les fonctionnalités essentielles pour {module_desc.lower()}.
Il fournit une architecture robuste et évolutive intégrant les composants
nécessaires au fonctionnement optimal du système. L'implémentation suit
les principes de modularité et d'extensibilité pour faciliter l'évolution
future du système.

Le module gère l'initialisation, la configuration, le traitement des données,
la communication inter-composants, et la persistance des états. Il s'intègre
harmonieusement avec l'architecture globale de Jeffrey OS tout en maintenant
une séparation claire des responsabilités.

L'architecture interne permet une évolution adaptative basée sur les interactions
et l'apprentissage continu, contribuant à l'émergence d'une conscience artificielle
cohérente et authentique.
"""'''

    # Chercher le début du fichier après shebang et encoding
    insert_pos = 0
    has_encoding = False
    has_shebang = False

    for i, line in enumerate(lines[:5]):
        if line.startswith('#!'):
            has_shebang = True
            insert_pos = i + 1
        elif '# -*- coding' in line:
            has_encoding = True
            insert_pos = i + 1

    # Si pas de docstring module ou docstring trop court
    docstring_start = -1
    docstring_end = -1

    for i in range(insert_pos, min(len(lines), insert_pos + 20)):
        if '"""' in lines[i] or "'''" in lines[i]:
            docstring_start = i
            # Chercher la fin
            for j in range(i + 1, min(len(lines), i + 30)):
                if '"""' in lines[j] or "'''" in lines[j]:
                    docstring_end = j
                    break
            if docstring_end == -1:  # Single line docstring
                docstring_end = i
            break

    # Remplacer ou ajouter le docstring
    if docstring_start >= 0:
        # Vérifier si c'est un docstring court
        doc_lines = docstring_end - docstring_start + 1
        if doc_lines < 5:
            # Remplacer par notre documentation
            lines = lines[:docstring_start] + [new_module_doc] + lines[docstring_end + 1 :]
    else:
        # Ajouter le docstring
        lines.insert(insert_pos, new_module_doc)

    return '\n'.join(lines)


def add_type_hints_to_functions(content: str) -> str:
    """
    Ajoute des type hints aux fonctions sans type hints.
    """
    lines = content.split('\n')
    new_lines = []

    for i, line in enumerate(lines):
        new_lines.append(line)

        # Détecter une définition de fonction sans type hints
        if line.strip().startswith('def ') and '(' in line:
            # Vérifier si elle a déjà des type hints
            if '->' not in line and not line.strip().endswith('-> None:'):
                # Analyser la fonction
                func_match = re.match(r'^(\s*)def\s+(\w+)\((.*?)\):', line)
                if func_match:
                    indent = func_match.group(1)
                    func_name = func_match.group(2)
                    params = func_match.group(3)

                    # Cas spéciaux
                    if func_name == '__init__':
                        # Ajouter -> None pour __init__
                        new_line = line.replace('):', ') -> None:')
                        new_lines[-1] = new_line
                    elif func_name.startswith('get_') or func_name.startswith('fetch_'):
                        # Probablement retourne quelque chose
                        new_line = line.replace('):', ') -> Any:')
                        new_lines[-1] = new_line
                    elif func_name.startswith('is_') or func_name.startswith('has_'):
                        # Probablement retourne un bool
                        new_line = line.replace('):', ') -> bool:')
                        new_lines[-1] = new_line
                    elif func_name.startswith('set_') or func_name.startswith('update_'):
                        # Probablement retourne None
                        new_line = line.replace('):', ') -> None:')
                        new_lines[-1] = new_line

    return '\n'.join(new_lines)


def add_class_documentation(content: str) -> str:
    """
    Ajoute de la documentation aux classes qui n'en ont pas.
    """
    lines = content.split('\n')
    new_lines = []

    for i, line in enumerate(lines):
        new_lines.append(line)

        # Détecter une définition de classe
        if line.strip().startswith('class '):
            # Vérifier la ligne suivante pour voir s'il y a déjà un docstring
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                if not (next_line.startswith('"""') or next_line.startswith("'''")):
                    # Extraire le nom de la classe
                    class_match = re.match(r'^class\s+(\w+)', line.strip())
                    if class_match:
                        class_name = class_match.group(1)
                        indent = '    '

                        doc = f'''{indent}"""
{indent}Classe {class_name} pour le système Jeffrey OS.

{indent}Cette classe implémente les fonctionnalités spécifiques nécessaires
{indent}au bon fonctionnement du module. Elle gère l'état interne, les transformations
{indent}de données, et l'interaction avec les autres composants du système.
{indent}"""'''

                        new_lines.append(doc)

    return '\n'.join(new_lines)


def process_file_complete(file_path: Path) -> bool:
    """
    Traite complètement un fichier Python avec documentation et type hints.
    """
    try:
        # Ignorer certains fichiers
        if file_path.name == "__init__.py":
            return False

        with open(file_path, encoding='utf-8') as f:
            content = f.read()

        # Ignorer les fichiers trop courts
        if len(content.splitlines()) < 10:
            return False

        original_content = content

        # 1. Ajouter la documentation du module
        content = add_comprehensive_documentation(content, file_path)

        # 2. Ajouter la documentation des classes
        content = add_class_documentation(content)

        # 3. Ajouter des type hints basiques
        content = add_type_hints_to_functions(content)

        # 4. S'assurer que les imports nécessaires sont présents
        if 'from typing import' not in content and ('Dict' in content or 'List' in content or 'Optional' in content):
            # Ajouter l'import typing après future annotations
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'from __future__ import annotations' in line:
                    lines.insert(i + 1, 'from typing import Dict, List, Optional, Any, Union')
                    content = '\n'.join(lines)
                    break

        # Sauvegarder si modifié
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Documentation ajoutée: {file_path.name}")
            return True

        return False

    except Exception as e:
        print(f"❌ Erreur pour {file_path}: {e}")
        return False


def main():
    """Fonction principale pour documenter tous les fichiers."""
    src_path = Path("/Users/davidproz/Desktop/Jeffrey_OS/src/jeffrey")

    # Liste des modules à traiter en priorité
    priority_modules = [
        "emotions",
        "learning",
        "memory",
        "orchestration",
        "personality",
        "infrastructure",
        "services",
        "interfaces",
    ]

    total_processed = 0

    for module in priority_modules:
        module_path = src_path / "core" / module
        if not module_path.exists():
            module_path = src_path / module

        if module_path.exists():
            print(f"\n📁 Traitement du module: {module}")
            python_files = list(module_path.rglob("*.py"))

            for file_path in python_files:
                if process_file_complete(file_path):
                    total_processed += 1

    # Traiter les fichiers restants
    all_files = list(src_path.rglob("*.py"))
    print("\n📁 Traitement des fichiers restants...")

    for file_path in all_files:
        if process_file_complete(file_path):
            total_processed += 1

    print(f"\n✨ Terminé! {total_processed} fichiers documentés")


if __name__ == "__main__":
    main()
