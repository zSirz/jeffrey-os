#!/usr/bin/env python3
"""
Script de diagnostic pour identifier la vraie signature du SelfLearningModule
"""

import inspect
import sys

from jeffrey.core.self_learning_module import get_learning_module

print("=" * 80)
print("🔍 DIAGNOSTIC LEARNING MODULE - SIGNATURES")
print("=" * 80)

# Récupérer le learning module
print("\n📌 Récupération du learning module...")
try:
    learning_module = get_learning_module()
    print(f"✅ Learning module récupéré : {type(learning_module).__name__}")
except Exception as e:
    print(f"❌ Erreur : {e}")
    sys.exit(1)

# Examiner la méthode learn_from_interaction
print("\n" + "=" * 80)
print("🔍 ANALYSE DE learn_from_interaction()")
print("=" * 80)

if hasattr(learning_module, 'learn_from_interaction'):
    method = getattr(learning_module, 'learn_from_interaction')

    # Récupérer la signature
    try:
        signature = inspect.signature(method)
        print("\n📋 SIGNATURE COMPLÈTE :")
        print(f"   learn_from_interaction{signature}")

        print("\n📝 PARAMÈTRES DÉTAILLÉS :")
        for param_name, param in signature.parameters.items():
            default = f" = {param.default}" if param.default != inspect.Parameter.empty else ""
            annotation = f": {param.annotation}" if param.annotation != inspect.Parameter.empty else ""
            print(f"   - {param_name}{annotation}{default}")

    except Exception as e:
        print(f"❌ Erreur inspection : {e}")

else:
    print("❌ Méthode learn_from_interaction non trouvée")

# Examiner toutes les méthodes publiques
print("\n" + "=" * 80)
print("🔍 TOUTES LES MÉTHODES PUBLIQUES")
print("=" * 80)

methods = [m for m in dir(learning_module) if not m.startswith('_') and callable(getattr(learning_module, m, None))]
for method_name in methods:
    method = getattr(learning_module, method_name)
    try:
        signature = inspect.signature(method)
        print(f"   {method_name}{signature}")
    except:
        print(f"   {method_name}(...)")

# Test avec différentes signatures
print("\n" + "=" * 80)
print("🧪 TESTS DE SIGNATURES POSSIBLES")
print("=" * 80)

test_cases = [
    # Test 1: Sans emotion
    {"name": "Sans emotion", "args": ["test message"], "kwargs": {"memory_context": {"memories": 0}}},
    # Test 2: Avec emotion string
    {
        "name": "Avec emotion string",
        "args": ["test message"],
        "kwargs": {"emotion": "neutre", "memory_context": {"memories": 0}},
    },
    # Test 3: Avec user_input explicite
    {
        "name": "Avec user_input explicite",
        "args": [],
        "kwargs": {"user_input": "test message", "memory_context": {"memories": 0}},
    },
    # Test 4: Seulement message
    {"name": "Seulement message", "args": ["test message"], "kwargs": {}},
    # Test 5: Avec emotion_data
    {
        "name": "Avec emotion_data",
        "args": ["test message"],
        "kwargs": {"emotion_data": {"emotion": "neutre"}, "memory_context": {"memories": 0}},
    },
]

for i, test_case in enumerate(test_cases, 1):
    print(f"\n🧪 Test {i}: {test_case['name']}")
    try:
        # Ne pas vraiment exécuter, juste tester la signature
        result = learning_module.learn_from_interaction(*test_case['args'], **test_case['kwargs'])
        print("   ✅ SUCCÈS - Signature acceptée")

        # Si c'est réussi, essayer get_stats
        try:
            stats = learning_module.get_stats()
            print(f"   📊 Stats après test : {stats}")
        except Exception as e:
            print(f"   ⚠️ get_stats() erreur : {e}")

        break  # Arrêter au premier succès

    except TypeError as e:
        print(f"   ❌ ERREUR SIGNATURE : {e}")
    except Exception as e:
        print(f"   ⚠️ AUTRE ERREUR : {e}")

# Examiner le code source si possible
print("\n" + "=" * 80)
print("🔍 INSPECTION DU CODE SOURCE")
print("=" * 80)

try:
    source_file = inspect.getfile(learning_module.__class__)
    print(f"📁 Fichier source : {source_file}")

    # Lire les premières lignes de learn_from_interaction
    with open(source_file, encoding='utf-8') as f:
        source = f.read()

    # Chercher la définition de learn_from_interaction
    if 'def learn_from_interaction' in source:
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'def learn_from_interaction' in line:
                print(f"\n📝 DÉFINITION TROUVÉE (ligne {i + 1}):")
                # Afficher la signature et quelques lignes suivantes
                for j in range(min(10, len(lines) - i)):
                    print(f"   {i + j + 1:3d} | {lines[i + j]}")
                break
    else:
        print("❌ Définition de learn_from_interaction non trouvée dans le source")

except Exception as e:
    print(f"❌ Impossible d'examiner le code source : {e}")

print("\n" + "=" * 80)
print("📊 RÉSUMÉ - SOLUTION RECOMMANDÉE")
print("=" * 80)

print("\n💡 Pour corriger l'appel dans test_jeffrey_life_simulation.py :")
print("   1. Utiliser la signature qui fonctionne (voir tests ci-dessus)")
print("   2. Modifier l'appel selon les résultats")
print("   3. Assurer la compatibilité avec get_stats()")

print("\n" + "=" * 80)
print("✅ Diagnostic terminé !")
print("=" * 80)
