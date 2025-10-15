#!/usr/bin/env python3
"""
Script de diagnostic pour identifier comment ÉCRIRE dans la mémoire
"""

import inspect

from jeffrey.core.orchestration.agi_orchestrator import AGIOrchestrator

print("=" * 80)
print("🔍 DIAGNOSTIC MÉMOIRE - MÉTHODES D'ÉCRITURE")
print("=" * 80)

# Initialiser l'orchestrator
print("\n📌 Initialisation...")
orch = AGIOrchestrator()
print("✅ Orchestrator initialisé")

# Liste des méthodes d'écriture possibles
write_methods = [
    'add_memory',
    'store',
    'add',
    'save',
    'record',
    'add_interaction',
    'store_memory',
    'save_memory',
    'add_message',
    'store_message',
]

# Test sur memory (UnifiedMemory)
print("\n" + "=" * 80)
print("🧠 TEST 1 : memory (UnifiedMemory)")
print("=" * 80)

mem = getattr(orch, 'memory', None)
if mem:
    print(f"✅ memory existe : {type(mem).__name__}")
    print("\n📋 Méthodes d'écriture disponibles :")

    found_methods = []
    for method_name in write_methods:
        if hasattr(mem, method_name):
            method = getattr(mem, method_name)
            if callable(method):
                try:
                    sig = inspect.signature(method)
                    print(f"\n   ✅ {method_name}{sig}")
                    found_methods.append((method_name, sig))

                    # Afficher la docstring si disponible
                    if method.__doc__:
                        doc_lines = method.__doc__.strip().split('\n')
                        print(f"      Doc: {doc_lines[0][:80]}")
                except Exception as e:
                    print(f"   ⚠️ {method_name} : Erreur signature - {e}")

    if not found_methods:
        print("   ⚠️ Aucune méthode d'écriture standard trouvée")
        print("\n   📋 Toutes les méthodes publiques :")
        all_methods = [m for m in dir(mem) if not m.startswith('_') and callable(getattr(mem, m, None))]
        for m in all_methods[:20]:
            print(f"      - {m}")
else:
    print("❌ memory n'existe pas")

# Test sur memory_v2_interface
print("\n" + "=" * 80)
print("🧠 TEST 2 : memory_v2_interface")
print("=" * 80)

mem_v2 = getattr(orch, 'memory_v2_interface', None)
if mem_v2:
    print(f"✅ memory_v2_interface existe : {type(mem_v2).__name__}")
    print("\n📋 Méthodes d'écriture disponibles :")

    found_methods_v2 = []
    for method_name in write_methods:
        if hasattr(mem_v2, method_name):
            method = getattr(mem_v2, method_name)
            if callable(method):
                try:
                    sig = inspect.signature(method)
                    print(f"\n   ✅ {method_name}{sig}")
                    found_methods_v2.append((method_name, sig))

                    if method.__doc__:
                        doc_lines = method.__doc__.strip().split('\n')
                        print(f"      Doc: {doc_lines[0][:80]}")
                except Exception as e:
                    print(f"   ⚠️ {method_name} : Erreur signature - {e}")

    if not found_methods_v2:
        print("   ⚠️ Aucune méthode d'écriture standard trouvée")
else:
    print("❌ memory_v2_interface n'existe pas")

# Test fonctionnel
print("\n" + "=" * 80)
print("🧪 TEST 3 : TEST FONCTIONNEL D'ÉCRITURE")
print("=" * 80)

# Tester avec memory
if mem and found_methods:
    print("\n🔬 Test avec memory :")
    method_name, sig = found_methods[0]
    print(f"   Utilisation de : {method_name}{sig}")

    try:
        # Essayer d'ajouter un souvenir de test
        params = str(sig.parameters.keys())
        print(f"   Paramètres attendus : {params}")

        # Tentative d'appel avec paramètres génériques
        method = getattr(mem, method_name)

        # Essayer plusieurs variantes
        test_data = {
            "content": "Test de mémoire",
            "message": "Test de mémoire",
            "text": "Test de mémoire",
            "data": "Test de mémoire",
            "emotion": "joie",
            "category": "test",
            "context": {},
            "metadata": {},
        }

        # Construire l'appel selon les paramètres
        param_names = list(sig.parameters.keys())
        call_args = {}
        for param in param_names:
            if param in test_data:
                call_args[param] = test_data[param]
            elif param in ['timestamp', 'time']:
                from datetime import datetime

                call_args[param] = datetime.now().isoformat()

        print(f"   Tentative d'écriture avec : {list(call_args.keys())}")
        result = method(**call_args)
        print(f"   ✅ SUCCÈS ! Retour : {result}")

        # Essayer de relire
        if hasattr(mem, 'search_memories'):
            memories = mem.search_memories("Test")
            print(f"   ✅ Relecture : {len(memories)} souvenirs trouvés")

    except Exception as e:
        print(f"   ⚠️ Échec : {e}")
        import traceback

        traceback.print_exc()

# RÉSUMÉ FINAL
print("\n" + "=" * 80)
print("📊 RÉSUMÉ - MÉTHODE À UTILISER")
print("=" * 80)

if mem and found_methods:
    method_name, sig = found_methods[0]
    print("\n✅ UTILISER CETTE MÉTHODE :")
    print(f"   mem_store.{method_name}{sig}")
    print("\n📋 Code à ajouter dans test_jeffrey_life_simulation.py :")
    print(f"""
    if mem_store and hasattr(mem_store, '{method_name}'):
        try:
            mem_store.{method_name}(
                # Remplir selon la signature ci-dessus
            )
        except Exception as e:
            print(f"⚠️ Erreur enregistrement : {{e}}")
    """)
elif mem_v2 and found_methods_v2:
    method_name, sig = found_methods_v2[0]
    print("\n✅ UTILISER CETTE MÉTHODE (memory_v2_interface) :")
    print(f"   mem_store.{method_name}{sig}")
else:
    print("\n❌ AUCUNE MÉTHODE D'ÉCRITURE TROUVÉE")
    print("   → Vérifier le code source des modules mémoire")

print("\n" + "=" * 80)
