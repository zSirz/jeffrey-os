#!/usr/bin/env python3
"""
Script de diagnostic pour identifier tous les attributs de l'AGI Orchestrator
"""

from jeffrey.core.orchestration.agi_orchestrator import AGIOrchestrator

print("=" * 80)
print("🔍 DIAGNOSTIC AGI ORCHESTRATOR - ATTRIBUTS")
print("=" * 80)

# Initialiser l'orchestrator
print("\n📌 Initialisation...")
orch = AGIOrchestrator()
print("✅ Orchestrator initialisé")

# Lister TOUS les attributs
print("\n" + "=" * 80)
print("📋 TOUS LES ATTRIBUTS DE L'ORCHESTRATOR")
print("=" * 80)

all_attrs = [attr for attr in dir(orch) if not attr.startswith('_')]

# Catégoriser les attributs
memory_related = []
learning_related = []
emotion_related = []
agi_related = []
other_attrs = []

for attr in all_attrs:
    attr_lower = attr.lower()

    if 'memory' in attr_lower or 'recall' in attr_lower or 'souvenir' in attr_lower:
        memory_related.append(attr)
    elif 'learn' in attr_lower or 'pattern' in attr_lower or 'self' in attr_lower:
        learning_related.append(attr)
    elif 'emotion' in attr_lower or 'sentiment' in attr_lower or 'feeling' in attr_lower:
        emotion_related.append(attr)
    elif 'agi' in attr_lower or 'system' in attr_lower:
        agi_related.append(attr)
    else:
        other_attrs.append(attr)

# Afficher par catégorie
print("\n🧠 ATTRIBUTS LIÉS À LA MÉMOIRE:")
if memory_related:
    for attr in memory_related:
        obj = getattr(orch, attr, None)
        obj_type = type(obj).__name__ if obj else "None"
        print(f"   ✅ {attr.ljust(30)} : {obj_type}")

        # Tester les méthodes de mémoire
        if obj and hasattr(obj, 'get_relevant_memories'):
            print("      → a get_relevant_memories() ✅")
        if obj and hasattr(obj, 'add_memory'):
            print("      → a add_memory() ✅")
        if obj and hasattr(obj, 'search'):
            print("      → a search() ✅")
else:
    print("   ⚠️ Aucun attribut trouvé")

print("\n📚 ATTRIBUTS LIÉS À L'APPRENTISSAGE:")
if learning_related:
    for attr in learning_related:
        obj = getattr(orch, attr, None)
        obj_type = type(obj).__name__ if obj else "None"
        print(f"   ✅ {attr.ljust(30)} : {obj_type}")

        # Tester les méthodes d'apprentissage
        if obj and hasattr(obj, 'learn_from_interaction'):
            print("      → a learn_from_interaction() ✅")
        if obj and hasattr(obj, 'get_stats'):
            print("      → a get_stats() ✅")
else:
    print("   ⚠️ Aucun attribut trouvé")

print("\n🎭 ATTRIBUTS LIÉS AUX ÉMOTIONS:")
if emotion_related:
    for attr in emotion_related[:5]:  # Limiter à 5
        obj = getattr(orch, attr, None)
        obj_type = type(obj).__name__ if obj else "None"
        print(f"   ✅ {attr.ljust(30)} : {obj_type}")
else:
    print("   ⚠️ Aucun attribut trouvé")

print("\n🧠 ATTRIBUTS LIÉS AUX SYSTÈMES AGI:")
if agi_related:
    for attr in agi_related[:5]:  # Limiter à 5
        obj = getattr(orch, attr, None)
        obj_type = type(obj).__name__ if obj else "None"
        print(f"   ✅ {attr.ljust(30)} : {obj_type}")
else:
    print("   ⚠️ Aucun attribut trouvé")

print("\n🔧 AUTRES ATTRIBUTS IMPORTANTS:")
important_others = [a for a in other_attrs if not callable(getattr(orch, a, None))][:10]
for attr in important_others:
    obj = getattr(orch, attr, None)
    obj_type = type(obj).__name__ if obj else "None"
    print(f"   ✅ {attr.ljust(30)} : {obj_type}")

# Tests approfondis sur les candidats mémoire
print("\n" + "=" * 80)
print("🧪 TESTS APPROFONDIS SUR LES CANDIDATS MÉMOIRE")
print("=" * 80)

for attr in memory_related:
    obj = getattr(orch, attr, None)
    if obj:
        print(f"\n🔍 Test de {attr} ({type(obj).__name__})")

        # Lister les méthodes disponibles
        methods = [m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m, None))]
        print(f"   Méthodes disponibles : {', '.join(methods[:10])}")

        # Tester l'ajout et le rappel
        try:
            if hasattr(obj, 'add_memory'):
                print("   ✅ Tentative add_memory()...")
                # Ne pas vraiment ajouter, juste tester la signature

            if hasattr(obj, 'get_relevant_memories'):
                print("   ✅ Tentative get_relevant_memories()...")
                result = obj.get_relevant_memories("test", limit=1)
                print(f"      → Résultat : {len(result) if hasattr(result, '__len__') else 'N/A'} items")

            if hasattr(obj, 'search'):
                print("   ✅ Tentative search()...")
                result = obj.search("test")
                print(f"      → Résultat : {len(result) if hasattr(result, '__len__') else 'N/A'} items")
        except Exception as e:
            print(f"   ⚠️ Erreur : {e}")

# Tests approfondis sur les candidats learning
print("\n" + "=" * 80)
print("🧪 TESTS APPROFONDIS SUR LES CANDIDATS LEARNING")
print("=" * 80)

for attr in learning_related:
    obj = getattr(orch, attr, None)
    if obj:
        print(f"\n🔍 Test de {attr} ({type(obj).__name__})")

        # Lister les méthodes disponibles
        methods = [m for m in dir(obj) if not m.startswith('_') and callable(getattr(obj, m, None))]
        print(f"   Méthodes disponibles : {', '.join(methods[:10])}")

        # Tester get_stats
        try:
            if hasattr(obj, 'get_stats'):
                print("   ✅ Tentative get_stats()...")
                stats = obj.get_stats()
                print(f"      → Stats : {stats}")

            if hasattr(obj, 'learn_from_interaction'):
                print("   ✅ a learn_from_interaction() !")
        except Exception as e:
            print(f"   ⚠️ Erreur : {e}")

# RÉSUMÉ FINAL
print("\n" + "=" * 80)
print("📊 RÉSUMÉ - NOMS À UTILISER DANS LE SCRIPT")
print("=" * 80)

print("\n💾 Pour la MÉMOIRE, utiliser :")
if memory_related:
    for attr in memory_related:
        obj = getattr(orch, attr, None)
        if obj and (hasattr(obj, 'get_relevant_memories') or hasattr(obj, 'search')):
            print(f"   → orch.{attr}")
else:
    print("   ⚠️ Aucun module mémoire fonctionnel trouvé")

print("\n📚 Pour l'APPRENTISSAGE, utiliser :")
if learning_related:
    for attr in learning_related:
        obj = getattr(orch, attr, None)
        if obj and hasattr(obj, 'get_stats'):
            print(f"   → orch.{attr}")
else:
    print("   ⚠️ Aucun module learning fonctionnel trouvé")

print("\n" + "=" * 80)
print("✅ Diagnostic terminé !")
print("=" * 80)
