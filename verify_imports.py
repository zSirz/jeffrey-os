import importlib
import sys
from pathlib import Path

# Ajouter le projet au path
sys.path.insert(0, str(Path.cwd() / "src"))

# Modules critiques à tester (chemins corrigés)
CRITICAL_MODULES = {
    "emotional_core": "jeffrey.core.emotions.core.jeffrey_emotional_core",
    "agi_orchestrator": "jeffrey.core.orchestration.agi_orchestrator",
    "memory_systems": "jeffrey.core.memory_systems",
    "memory_interface": "jeffrey.core.memory_interface",
    "self_learning": "jeffrey.core.self_learning",
    "dialogue_engine": "jeffrey.core.orchestration.dialogue_engine",
    "agi_fusion": "jeffrey.core.agi_fusion.agi_orchestrator",
    "consciousness": "jeffrey.core.consciousness.jeffrey_consciousness_v3",
    "emotional_effects": "jeffrey.core.emotions.emotional_effects_engine",
}

print("🔍 VÉRIFICATION DES IMPORTS CRITIQUES\n")
print(f"📁 PYTHONPATH: {sys.path[0]}")
print("=" * 60)

working_modules = []
broken_modules = []

for name, module_path in CRITICAL_MODULES.items():
    try:
        mod = importlib.import_module(module_path)
        print(f"✅ {name:20} → {module_path}")
        working_modules.append((name, mod))

        # Tests spécifiques par module
        if name == "emotional_core":
            core_cls = getattr(mod, "JeffreyEmotionalCore", None)
            print(f"   Class: {core_cls.__name__ if core_cls else '❌ manquante'}")
            if core_cls:
                try:
                    core = core_cls()
                    has_hybrid = hasattr(core, "analyze_emotion_hybrid")
                    has_classic = hasattr(core, "analyze_and_resonate")
                    print(f"   API: hybrid={has_hybrid}, classic={has_classic}")

                    # Test rapide
                    if has_hybrid:
                        result = core.analyze_emotion_hybrid("Je suis super heureux ! 🎉✨", "")
                        print(f"   Test: {result}")
                except Exception as e:
                    print(f"   ⚠️  Erreur instantiation: {e}")

        elif name == "agi_orchestrator":
            orchestrator_cls = getattr(mod, "AGIOrchestrator", None) or getattr(mod, "JeffreyAGIOrchestrator", None)
            print(f"   Class: {orchestrator_cls.__name__ if orchestrator_cls else '❌ manquante'}")

        elif name == "memory_systems":
            memory_cls = getattr(mod, "MemoryCore", None) or getattr(mod, "UnifiedMemory", None)
            print(f"   Class: {memory_cls.__name__ if memory_cls else '❌ manquante'}")

        elif name == "self_learning":
            learning_cls = getattr(mod, "SelfLearningModule", None)
            print(f"   Class: {learning_cls.__name__ if learning_cls else '❌ manquante'}")

        elif name == "dialogue_engine":
            dialogue_cls = getattr(mod, "DialogueEngine", None)
            print(f"   Class: {dialogue_cls.__name__ if dialogue_cls else '❌ manquante'}")

        # Lister les classes/fonctions principales
        classes = [
            attr
            for attr in dir(mod)
            if not attr.startswith('_')
            and hasattr(getattr(mod, attr), '__class__')
            and str(type(getattr(mod, attr))) == "<class 'type'>"
        ]
        if classes:
            print(f"   Classes: {', '.join(classes[:5])}")

    except Exception as e:
        print(f"❌ {name:20} → ERREUR : {e}")
        broken_modules.append((name, module_path, str(e)))

print("\n" + "=" * 60)
print(f"📊 BILAN: {len(working_modules)}/{len(CRITICAL_MODULES)} modules fonctionnels")

if broken_modules:
    print("\n🚨 MODULES CASSÉS À RÉPARER:")
    for name, path, error in broken_modules:
        print(f"  - {name}: {error}")

if working_modules:
    print("\n✅ MODULES OPÉRATIONNELS:")
    for name, mod in working_modules:
        print(f"  - {name}: {len([x for x in dir(mod) if not x.startswith('_')])} exports")

print("\n🧪 SANITY CHECK RAPIDE:")
print("Pour tester manuellement un module:")
print(
    "PYTHONPATH=src python -c \"from jeffrey.core.emotions.core.jeffrey_emotional_core import JeffreyEmotionalCore; print('OK')\""
)
