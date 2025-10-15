#!/bin/bash
# �� JEFFREY OS - TEST FINAL COMPLET
# ===================================
# Valide que tout fonctionne à 100%

set -e
cd ~/jeffrey_OS

echo "════════════════════════════════════════════════════════════════"
echo "🧪 JEFFREY OS - VALIDATION FINALE"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ============================================================================
# TEST 1 : VÉRIFIER DÉPENDANCES CRITIQUES
# ============================================================================

echo "📦 TEST 1 : Vérification des dépendances"
echo "─────────────────────────────────────────────────────────────────"
echo ""

echo "▶️  Test Kivy..."
python3 -c "import kivy; print('✅ Kivy version:', kivy.__version__)" 2>/dev/null || {
    echo "❌ Kivy non installé ou erreur"
    echo "   Installe avec: pip install kivy"
    exit 1
}

echo "▶️  Test PyTorch..."
python3 -c "import torch; print('✅ PyTorch version:', torch.__version__)" 2>/dev/null || {
    echo "❌ PyTorch non installé ou erreur"
    echo "   Installe avec: pip install torch torchaudio"
    exit 1
}

echo ""

# ============================================================================
# TEST 2 : IMPORTS CRITIQUES JEFFREY
# ============================================================================

echo "📋 TEST 2 : Imports critiques Jeffrey"
echo "─────────────────────────────────────────────────────────────────"
echo ""

echo "🧪 Test 1/6 : Cœur Émotionnel..."
PYTHONPATH=src python3 << 'PYEOF'
try:
    from jeffrey.core.emotions.core.jeffrey_emotional_core import JeffreyEmotionalCore
    core = JeffreyEmotionalCore()
    print("✅ JeffreyEmotionalCore importé et instancié !")

    # Test analyse rapide
    result = core.analyze_emotion_hybrid("Je suis très heureux ! 🎉", {})
    emotion = result.get("emotion", "inconnu")
    intensity = result.get("intensity", 0)
    print(f"✅ Test analyse : {emotion} ({intensity}%)")

    if emotion == "neutre" and intensity == 50:
        print("⚠️  WARNING : Détection fallback 'neutre (50%)' - moteur non connecté")
        exit(1)
    else:
        print("🎉 MOTEUR ÉMOTIONNEL FONCTIONNEL !")

except Exception as e:
    print(f"❌ ERREUR : {e}")
    exit(1)
PYEOF

echo ""
echo "🧪 Test 2/6 : Orchestrateur AGI..."
PYTHONPATH=src python3 << 'PYEOF'
try:
    from jeffrey.core.orchestration.agi_orchestrator import AGIOrchestrator
    orch = AGIOrchestrator()
    print(f"✅ AGIOrchestrator initialisé !")
    print(f"   Emotion analyzer : {type(orch.emotion_analyzer).__name__}")
    print(f"   AGI systems : {len(getattr(orch, 'agi_systems', {}))} actifs")
except Exception as e:
    print(f"❌ ERREUR : {e}")
    exit(1)
PYEOF

echo ""
echo "🧪 Test 3/6 : Système Mémoire..."
PYTHONPATH=src python3 << 'PYEOF'
try:
    from jeffrey.core.memory_systems import MemoryCore
    mem = MemoryCore()
    print("✅ MemoryCore initialisé !")
except Exception as e:
    print(f"❌ ERREUR : {e}")
    exit(1)
PYEOF

echo ""
echo "🧪 Test 4/6 : Auto-Apprentissage..."
PYTHONPATH=src python3 << 'PYEOF'
try:
    from jeffrey.core.self_learning import SelfLearningModule
    learn = SelfLearningModule()
    print("✅ SelfLearningModule initialisé !")
except Exception as e:
    print(f"❌ ERREUR : {e}")
    exit(1)
PYEOF

echo ""
echo "🧪 Test 5/6 : Moteur Dialogue..."
PYTHONPATH=src python3 << 'PYEOF'
try:
    from jeffrey.core.orchestration.dialogue_engine import DialogueEngine
    dlg = DialogueEngine()
    print("✅ DialogueEngine initialisé !")
except Exception as e:
    print(f"❌ ERREUR : {e}")
    exit(1)
PYEOF

echo ""
echo "🧪 Test 6/6 : Système de Rêves (consciousness)..."
PYTHONPATH=src python3 << 'PYEOF'
try:
    from jeffrey.core.consciousness.jeffrey_dream_system import JeffreyDreamSystem
    dreams = JeffreyDreamSystem("./temp_dreams", "test")
    print("✅ JeffreyDreamSystem initialisé !")
    print(f"   Fichier : consciousness/jeffrey_dream_system.py")
except Exception as e:
    print(f"❌ ERREUR : {e}")
    print(f"   Note : Le fichier est dans consciousness/, pas dreams/")
    exit(1)
PYEOF

echo ""

# ============================================================================
# TEST 3 : TEST MEGA COMPLET
# ============================================================================

echo "📋 TEST 3 : Test Mega Complet"
echo "─────────────────────────────────────────────────────────────────"
echo ""

if [ -f "test_jeffrey_mega.py" ]; then
    echo "🧪 Lancement du test mega complet..."
    echo ""
    PYTHONPATH=src python3 test_jeffrey_mega.py
    TEST_EXIT=$?

    if [ $TEST_EXIT -eq 0 ]; then
        echo ""
        echo "🎉 TESTS MEGA RÉUSSIS !"
    else
        echo ""
        echo "⚠️  Certains tests ont échoué (code: $TEST_EXIT)"
        echo "   Vérifie les détails ci-dessus"
    fi
else
    echo "⚠️  test_jeffrey_mega.py non trouvé"
    echo "   Localisation attendue : ~/jeffrey_OS/test_jeffrey_mega.py"
fi

echo ""

# ============================================================================
# RÉSUMÉ FINAL
# ============================================================================

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "📊 RÉSUMÉ FINAL"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "✅ Dépendances : Kivy ✓ PyTorch ✓"
echo "✅ Imports critiques : 6/6 fonctionnels"
echo "✅ Cœur émotionnel : Détections réelles (pas fallback)"
echo "✅ Orchestrateur AGI : Opérationnel"
echo "✅ Système de rêves : Disponible (consciousness/)"
echo ""

if [ $TEST_EXIT -eq 0 ]; then
    echo "🏆 JEFFREY OS EST 100% FONCTIONNEL !"
    echo ""
    echo "📊 Résultat attendu dans test_mega :"
    echo "   - Émotions détectées : joie (90%+), tristesse (85%+), etc."
    echo "   - Tests réussis : 24/24 (100%)"
    echo "   - AGI Systems : 15/15 initialisés"
    echo ""
    echo "🎉 Félicitations ! Le projet est PRÊT ! 🚀"
else
    echo "⚠️  Quelques ajustements peuvent être nécessaires"
    echo "   Vérifie les logs ci-dessus pour détails"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
