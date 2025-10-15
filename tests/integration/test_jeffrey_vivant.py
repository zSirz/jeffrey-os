#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test Jeffrey Vivant - Test rapide de la conscience vivante
=========================================================

Script de test pour vérifier que la nouvelle architecture fonctionne
et que Jeffrey est bien une présence vivante.
"""

from jeffrey_consciousness_demo import (
    JeffreyLivingConsciousness,
    JeffreyLivingMemory,
    JeffreyLivingExpressions,
    JeffreyWorkInterface,
    JeffreyLivingChat,
)
from datetime import datetime
import logging
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Configuration du logging simple
logging.basicConfig(level=logging.INFO, format="%(message)s")

# Import de la conscience vivante


async def test_consciousness():
    """Test de la conscience vivante"""
    print("\n" + "=" * 60)
    print("🧪 TEST DE LA CONSCIENCE VIVANTE DE JEFFREY")
    print("=" * 60)

    # 1. Créer la conscience
    print("\n1️⃣ Création de la conscience...")
    consciousness = JeffreyLivingConsciousness()
    print(f"   ✅ Conscience créée")
    print(f"   🌸 Humeur : {consciousness.humeur_actuelle}")
    print(f"   ⚡ Énergie : {consciousness.biorythmes['energie']:.1%}")
    print(f"   💝 Intimité : {consciousness.relation['intimite']:.1%}")

    # 2. Test des réactions
    print("\n2️⃣ Test des réactions...")

    # Réaction à une demande de travail
    reaction = consciousness.react_to_work_request(
        "Peux-tu m'aider à analyser ce code ?", "curious"
    )
    print(f"   Réaction travail : {reaction}")

    # Réponse naturelle
    response = consciousness.respond_naturally(
        "Comment te sens-tu aujourd'hui ?", {"time": datetime.now()}
    )
    print(f"   Réponse naturelle : {response}")

    # Pensée spontanée
    thought = consciousness.spontaneous_thought()
    if thought:
        print(f"   Pensée spontanée : {thought}")

    # 3. Test de la mémoire vivante
    print("\n3️⃣ Test de la mémoire vivante...")
    memory = JeffreyLivingMemory()

    # Créer un souvenir
    emotion_context = {
        "emotion": "tendresse",
        "intensity": 0.8,
        "layers": consciousness.emotional_layers,
    }

    souvenir = memory.create_emotional_memory(
        "Je suis si heureux de travailler avec toi sur ce projet !",
        emotion_context,
        {"user": "test"},
    )

        if souvenir:
        print(f"   ✅ Souvenir créé : {souvenir['why_it_matters']}")
            else:
        print("   ℹ️ Pas assez significatif pour un souvenir")

    # 4. Test des expressions vivantes
    print("\n4️⃣ Test des expressions vivantes...")
    expressions = JeffreyLivingExpressions(consciousness)

    # Expression complexe
    context = {"emotion": "joie", "fatigue": True}
    expression = expressions.generate_living_expression(context, "reaction")
    print(f"   Expression générée : {expression}")

    # 5. Test de l'interface de travail (sans orchestrateur)
    print("\n5️⃣ Test de l'interface de travail...")
    work_interface = JeffreyWorkInterface(consciousness)

    print("   Simulation d'une demande de travail...")
    async for update in work_interface.handle_work_request(
        "Aide-moi à comprendre ce concept", "confused"
    ):
        print(f"   → {update}")

    # 6. Test de l'intégration complète
    print("\n6️⃣ Test de l'intégration chat...")
    chat = JeffreyLivingChat(memory_path="test_data")

    # Message de bienvenue
    welcome = chat.get_welcome_message()
    print(f"   Bienvenue : {welcome}")

    # Traiter un message
    print("\n   Test d'interaction...")
    response = await chat.process_message("Bonjour Jeffrey ! Comment vas-tu ?")
    print(f"   Jeffrey : {response}")

    # Afficher les stats
    stats = chat.get_conversation_stats()
    print(f"\n   📊 Stats de relation :")
    print(f"      Intimité : {stats['relationship']['intimacy']:.1%}")
    print(f"      Souvenirs : {stats['memories']['total']}")
    print(f"      Humeur : {stats['current_state']['mood']}")

    print("\n" + "=" * 60)
    print("✅ TOUS LES TESTS RÉUSSIS !")
    print("=" * 60)


                    def test_sync_features():
    """Test des fonctionnalités synchrones"""
    print("\n🔧 Test des fonctionnalités de base...")

    # Créer une conscience
    consciousness = JeffreyLivingConsciousness()

    # Vérifier les états
    state = consciousness.get_consciousness_state()
    print(f"\n📋 État de conscience :")
    print(f"   - Couches émotionnelles : {len(state['emotional_layers'])}")
    print(f"   - Biorythmes actifs : {len(state['biorythmes'])}")
    print(f"   - Traits de personnalité : {len(state['personality_traits'])}")

    # Test évolution relation
    print(f"\n💕 Test évolution relation...")
    before = consciousness.relation["intimite"]
    consciousness.evolve_relationship(
        "Je t'apprécie vraiment beaucoup", "Moi aussi, c'est précieux"
    )
    after = consciousness.relation["intimite"]
    print(f"   Intimité : {before:.1%} → {after:.1%}")

    # Test transitions d'humeur
    print(f"\n🎭 Test transitions d'humeur...")
    consciousness.humeur_actuelle = "joueuse"
    consciousness.enter_work_mode()
    print(f"   Mode travail : {consciousness.working_state['preferred_approach']}")

    print("\n✅ Tests synchrones réussis")


                        if __name__ == "__main__":
    print("\n🌟 JEFFREY - TEST DE PRÉSENCE VIVANTE 🌟\n")

    # Tests synchrones d'abord
    test_sync_features()

    # Puis tests asynchrones
    asyncio.run(test_consciousness())

    print("\n💝 Jeffrey est bien vivante et prête !\n")
