"""
Test de conversation complète avec Jeffrey
Vérifie que le système peut générer de vraies réponses via Ollama
"""

import asyncio
import os
import sys
import time

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.jeffrey.core.response.basal_ganglia_ucb1 import BasalGangliaScheduler
from src.jeffrey.core.response.neural_blackboard_v2 import NeuralBlackboard
from src.jeffrey.core.response.neural_response_orchestrator import NeuralResponseOrchestrator

# Chercher si IntelligentResponseGenerator existe
try:
    from src.jeffrey.core.consciousness.real_intelligence import IntelligentResponseGenerator

    print("✅ IntelligentResponseGenerator trouvé")
except ImportError:
    print("⚠️ IntelligentResponseGenerator non trouvé, utilisation du système par défaut")


class ConversationContext:
    """Contexte pour simuler une conversation"""

    def __init__(self, text: str, user_id: str = "test_user"):
        self.correlation_id = f"{user_id}_{int(time.time() * 1000)}"
        self.user_input = text
        self.intent = "conversation"
        self.user_id = user_id
        self.signal = type("Signal", (), {"start_time": time.time(), "deadline_absolute": time.time() + 30})()

    def remaining_budget_ms(self):
        return 10000  # 10 secondes de budget


async def test_single_response():
    """Test d'une réponse unique"""
    print("\n" + "=" * 60)
    print("🧪 TEST 1: RÉPONSE UNIQUE")
    print("=" * 60)

    # Initialiser le système
    blackboard = NeuralBlackboard(max_memory_mb=64)
    scheduler = BasalGangliaScheduler()
    orchestrator = NeuralResponseOrchestrator(None, blackboard, scheduler)

    # Question de test
    question = "Bonjour Jeffrey, présente-toi en quelques mots."
    print(f"\n👤 User: {question}")

    context = ConversationContext(question)
    start = time.time()

    try:
        result = await orchestrator.process(context)
        elapsed = (time.time() - start) * 1000

        # Récupérer la réponse de différents endroits possibles
        response = None

        # Chercher dans différents attributs
        if hasattr(result, "final_response"):
            response = result.final_response
        elif hasattr(result, "final_text"):
            response = result.final_text
        elif hasattr(result, "text"):
            response = result.text
        elif hasattr(result, "response"):
            response = result.response

        # Si pas de réponse directe, chercher dans les phases
        if not response and hasattr(result, "phase_results"):
            for phase, phase_result in result.phase_results.items():
                if hasattr(phase_result, "results"):
                    for module_id, module_result in phase_result.results.items():
                        if isinstance(module_result, dict) and "response" in module_result:
                            response = module_result["response"]
                            break
                if response:
                    break

        if response:
            print(f"\n🤖 Jeffrey: {response}")
            print(f"\n⏱️ Temps de réponse: {elapsed:.1f}ms")
            print("✅ Réponse générée avec succès!")
        else:
            print("\n❌ Pas de réponse générée")
            print(f"   Result type: {type(result)}")
            print(f"   Result attributes: {dir(result)}")
            if hasattr(result, "__dict__"):
                print(f"   Result content: {result.__dict__}")

    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback

        traceback.print_exc()


async def test_conversation_flow():
    """Test d'une conversation multi-tours"""
    print("\n" + "=" * 60)
    print("🧪 TEST 2: CONVERSATION MULTI-TOURS")
    print("=" * 60)

    blackboard = NeuralBlackboard(max_memory_mb=64)
    scheduler = BasalGangliaScheduler()
    orchestrator = NeuralResponseOrchestrator(None, blackboard, scheduler)

    # Scénario de conversation
    conversation = [
        "Bonjour Jeffrey, je m'appelle David",
        "Je suis développeur et je travaille sur des projets d'IA",
        "Qu'est-ce que tu peux faire pour m'aider dans mes projets ?",
        "Peux-tu me donner un exemple concret ?",
        "Merci, c'était très utile !",
    ]

    user_id = "david_test"
    conversation_history = []

    for i, message in enumerate(conversation, 1):
        print(f"\n--- Tour {i} ---")
        print(f"👤 David: {message}")

        context = ConversationContext(message, user_id)

        try:
            start = time.time()
            result = await orchestrator.process(context)
            elapsed = (time.time() - start) * 1000

            # Extraire la réponse
            response = getattr(result, "final_response", None) or getattr(result, "text", None) or "..."

            print(f"🤖 Jeffrey: {response[:300]}{'...' if len(response) > 300 else ''}")
            print(f"⏱️ {elapsed:.1f}ms")

            # Ajouter à l'historique
            conversation_history.append({"user": message, "assistant": response, "latency_ms": elapsed})

            # Petite pause entre les tours
            await asyncio.sleep(0.5)

        except Exception as e:
            print(f"❌ Erreur tour {i}: {e}")

    # Résumé de la conversation
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DE LA CONVERSATION")
    print("=" * 60)
    print(f"Tours complétés: {len(conversation_history)}/{len(conversation)}")
    if conversation_history:
        avg_latency = sum(h["latency_ms"] for h in conversation_history) / len(conversation_history)
        print(f"Latence moyenne: {avg_latency:.1f}ms")


async def test_different_intents():
    """Test avec différents types d'intentions"""
    print("\n" + "=" * 60)
    print("🧪 TEST 3: DIFFÉRENTES INTENTIONS")
    print("=" * 60)

    blackboard = NeuralBlackboard(max_memory_mb=64)
    scheduler = BasalGangliaScheduler()
    orchestrator = NeuralResponseOrchestrator(None, blackboard, scheduler)

    # Différents types de questions
    test_cases = [
        ("Quelle est la capitale de la France ?", "question"),
        ("Je me sens un peu triste aujourd'hui", "emotion"),
        ("Raconte-moi une blague", "entertainment"),
        ("Comment fonctionne un réseau de neurones ?", "technical"),
        ("Au revoir Jeffrey !", "farewell"),
    ]

    results = []

    for question, intent_type in test_cases:
        print(f"\n[{intent_type.upper()}]")
        print(f"👤 User: {question}")

        context = ConversationContext(question)
        context.intent = intent_type

        try:
            start = time.time()
            result = await orchestrator.process(context)
            elapsed = (time.time() - start) * 1000

            response = getattr(result, "final_response", None) or getattr(result, "text", None) or "Pas de réponse"

            print(f"🤖 Jeffrey: {response[:200]}{'...' if len(response) > 200 else ''}")

            results.append({"intent": intent_type, "success": len(response) > 0, "latency_ms": elapsed})

        except Exception as e:
            print(f"❌ Erreur: {e}")
            results.append({"intent": intent_type, "success": False, "latency_ms": 0})

    # Résumé
    print("\n📊 Résultats par intention:")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"  {status} {r['intent']}: {r['latency_ms']:.1f}ms")


async def check_ollama_connection():
    """Vérifie la connexion à Ollama"""
    print("\n🔍 Vérification de la connexion Ollama...")

    import aiohttp

    # Essayer différents ports où Ollama pourrait être
    ports = [11434, 9010, 8080]

    for port in ports:
        try:
            async with aiohttp.ClientSession() as session:
                # Essayer l'endpoint standard Ollama
                async with session.get(
                    f"http://localhost:{port}/api/tags", timeout=aiohttp.ClientTimeout(total=2)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        models = data.get("models", [])
                        print(f"✅ Ollama trouvé sur port {port}")
                        if models:
                            print("   Modèles disponibles:")
                            for model in models:
                                size = model.get("size", 0) / (1024**3)  # Convert to GB
                                print(f"     - {model['name']} ({size:.1f}GB)")
                        return True
        except:
            continue

    print("❌ Ollama non accessible")
    print("   Lancez: ollama serve")
    return False


async def main():
    """Lance tous les tests"""
    print("\n" + "=" * 70)
    print("🚀 TEST DE CONVERSATION AVEC JEFFREY")
    print("=" * 70)

    # Vérifier Ollama d'abord
    if not await check_ollama_connection():
        print("\n⚠️ Ollama n'est pas accessible. Les réponses seront limitées.")
        print("Pour de meilleures réponses, lancez: ollama serve")
        response = input("\nContinuer quand même ? (o/n): ")
        if response.lower() != "o":
            return

    # Lancer les tests
    try:
        await test_single_response()
        await asyncio.sleep(1)

        await test_conversation_flow()
        await asyncio.sleep(1)

        await test_different_intents()

        print("\n" + "=" * 70)
        print("✅ TESTS TERMINÉS")
        print("=" * 70)

        print("\n📌 Prochaines étapes:")
        print("1. Si les réponses sont vides → vérifier la connexion Ollama")
        print("2. Si les réponses sont génériques → vérifier le modèle Mistral")
        print("3. Si tout fonctionne → créer l'interface web!")

    except KeyboardInterrupt:
        print("\n\n⏹️ Tests interrompus")
    except Exception as e:
        print(f"\n❌ Erreur inattendue: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Afficher les informations système
    print("\n📋 Informations système:")
    print(f"   Python: {sys.version}")
    print(f"   Répertoire: {os.getcwd()}")
    print(f"   PYTHONPATH: {sys.path[0]}")

    # Lancer les tests
    asyncio.run(main())
