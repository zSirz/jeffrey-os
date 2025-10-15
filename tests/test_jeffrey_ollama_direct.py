#!/usr/bin/env python3
"""
Test direct de la connexion Jeffrey → Ollama
Vérifie que Jeffrey génère de vraies réponses (pas des stubs)
"""

import asyncio
import os
import sys
import time

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import ApertusClient qui gère la connexion Ollama
from src.jeffrey.core.llm.apertus_client import ApertusClient


def check_if_stub(response: str) -> bool:
    """Vérifie si la réponse est un stub/placeholder"""
    stub_patterns = [
        "Je suis Jeffrey",  # Trop générique
        "Hello, I am",
        "Error:",
        "STUB:",
        "DEFAULT_RESPONSE",
        "placeholder",
        "not implemented",
        "TODO:",
        "[MOCK]",
        "...",
    ]

    # Réponse trop courte = probablement stub
    if len(response) < 20:
        return True

    # Chercher les patterns de stub
    response_lower = response.lower()
    for pattern in stub_patterns:
        if pattern.lower() in response_lower:
            return True

    # Réponse identique répétée = stub
    words = response.split()
    if len(set(words)) < 5:  # Moins de 5 mots uniques
        return True

    return False


async def test_direct_ollama():
    """Test direct avec ApertusClient → Ollama"""
    print("\n" + "=" * 70)
    print("🧪 TEST DIRECT OLLAMA → JEFFREY")
    print("=" * 70)

    # Créer le client
    print("\n📦 Initialisation d'ApertusClient...")
    client = ApertusClient()

    # Questions de test variées
    test_questions = [
        {
            "prompt": "Bonjour, je m'appelle David. Peux-tu te présenter ?",
            "expected": "réponse personnalisée avec prénom David",
        },
        {
            "prompt": "Quelle est la capitale de la France et pourquoi est-elle importante ?",
            "expected": "Paris avec explication",
        },
        {
            "prompt": "Raconte-moi une blague courte sur les programmeurs",
            "expected": "blague humoristique",
        },
        {
            "prompt": "Explique en termes simples ce qu'est un réseau de neurones",
            "expected": "explication technique simplifiée",
        },
        {
            "prompt": "Si tu étais un animal, lequel serais-tu et pourquoi ?",
            "expected": "réponse créative et personnelle",
        },
    ]

    results = []

    for i, test in enumerate(test_questions, 1):
        print(f"\n--- Test {i}/5 ---")
        print(f"👤 Question: {test['prompt'][:80]}...")
        print(f"📝 Attendu: {test['expected']}")

        try:
            start = time.time()

            # Appel direct à Ollama via ApertusClient
            response = await client.chat(
                system_prompt="Tu es Jeffrey, un assistant AI intelligent et créatif.",
                user_message=test["prompt"],
                max_tokens=200,
                temperature=0.7,
            )

            elapsed = (time.time() - start) * 1000

            # Extraire le texte de la réponse (chat retourne un tuple)
            if isinstance(response, tuple):
                text = response[0]  # Le premier élément est le texte
            elif isinstance(response, dict):
                text = response.get("content", response.get("text", str(response)))
            else:
                text = str(response)

            # Vérifier si c'est un stub
            is_stub = check_if_stub(text)

            if is_stub:
                print("⚠️ STUB DÉTECTÉ!")
                print(f"   Réponse: {text[:100]}")
            else:
                print("✅ VRAIE RÉPONSE GÉNÉRÉE!")
                print(f"🤖 Jeffrey: {text[:200]}{'...' if len(text) > 200 else ''}")

            print(f"⏱️ Temps: {elapsed:.1f}ms")
            print(f"📏 Longueur: {len(text)} caractères")

            results.append(
                {
                    "question": test["prompt"][:50],
                    "is_stub": is_stub,
                    "response_length": len(text),
                    "latency_ms": elapsed,
                    "response_preview": text[:100],
                }
            )

        except Exception as e:
            print(f"❌ Erreur: {e}")
            results.append(
                {
                    "question": test["prompt"][:50],
                    "is_stub": True,
                    "response_length": 0,
                    "latency_ms": 0,
                    "error": str(e),
                }
            )

        # Petite pause entre les requêtes
        await asyncio.sleep(0.5)

    # Résumé des résultats
    print("\n" + "=" * 70)
    print("📊 ANALYSE DES RÉPONSES")
    print("=" * 70)

    total_tests = len(results)
    real_responses = sum(1 for r in results if not r["is_stub"])
    stub_responses = total_tests - real_responses

    print("\n📈 Statistiques:")
    print(f"   Total de tests: {total_tests}")
    print(f"   ✅ Vraies réponses: {real_responses}")
    print(f"   ⚠️ Stubs/Erreurs: {stub_responses}")

    if real_responses > 0:
        avg_length = sum(r["response_length"] for r in results if not r["is_stub"]) / real_responses
        avg_latency = sum(r["latency_ms"] for r in results if not r["is_stub"]) / real_responses
        print(f"   📏 Longueur moyenne: {avg_length:.0f} caractères")
        print(f"   ⏱️ Latence moyenne: {avg_latency:.1f}ms")

    # Détail par question
    print("\n📋 Détail par question:")
    for i, r in enumerate(results, 1):
        status = "❌ STUB" if r["is_stub"] else "✅ REAL"
        print(f"   {i}. {status} - {r['response_length']} chars - {r.get('latency_ms', 0):.0f}ms")
        if "error" in r:
            print(f"      Erreur: {r['error']}")

    # Diagnostic final
    print("\n" + "=" * 70)
    if real_responses == total_tests:
        print("🎉 SUCCÈS TOTAL - Ollama génère de vraies réponses!")
        print("Jeffrey peut maintenant tenir de vraies conversations!")
    elif real_responses > 0:
        print("⚠️ SUCCÈS PARTIEL - Certaines réponses sont générées")
        print("Vérifier la configuration pour améliorer la fiabilité")
    else:
        print("❌ ÉCHEC - Aucune vraie réponse générée")
        print("Les réponses sont des stubs ou des erreurs")
        print("\nActions recommandées:")
        print("1. Vérifier que Ollama tourne: ollama serve")
        print("2. Vérifier le modèle: ollama list")
        print("3. Tester manuellement: ollama run mistral:7b-instruct")


async def test_conversation_continuity():
    """Test de continuité de conversation"""
    print("\n" + "=" * 70)
    print("🧪 TEST DE CONTINUITÉ DE CONVERSATION")
    print("=" * 70)

    client = ApertusClient()

    # Conversation avec contexte
    conversation = [
        "Je m'appelle Alice et j'aime la programmation Python",
        "Quel est mon prénom ?",  # Doit se rappeler "Alice"
        "Et qu'est-ce que j'aime faire ?",  # Doit se rappeler "programmation Python"
    ]

    context = []

    for i, message in enumerate(conversation, 1):
        print(f"\n--- Tour {i} ---")
        print(f"👤 User: {message}")

        # Construire le prompt avec contexte
        full_prompt = ""
        if context:
            full_prompt = "Contexte de la conversation:\n"
            for turn in context:
                full_prompt += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n"
            full_prompt += f"\nUser: {message}\nAssistant:"
        else:
            full_prompt = message

        try:
            response = await client.chat(
                system_prompt="Tu es Jeffrey. Tu te souviens de la conversation précédente.",
                user_message=message,  # Just the current message, context is in system prompt
                max_tokens=150,
                temperature=0.7,
            )

            # chat retourne un tuple (text, metadata)
            if isinstance(response, tuple):
                text = response[0]
            elif isinstance(response, dict):
                text = response.get("content", response.get("text", str(response)))
            else:
                text = str(response)

            is_stub = check_if_stub(text)

            if not is_stub:
                print(f"🤖 Jeffrey: {text[:200]}")

                # Vérifier la mémoire contextuelle
                if i == 2 and "alice" in text.lower():
                    print("✅ Mémoire OK - Se souvient du prénom!")
                elif i == 3 and ("python" in text.lower() or "programm" in text.lower()):
                    print("✅ Mémoire OK - Se souvient de l'activité!")
            else:
                print(f"⚠️ Stub détecté: {text[:100]}")

            # Ajouter au contexte
            context.append({"user": message, "assistant": text})

        except Exception as e:
            print(f"❌ Erreur: {e}")


async def main():
    """Lance tous les tests"""
    print("\n" + "=" * 70)
    print("🚀 TEST JEFFREY → OLLAMA (DÉTECTION DE STUBS)")
    print("=" * 70)

    # Info système
    print("\n📋 Configuration:")
    print(f"   Python: {sys.version.split()[0]}")
    print(f"   Répertoire: {os.getcwd()}")

    try:
        # Test principal
        await test_direct_ollama()

        # Test de continuité
        await test_conversation_continuity()

        print("\n" + "=" * 70)
        print("✅ TESTS TERMINÉS")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n⏹️ Tests interrompus")
    except Exception as e:
        print(f"\n❌ Erreur critique: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
