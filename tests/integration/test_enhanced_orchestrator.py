#!/usr/bin/env python3
"""
Test de l'orchestrateur amélioré avec sélection dynamique par capacités
"""

from core.orchestration.enhanced_orchestrator import EnhancedOrchestrator
import sys
import os

# Configuration des imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class MockIAClient:
    """Client IA simulé pour les tests"""

    def __init__(self, name: str, style: str):
        self.name = name
        self.style = style

    async def ask(self, prompt: str) -> str:
        """Simule une réponse selon le style du client"""
        responses = {
            "creative": f"✨ {self.name}: Imagine un monde où {prompt}... Ce serait fascinant !",
            "analytical": f"🔍 {self.name}: Analysons cela. {prompt} peut s'expliquer par...",
            "empathetic": f"💙 {self.name}: Je comprends ce que tu ressens. {prompt}. Je suis là.",
            "technical": f"💻 {self.name}: Voici la solution technique pour {prompt}: [code]",
        }
            return responses.get(self.style, f"{self.name}: Réponse à {prompt}")


            def test_enhanced_orchestrator():
    """Test l'orchestrateur avec sélection dynamique"""

    print("🎯 Test de l'Orchestrateur Amélioré avec Capacités\n")

    # Créer des clients simulés
    mock_clients = {
        "grok": MockIAClient("Grok", "creative"),
        "gpt-4": MockIAClient("GPT-4", "analytical"),
        "claude": MockIAClient("Claude", "empathetic"),
    }

    # Créer l'orchestrateur
    orchestrator = EnhancedOrchestrator(ia_clients=mock_clients)

    # Cas de test avec différents types de prompts
    test_cases = [
        # (prompt, capacités attendues, providers attendus)
        ("Imagine un monde où les voitures volent", ["creative", "narrative"], ["grok"]),
        ("Explique-moi comment fonctionne un ordinateur", ["analytical", "technical"], ["gpt-4"]),
        ("Je me sens triste aujourd'hui 😢", ["empathetic", "emotional"], ["claude"]),
        ("Raconte-moi une blague drôle", ["humorous", "creative"], ["grok"]),
        ("Quel est le sens de la vie ?", ["philosophical", "analytical"], ["gpt-4"]),
        ("J'ai un bug dans mon code Python", ["technical", "analytical"], ["gpt-4"]),
        ("Je t'aime tellement ❤️", ["emotional", "empathetic"], ["claude", "grok"]),
        (
            "Comment puis-je être plus heureux ?",
            ["empathetic", "philosophical"],
            ["claude", "gpt-4"],
        ),
    ]

    print("📝 Test de détection des capacités:\n")

                for prompt, expected_caps, expected_providers in test_cases:
        print(f'Prompt: "{prompt}"')

        # Analyser les capacités
        analysis = orchestrator.get_capability_analysis(prompt)

        print(f"➤ Capacités détectées: {analysis['required_capabilities']}")
        print(f"➤ Attendu: {expected_caps}")
        print(
            f"➤ Scores providers: {dict(sorted(analysis['provider_scores'].items(), key=lambda x: x[1], reverse=True))}"
        )
        print(f"➤ Recommandés: {[p[0] for p in analysis['recommended_providers']]}")
        print(
            f"➤ Émotion: {
                analysis['emotional_context']['emotion']} (confiance: {
                analysis['emotional_context']['confidence']:.0%})"
        )

        # Vérifier si au moins un provider attendu est recommandé
        recommended = [p[0] for p in analysis["recommended_providers"]]
        match = any(p in recommended for p in expected_providers)
        print(f"➤ Match: {'✅' if match else '❌'}")

        print()

    # Test de l'orchestration intelligente
    print("\n🤖 Test de sélection dynamique des IA:\n")

    dynamic_tests = [
        "Imagine une histoire créative sur des robots",
        "Explique-moi la théorie de la relativité",
        "J'ai peur de l'échec, aide-moi",
    ]

                    for prompt in dynamic_tests:
        print(f'Prompt: "{prompt}"')

        # Utiliser ask_smart (version simulée sans asyncio pour le test)
        caps = orchestrator.detect_required_capabilities(prompt)
        scores = orchestrator.score_providers_by_capability(caps)

        print(f"➤ Capacités: {[c.value for c in caps]}")
        print(f"➤ Sélection: {sorted(scores.items(), key=lambda x: x[1], reverse=True)[0][0]}")
        print()


                        if __name__ == "__main__":
    test_enhanced_orchestrator()
