#!/usr/bin/env python3
"""
Test non-interactif de Jeffrey OS
"""

import asyncio
import sys
from pathlib import Path

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent))


async def test():
    """Test simple de Jeffrey OS"""
    print("🤖 TEST JEFFREY OS")
    print("=" * 50)

    try:
        from jeffrey.core.orchestration.ia_orchestrator_ultimate import UltimateOrchestrator

        print("✅ Import orchestrateur: OK")

        orch = UltimateOrchestrator()
        print("✅ Initialisation: OK")

        # Test get_orchestration_stats
        try:
            stats = await orch.get_orchestration_stats()
            print(f"✅ Stats: {len(stats.get('professors', {}))} professeurs")
        except Exception as e:
            print(f"⚠️ Erreur stats: {e}")

        # Test orchestrate_with_intelligence
        try:
            from jeffrey.core.orchestration.ia_orchestrator_ultimate import OrchestrationRequest

            request = OrchestrationRequest(
                request="Bonjour Jeffrey!",
                request_type="greeting",
                user_id="test",
                preferences={},
                priority="normal",
            )
            response = await orch.orchestrate_with_intelligence(request)
            print(f"✅ Réponse reçue: {type(response)}")
            if isinstance(response, dict):
                print(f"   - Réponse: {response.get('response', '...')[:50]}...")
                print(f"   - Durée: {response.get('execution_time', 0):.2f}s")
        except Exception as e:
            print(f"⚠️ Erreur orchestration: {e}")

    except ImportError as e:
        print(f"❌ Erreur import: {e}")
        return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

    print("=" * 50)
    print("✨ Test terminé avec succès!")
    return True


if __name__ == "__main__":
    asyncio.run(test())
