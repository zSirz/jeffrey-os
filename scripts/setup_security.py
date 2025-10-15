#!/usr/bin/env python3
"""
Setup Security - Initialisation des composants de sécurité
"""

import asyncio
import logging
import sys
from pathlib import Path

# Ajouter le projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from jeffrey.core.neuralbus.registry import registry
from jeffrey.core.security.signer.signer_eddsa import SignerEdDSA

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_security():
    """Configure les composants de sécurité"""
    print("🔧 Configuration de la sécurité Jeffrey OS...")
    print("-" * 50)

    # 1. Créer les répertoires nécessaires
    dirs = ["keys", "models", "reports", "logs"]
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
    print(f"✅ Répertoires créés: {', '.join(dirs)}")

    # 2. Générer les clés EdDSA si nécessaires
    key_path = Path("keys/eddsa_private.pem")
    if not key_path.exists():
        print("\n🔑 Génération des clés EdDSA...")
        signer = SignerEdDSA()
        await signer.start()
        await signer.save_keys("keys/eddsa_private.pem", "keys/eddsa_public.pem")
        print("✅ Clés EdDSA générées")
    else:
        print("📝 Clés EdDSA existantes")

    # 3. Charger le registry
    print("\n📋 Chargement du registry...")
    if registry.load_registry():
        status = registry.get_status()
        print(f"✅ Registry chargé: {status['services']} services")

        # Afficher les services critiques
        critical = registry.get_critical_services()
        if critical:
            print(f"\n⚠️  Services critiques: {', '.join(critical)}")
    else:
        print("❌ Échec du chargement du registry")

    # 4. Créer les fichiers de configuration
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(
                """# Jeffrey OS Environment
SECURITY_MODE=dev
REDIS_URL=redis://localhost:6379
LOG_LEVEL=INFO
"""
            )
        print("\n✅ Fichier .env créé")

    # 5. Vérifier les dépendances Python
    print("\n📦 Vérification des dépendances...")
    try:
        import cryptography
        import fastapi
        import pyarrow
        import redis

        print("✅ Toutes les dépendances critiques sont installées")
    except ImportError as e:
        print(f"❌ Dépendance manquante: {e}")
        print("   Exécutez: pip install -r requirements.txt")
        return False

    # 6. Test de connectivité Redis (optionnel)
    print("\n🔄 Test Redis...")
    try:
        import redis.asyncio as redis_async

        client = await redis_async.from_url("redis://localhost:6379", socket_connect_timeout=2)
        await client.ping()
        await client.close()
        print("✅ Redis accessible")
    except Exception:
        print("📝 Redis non disponible (normal en mode DEV)")

    print("\n" + "=" * 50)
    print("✅ Setup de sécurité terminé !")
    print("\nProchaines étapes:")
    print("  1. make start     - Démarrer l'API")
    print("  2. make smoke     - Tests smoke")
    print("  3. make test      - Tests complets")

    return True


if __name__ == "__main__":
    success = asyncio.run(setup_security())
    sys.exit(0 if success else 1)
