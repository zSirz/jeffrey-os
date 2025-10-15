#!/usr/bin/env python3
"""
Smoke Boot - Tests de fumée pour vérifier que le système démarre
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path

# Ajouter le projet au path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SmokeTests:
    """Tests de fumée rapides (20 secondes max)"""

    def __init__(self):
        self.results = {"passed": [], "failed": [], "warnings": [], "start_time": time.time()}

    async def test_imports(self):
        """Test : Tous les imports critiques fonctionnent"""
        print("\n🔍 Test des imports...")

        critical_imports = [
            ("FastAPI", "fastapi", "FastAPI"),
            ("Redis", "redis.asyncio", "Redis"),
            ("Cryptography", "cryptography.hazmat.primitives.asymmetric.ed25519", None),
            ("PyArrow", "pyarrow", None),
            ("Prometheus", "prometheus_client", None),
            ("Uvicorn", "uvicorn", None),
        ]

        for name, module, attr in critical_imports:
            try:
                imported = __import__(module, fromlist=[attr] if attr else [])
                if attr:
                    getattr(imported, attr)
                self.results["passed"].append(f"Import {name}")
                print(f"  ✅ {name}")
            except ImportError as e:
                self.results["failed"].append(f"Import {name}: {e}")
                print(f"  ❌ {name}: {e}")

    async def test_registry(self):
        """Test : Le registry se charge correctement"""
        print("\n📋 Test du registry...")

        try:
            from jeffrey.core.neuralbus.registry import registry

            if registry.load_registry():
                status = registry.get_status()
                services = status.get("services", 0)
                if services > 0:
                    self.results["passed"].append(f"Registry: {services} services")
                    print(f"  ✅ Registry chargé: {services} services")
                else:
                    self.results["failed"].append("Registry: 0 services")
                    print("  ❌ Registry vide")
            else:
                self.results["failed"].append("Registry: échec chargement")
                print("  ❌ Échec du chargement")

        except Exception as e:
            self.results["failed"].append(f"Registry: {e}")
            print(f"  ❌ Erreur registry: {e}")

    async def test_security_components(self):
        """Test : Les composants de sécurité s'initialisent"""
        print("\n🔐 Test des composants de sécurité...")

        # Test Cache Guardian
        try:
            from jeffrey.core.security.cache_guardian import CacheGuardian

            cg = CacheGuardian()
            await cg.start()
            self.results["passed"].append("Cache Guardian")
            print("  ✅ Cache Guardian")
        except Exception as e:
            self.results["failed"].append(f"Cache Guardian: {e}")
            print(f"  ❌ Cache Guardian: {e}")

        # Test Anti-Replay
        try:
            from jeffrey.core.security.anti_replay import AntiReplaySystem

            ar = AntiReplaySystem()
            await ar.start()
            status = ar.get_status()
            if status["using_redis"]:
                self.results["passed"].append("Anti-Replay (Redis)")
                print("  ✅ Anti-Replay (Redis)")
            else:
                self.results["warnings"].append("Anti-Replay (fallback mémoire)")
                print("  ⚠️  Anti-Replay (fallback mémoire)")
        except Exception as e:
            self.results["failed"].append(f"Anti-Replay: {e}")
            print(f"  ❌ Anti-Replay: {e}")

        # Test mTLS Bridge
        try:
            from jeffrey.core.security.mtls_bridge import MTLSBridge

            mtls = MTLSBridge()
            await mtls.start()
            self.results["passed"].append("mTLS Bridge")
            print("  ✅ mTLS Bridge")
        except Exception as e:
            self.results["failed"].append(f"mTLS Bridge: {e}")
            print(f"  ❌ mTLS Bridge: {e}")

    async def test_ffi_bridge(self):
        """Test : Le FFI Bridge s'initialise"""
        print("\n🌉 Test du FFI Bridge...")

        try:
            from jeffrey.core.neuralbus.ffi_cdata import ZeroCopyFFIBridge

            bridge = ZeroCopyFFIBridge()
            bridge.start()
            status = bridge.get_status()

            if status["stub_mode"]:
                self.results["warnings"].append("FFI Bridge en mode stub")
                print("  ⚠️  FFI Bridge en mode stub (normal)")
            else:
                self.results["passed"].append("FFI Bridge (Rust chargé)")
                print("  ✅ FFI Bridge (Rust chargé)")

        except Exception as e:
            self.results["failed"].append(f"FFI Bridge: {e}")
            print(f"  ❌ FFI Bridge: {e}")

    async def test_guardians(self):
        """Test : Les Guardians s'initialisent"""
        print("\n🛡️  Test des Guardians...")

        try:
            from jeffrey.core.guardians.guardians_hub import GuardiansHub

            hub = GuardiansHub()
            await hub.start()
            status = hub.get_status()

            if status["initialized"]:
                self.results["passed"].append("Guardians Hub")
                print("  ✅ Guardians Hub initialisé")

                # Vérifier les composants
                components = status.get("components", {})
                for name, comp_status in components.items():
                    if comp_status.get("status") == "not loaded":
                        self.results["warnings"].append(f"Guardian {name} non chargé")
                        print(f"  ⚠️  {name}: non chargé")
                    else:
                        print(f"  ✅ {name}: disponible")
            else:
                self.results["failed"].append("Guardians Hub non initialisé")
                print("  ❌ Guardians Hub non initialisé")

        except Exception as e:
            self.results["failed"].append(f"Guardians Hub: {e}")
            print(f"  ❌ Guardians Hub: {e}")

    async def test_keys_and_certs(self):
        """Test : Les clés et certificats existent"""
        print("\n🔑 Test des clés et certificats...")

        # Vérifier les certificats serveur
        cert_files = [
            ("Certificat serveur", "keys/server.crt"),
            ("Clé serveur", "keys/server.key"),
            ("CA", "keys/ca.crt"),
        ]

        for name, path in cert_files:
            if Path(path).exists():
                self.results["passed"].append(name)
                print(f"  ✅ {name}")
            else:
                self.results["warnings"].append(f"{name} manquant")
                print(f"  ⚠️  {name} manquant (exécutez: bash scripts/mkcert_dev.sh)")

        # Vérifier les clés EdDSA
        if Path("keys/eddsa_private.pem").exists():
            self.results["passed"].append("Clés EdDSA")
            print("  ✅ Clés EdDSA")
        else:
            self.results["warnings"].append("Clés EdDSA manquantes")
            print("  ⚠️  Clés EdDSA manquantes")

    async def test_api_startup(self):
        """Test : L'API peut démarrer"""
        print("\n🚀 Test de démarrage API...")

        try:
            from jeffrey.core.control.control_plane import app

            # Vérifier que l'app existe et a les routes
            routes = [r.path for r in app.routes]
            expected = ["/health", "/ready", "/metrics", "/status"]

            for route in expected:
                if route in routes:
                    print(f"  ✅ Route {route}")
                else:
                    self.results["failed"].append(f"Route {route} manquante")
                    print(f"  ❌ Route {route} manquante")

            if all(r in routes for r in expected):
                self.results["passed"].append("API routes")

        except Exception as e:
            self.results["failed"].append(f"API: {e}")
            print(f"  ❌ Erreur API: {e}")

    async def run_all(self):
        """Exécute tous les tests de fumée"""
        print("=" * 50)
        print("🔥 TESTS DE FUMÉE JEFFREY OS")
        print("=" * 50)

        # Exécuter tous les tests
        await self.test_imports()
        await self.test_registry()
        await self.test_security_components()
        await self.test_ffi_bridge()
        await self.test_guardians()
        await self.test_keys_and_certs()
        await self.test_api_startup()

        # Calculer le temps total
        duration = time.time() - self.results["start_time"]

        # Afficher le résumé
        print("\n" + "=" * 50)
        print("📊 RÉSUMÉ")
        print("=" * 50)
        print(f"✅ Réussis:  {len(self.results['passed'])}")
        print(f"⚠️  Warnings: {len(self.results['warnings'])}")
        print(f"❌ Échecs:   {len(self.results['failed'])}")
        print(f"⏱️  Durée:    {duration:.1f}s")

        # Sauvegarder le rapport
        report_path = Path("reports/smoke_test.json")
        report_path.parent.mkdir(exist_ok=True)

        with open(report_path, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n📄 Rapport sauvé: {report_path}")

        # Déterminer le statut final
        if self.results["failed"]:
            print("\n❌ ÉCHEC: Des tests critiques ont échoué")
            return False
        elif self.results["warnings"]:
            print("\n⚠️  SUCCÈS PARTIEL: Système fonctionnel avec warnings")
            return True
        else:
            print("\n✅ SUCCÈS: Tous les tests passés !")
            return True


async def main():
    """Point d'entrée principal"""
    tester = SmokeTests()
    success = await tester.run_all()

    if success:
        print("\n🎉 Jeffrey OS est prêt à démarrer !")
        print("   Exécutez: make start")
    else:
        print("\n⚠️  Corrections nécessaires avant le démarrage")
        print("   Exécutez: make setup")

    return 0 if success else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
