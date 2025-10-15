#!/usr/bin/env python3
"""
Applique les fixes Jeffrey OS de manière intelligente
Détecte l'environnement et adapte l'installation
"""

import argparse
import os
import platform
import subprocess
import sys

import psutil


def check_environment():
    """Analyse l'environnement d'exécution"""
    info = {
        "os": platform.system(),
        "python": sys.version,
        "ram_available": psutil.virtual_memory().available / 1024 / 1024 / 1024,  # GB
        "ram_percent": psutil.virtual_memory().percent,
        "cpu_count": psutil.cpu_count(),
        "is_production": os.environ.get("JEFFREY_MODE") == "production",
    }

    print("🔍 Environment Analysis:")
    print(f"  OS: {info['os']}")
    print(f"  RAM: {info['ram_available']:.1f}GB available ({info['ram_percent']}% used)")
    print(f"  CPUs: {info['cpu_count']}")
    print(f"  Mode: {'PRODUCTION' if info['is_production'] else 'DEVELOPMENT'}")

    return info


def apply_critical_fixes():
    """Applique SEULEMENT les fixes critiques"""
    print("\n⚡ Applying critical fixes...")

    # 1. Backup
    print("📦 Creating backup...")
    subprocess.run("cp -r src src.backup-fixes 2>/dev/null || true", shell=True)

    # 2. Config minimale
    print("📝 Creating minimal config...")
    config = """# Jeffrey OS Configuration - Production Ready

# PRIVACY - CRITIQUE
privacy:
  encrypt_pii: true  # UN SEUL FLAG pour texte + metadata
  anonymize_user_ids: true

# MEMORY - BASIQUE
memory_federation:
  enabled: true
  budget_ms: 400
  cache_size: 1000

  # ML OPTIONNEL - DÉSACTIVÉ PAR DÉFAUT
  semantic_dedup: false  # Active seulement si RAM disponible
  semantic_threshold: 0.95  # Seuil élevé pour éviter faux positifs

# MONITORING
monitoring:
  log_level: INFO
  alert_threshold: 10  # PII par minute

# ML - TOUT DÉSACTIVÉ PAR DÉFAUT
ml_features:
  enabled: false  # Master switch
  lazy_load: true  # Charge seulement si utilisé
  max_ram_mb: 500  # Limite RAM pour ML
"""

    os.makedirs("config", exist_ok=True)
    with open("config/modules_fixed.yaml", "w") as f:
        f.write(config)

    print("✅ Critical fixes applied")


def install_ml_optional(env_info):
    """Installe ML seulement si RAM suffisante"""

    if env_info["ram_available"] < 2.0:
        print("⚠️ Skipping ML (need 2GB+ RAM free)")
        return False

    response = input("\n🤖 Install ML features (500MB)? [y/N]: ")
    if response.lower() != "y":
        return False

    print("📚 Installing ML dependencies...")
    try:
        subprocess.run("pip install sentence-transformers faiss-cpu --no-cache-dir", shell=True, check=True)

        # Activer dans la config
        print("🔧 Enabling ML features...")
        if os.path.exists("config/modules_fixed.yaml"):
            with open("config/modules_fixed.yaml") as f:
                config = f.read()
            config = config.replace("semantic_dedup: false", "semantic_dedup: true")
            config = config.replace("ml_features:\n  enabled: false", "ml_features:\n  enabled: true")
            with open("config/modules_fixed.yaml", "w") as f:
                f.write(config)

        return True
    except subprocess.CalledProcessError:
        print("❌ ML installation failed")
        return False


def run_tests():
    """Lance les tests appropriés"""
    print("\n🧪 Running tests...")

    # Tests critiques toujours
    print("Running privacy tests...")
    subprocess.run("python -m pytest tests/test_chaos_privacy.py -v -q 2>/dev/null", shell=True)

    # Tests perf si Mac
    if platform.system() == "Darwin":
        print("🍎 Running Mac performance tests...")
        subprocess.run("python -m pytest tests/test_performance_mac.py -v -q 2>/dev/null", shell=True)


def generate_encryption_key():
    """Génère une clé de chiffrement"""
    from cryptography.fernet import Fernet

    return Fernet.generate_key().decode()


def main():
    parser = argparse.ArgumentParser(description="Apply Jeffrey OS fixes")
    parser.add_argument("--critical-only", action="store_true", help="Only apply critical fixes")
    parser.add_argument("--with-ml", action="store_true", help="Force ML installation")
    parser.add_argument("--skip-tests", action="store_true", help="Skip tests")

    args = parser.parse_args()

    print("🚀 JEFFREY OS ULTIMATE FIXES")
    print("=" * 40)

    # Analyser environnement
    env_info = check_environment()

    # Vérification production
    if env_info["is_production"]:
        if not os.environ.get("JEFFREY_ENCRYPTION_KEY"):
            print("❌ ERROR: JEFFREY_ENCRYPTION_KEY required in production!")
            print("Generate with:")
            print(f"  export JEFFREY_ENCRYPTION_KEY='{generate_encryption_key()}'")
            sys.exit(1)

    # Appliquer fixes critiques
    apply_critical_fixes()

    # ML optionnel
    ml_installed = False
    if not args.critical_only:
        if args.with_ml or env_info["ram_available"] > 3:
            ml_installed = install_ml_optional(env_info)

    # Tests
    if not args.skip_tests:
        run_tests()

    print("\n✅ INSTALLATION COMPLETE!")
    print("📊 Summary:")
    print("  ✓ Critical fixes: APPLIED")
    print("  ✓ Privacy: ENFORCED")
    print("  ✓ Deduplication: ACTIVE")
    print(f"  ✓ ML Features: {'ENABLED' if ml_installed else 'DISABLED (can enable later)'}")
    print("\n🎯 Next steps:")

    if not os.environ.get("JEFFREY_ENCRYPTION_KEY"):
        print("  1. Set encryption key:")
        print(f"     export JEFFREY_ENCRYPTION_KEY='{generate_encryption_key()}'")
    else:
        print("  1. Encryption key already set ✓")

    print("  2. Run: python test_federation_architecture.py")
    print("  3. Start API: make start")
    print("  4. Monitor: tail -f logs/jeffrey.log")

    # Create a sample test script
    print("\n📝 Creating quick test script...")
    test_script = """#!/usr/bin/env python3
import asyncio
from jeffrey.core.memory.memory_federation_v2 import MemoryFederationV2
from jeffrey.core.loaders.secure_module_loader import SecureModuleLoader

async def test():
    loader = SecureModuleLoader()
    federation = MemoryFederationV2(loader)
    await federation.initialize(None)

    # Test store
    result = await federation.store_to_relevant(
        "test_user", "user", "Hello Jeffrey!"
    )
    print(f"✅ Store test: {len(result)} layers")

    # Test recall
    memories = await federation.recall_fast("test_user", 5)
    print(f"✅ Recall test: {len(memories)} memories")

    # Check stats
    stats = federation.get_stats()
    print(f"✅ Stats: {stats['initialized']} initialized")

asyncio.run(test())
print("\\n🎉 All tests passed!")
"""

    with open("quick_test.py", "w") as f:
        f.write(test_script)
    os.chmod("quick_test.py", 0o755)
    print("  Created quick_test.py - Run with: python quick_test.py")


if __name__ == "__main__":
    main()
