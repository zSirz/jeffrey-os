#!/usr/bin/env python3
"""
Test rapide de l'installation de Jeffrey OS
Vérifie que tous les composants critiques sont disponibles
"""

import importlib
import sys


def test_import(module: str, name: str) -> tuple[bool, str]:
    """Test l'import d'un module"""
    try:
        importlib.import_module(module)
        return True, f"✅ {name}"
    except ImportError as e:
        return False, f"❌ {name}: {e}"


def test_jeffrey_modules():
    """Test les modules Jeffrey internes"""
    modules = [
        ("src.jeffrey.runtime", "Jeffrey Runtime"),
        ("src.jeffrey.core.response.neural_response_orchestrator", "Neural Orchestrator"),
        ("src.jeffrey.core.response.neural_blackboard_v2", "Neural Blackboard"),
        ("src.jeffrey.interfaces.ui.jeffrey_ui_bridge", "UI Bridge V3"),
    ]

    all_ok = True
    for module, name in modules:
        ok, msg = test_import(module, name)
        print(msg)
        if not ok:
            all_ok = False

    return all_ok


def test_external_deps():
    """Test les dépendances externes"""
    deps = [
        ("httpx", "HTTP Client"),
        ("networkx", "Graph Library"),
        ("kivy", "UI Framework"),
        ("msgpack", "Binary Serialization"),
        ("pydantic", "Data Validation"),
        ("numpy", "Numerical Computing"),
        ("redis", "Cache Storage"),
        ("openai", "LLM Interface"),
    ]

    all_ok = True
    for module, name in deps:
        ok, msg = test_import(module, name)
        print(msg)
        if not ok:
            all_ok = False

    return all_ok


def test_ollama_connection():
    """Test la connexion à Ollama"""
    try:
        import httpx

        response = httpx.get("http://localhost:9010/api/tags", timeout=2)
        if response.status_code == 200:
            return True, "✅ Ollama connected"
        else:
            return False, f"⚠️ Ollama returned {response.status_code}"
    except Exception as e:
        return False, f"⚠️ Ollama not available: {e}"


def test_redis_connection():
    """Test la connexion Redis"""
    try:
        import redis

        r = redis.Redis(host="localhost", port=6379, decode_responses=True)
        r.ping()
        return True, "✅ Redis connected"
    except Exception:
        return False, "⚠️ Redis not available (optional)"


def main():
    print("=" * 60)
    print("🔍 Jeffrey OS - Installation Test")
    print("=" * 60)

    # 1. External dependencies
    print("\n📦 External Dependencies:")
    ext_ok = test_external_deps()

    # 2. Jeffrey modules
    print("\n🧠 Jeffrey Modules:")
    jeff_ok = test_jeffrey_modules()

    # 3. Services
    print("\n🌐 External Services:")
    ollama_ok, ollama_msg = test_ollama_connection()
    print(ollama_msg)

    redis_ok, redis_msg = test_redis_connection()
    print(redis_msg)

    # Summary
    print("\n" + "=" * 60)

    if ext_ok and jeff_ok:
        print("✅ Core installation: OK")
    else:
        print("❌ Core installation: FAILED")
        print("   Run: pip install -r requirements.txt")

    if ollama_ok:
        print("✅ LLM capability: OK")
    else:
        print("⚠️ LLM capability: Not available")
        print("   Install Ollama for conversation features")

    if redis_ok:
        print("✅ Cache system: OK")
    else:
        print("⚠️ Cache system: Not available (optional)")

    # Test simple runtime
    if ext_ok and jeff_ok:
        print("\n🚀 Testing Runtime initialization...")
        try:
            from src.jeffrey.runtime import get_runtime

            runtime = get_runtime()
            print("✅ Runtime initialized successfully!")

            # Check components
            if runtime.orchestrator:
                print("✅ Orchestrator ready")
            if runtime.blackboard:
                print("✅ Blackboard ready")
            if runtime.scheduler:
                print("✅ Scheduler ready")

            print("\n🎉 Jeffrey OS is ready to run!")
            return 0
        except Exception as e:
            print(f"❌ Runtime initialization failed: {e}")
            return 1
    else:
        print("\n❌ Cannot test runtime - dependencies missing")
        return 1


if __name__ == "__main__":
    sys.exit(main())
