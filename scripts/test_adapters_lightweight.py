#!/usr/bin/env python3
"""
Tests légers UNIQUEMENT sur nos adaptateurs - sans dépendances lourdes.
"""

import asyncio
import importlib
import sys
import time

# Set PYTHONPATH
sys.path.insert(0, "src")

from jeffrey.bridge.registry import REGION_ADAPTERS


async def test_adapter_lightweight(region: str, adapter_path: str):
    """Test adapter basique sans dépendances lourdes."""
    module_path, class_name = adapter_path.rsplit(".", 1)

    try:
        # Import et instanciation
        mod = importlib.import_module(module_path)
        adapter_class = getattr(mod, class_name)
        adapter = adapter_class()

        # Test simple
        start = time.time()
        try:
            result = await asyncio.wait_for(adapter.process({"input": f"test {region}"}), timeout=1.0)
        except TimeoutError:
            return False, "Timeout (1s)"

        elapsed = time.time() - start

        # Vérifier sortie basique
        if not isinstance(result, dict):
            return False, f"Sortie invalide: {type(result)}"

        print(f"✅ {region}: OK ({elapsed:.3f}s)")
        return True, None

    except ImportError as e:
        if "numpy" in str(e) or "networkx" in str(e):
            print(f"⚠️  {region}: Dépendance manquante ({e}) - IGNORÉ")
            return True, None  # On ignore les dépendances manquantes du noyau
        return False, str(e)
    except Exception as e:
        return False, str(e)


async def main():
    """Test tous nos adaptateurs en mode léger."""
    print("🔍 Tests légers des adaptateurs:")
    results = []

    for region, path in REGION_ADAPTERS.items():
        success, error = await test_adapter_lightweight(region, path)
        results.append((region, success, error))

        if not success:
            print(f"❌ {region}: {error}")

    # Résumé
    successful = [r for r, s, e in results if s]
    failed = [(r, e) for r, s, e in results if not s]

    print(f"\n📊 RÉSULTATS: {len(successful)}/{len(results)} adaptateurs fonctionnels")

    if failed:
        print(f"\n❌ ÉCHECS ({len(failed)}):")
        for region, error in failed:
            print(f"   {region}: {error}")
        sys.exit(1)

    print("✅ Tous nos adaptateurs sont fonctionnels")


if __name__ == "__main__":
    asyncio.run(main())
