#!/usr/bin/env python3
"""
Tests avec effets I/O RÉELS et timeouts.
"""

import asyncio
import importlib
import sys
import time
from pathlib import Path

# Set PYTHONPATH
sys.path.insert(0, "src")

from jeffrey.bridge.registry import REGION_ADAPTERS


async def test_adapter_with_io(region: str, adapter_path: str, timeout: float = 2.0):
    """Test adapter avec timeout et vérif I/O si applicable."""
    module_path, class_name = adapter_path.rsplit(".", 1)

    try:
        # Import et instanciation
        mod = importlib.import_module(module_path)
        adapter_class = getattr(mod, class_name)
        adapter = adapter_class()

        # Test avec timeout
        start = time.time()
        try:
            result = await asyncio.wait_for(
                adapter.process({"input": f"test {region}", "meta": {"test": True}}),
                timeout=timeout,
            )
        except TimeoutError:
            return False, f"Timeout ({timeout}s dépassé)"

        elapsed = time.time() - start

        # Vérifier sortie
        if not isinstance(result, dict):
            return False, f"Sortie invalide: {type(result)}"

        # Vérif I/O pour memory
        if region == "memory":
            # Chercher fichiers DB créés
            db_files = list(Path(".").glob("**/*.db"))
            data_dirs = [d for d in Path(".").iterdir() if d.is_dir() and "data" in d.name.lower()]

            if db_files or data_dirs:
                print(f"   ✓ I/O détecté: {len(db_files)} DB, {len(data_dirs)} dossiers data")
            else:
                return False, "Aucun effet I/O détecté (DB/dossiers data manquants)"

        print(f"✅ {region}: OK ({elapsed:.2f}s)")
        return True, None

    except Exception as e:
        return False, str(e)


async def main():
    """Test tous les adaptateurs."""
    print("🔍 Tests I/O réels avec timeout sur les 8 adaptateurs:")
    results = []

    for region, path in REGION_ADAPTERS.items():
        print(f"\n[{region.upper()}]")
        success, error = await test_adapter_with_io(region, path)
        results.append((region, success, error))

        if not success:
            print(f"❌ {region}: {error}")

    # Résumé
    successful = [r for r, s, e in results if s]
    failed = [(r, e) for r, s, e in results if not s]

    print(f"\n{'=' * 50}")
    print(f"📊 RÉSULTATS: {len(successful)}/{len(results)} adaptateurs passés")

    if failed:
        print(f"\n❌ ÉCHECS ({len(failed)}):")
        for region, error in failed:
            print(f"   {region}: {error}")
        sys.exit(1)

    print("\n✅ Tous les adaptateurs passent les tests I/O réels")


if __name__ == "__main__":
    asyncio.run(main())
