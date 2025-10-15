#!/usr/bin/env python3
"""
Test amélioré pour le système de discovery Jeffrey OS
"""

import asyncio
import json
import logging
from pathlib import Path

import yaml

from jeffrey_brain import JeffreyBrain

# Configuration du logging avec couleurs si possible
try:
    from rich.logging import RichHandler

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(rich_tracebacks=True)])
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def print_banner(text):
    """Afficher un banner"""
    width = 60
    print("\n" + "=" * width)
    print(text.center(width))
    print("=" * width)


def analyze_manifest(manifest_path):
    """Analyser et afficher les informations du manifest"""

    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return None

    with open(manifest_path) as f:
        if manifest_path.suffix == '.yaml':
            manifest = yaml.safe_load(f)
        else:
            manifest = json.load(f)

    print("\n📊 MANIFEST ANALYSIS:")
    print("-" * 40)

    # Résumé
    summary = manifest.get('summary', {})
    print(f"Total modules: {summary.get('total_discovered', 0)}")
    print(f"├─ Plugins (with JEFFREY_PLUGIN): {summary.get('plugin_count', 0)}")
    print(f"└─ Legacy (without): {summary.get('legacy_count', 0)}")

    # Par catégorie
    if 'scan_stats' in summary and 'by_category' in summary['scan_stats']:
        print("\nBy category:")
        for cat, count in sorted(summary['scan_stats']['by_category'].items()):
            print(f"  - {cat}: {count}")

    # Métriques du graphe
    if 'graph_metrics' in manifest:
        metrics = manifest['graph_metrics']
        print("\nGraph metrics:")
        print(f"  - Nodes: {metrics.get('node_count', 0)}")
        print(f"  - Edges: {metrics.get('edge_count', 0)}")
        print(f"  - Avg fitness: {metrics.get('avg_fitness', 0):.2f}")
        print(f"  - Total connections: {metrics.get('total_connections', 0)}")
        print(f"  - Avg strength: {metrics.get('avg_strength', 0.5):.2f}")

    # Exemples de modules
    modules = manifest.get('modules', {})
    if modules:
        print("\nSample modules (first 5 of each type):")

        # Plugins
        plugins = [key for key, info in modules.items() if info.get('type') == 'plugin']
        if plugins:
            print(f"\n  Plugins ({len(plugins)} total):")
            for key in plugins[:5]:
                info = modules[key]
                name = info.get('display_name', key)
                topics = info.get('contract', {}).get('topics_in', [])
                print(f"    • {name}: {', '.join(topics) if topics else 'no topics'}")
            if len(plugins) > 5:
                print(f"    ... and {len(plugins) - 5} more")

        # Legacy
        legacy = [key for key, info in modules.items() if info.get('type') == 'legacy']
        if legacy:
            print(f"\n  Legacy ({len(legacy)} total):")
            for key in legacy[:10]:
                info = modules[key]
                name = info.get('display_name', key)
                cat = info.get('category', 'unknown')
                print(f"    • {name} ({cat})")
            if len(legacy) > 10:
                print(f"    ... and {len(legacy) - 10} more")

    return manifest


async def test_discovery():
    """Test principal du système de discovery"""

    print_banner("JEFFREY OS - AUTO-DISCOVERY TEST V4")

    try:
        # Initialiser le cerveau
        print("\n🧠 Initializing Jeffrey Brain...")
        brain = JeffreyBrain()
        await brain.initialize()
        print("✅ Brain initialized successfully")

        # Phase 1: Dry-run
        print_banner("PHASE 1: DRY-RUN DISCOVERY")

        connected, failed = await brain.setup_discovery(dry_run=True)

        # Analyser le manifest
        manifest_path = Path("discovered_brain.yaml")
        if not manifest_path.exists():
            manifest_path = Path("discovered_brain.json")

        manifest = analyze_manifest(manifest_path)

        if not manifest:
            print("❌ Discovery failed - no manifest created")
            return

        # Statistiques de scan
        scan_stats = manifest.get('summary', {}).get('scan_stats', {})
        if scan_stats:
            print("\n📈 Scan Statistics:")
            print(f"  - Total found: {scan_stats.get('total', 0)}")
            print(f"  - Plugins: {scan_stats.get('plugin', 0)}")
            print(f"  - Legacy: {scan_stats.get('legacy', 0)}")
            print(f"  - Skipped: {scan_stats.get('skipped', 0)}")

        # Proposer la connexion si des plugins existent
        plugin_count = manifest.get('summary', {}).get('plugin_count', 0)

        if plugin_count > 0:
            print_banner("PHASE 2: PLUGIN CONNECTION")
            print(f"\n{plugin_count} plugin module(s) available for connection.")
            print("Legacy modules are already connected by boot and won't be reconnected.")

            response = input("\nConnect plugin modules now? (y/N): ").strip().lower()

            if response == 'y':
                print("\n🔌 Connecting plugin modules...")
                connected, failed = await brain.setup_discovery(dry_run=False)

                print("\n✅ Results:")
                print(f"  - Connected: {connected}")
                print(f"  - Failed: {len(failed)}")

                if failed and len(failed) <= 5:
                    print(f"  - Failed modules: {', '.join(failed)}")
                elif failed:
                    print(f"  - Failed modules (first 5): {', '.join(failed[:5])}...")
                    print(f"    ... and {len(failed) - 5} more")

                # Test de message après connexion
                if connected > 0:
                    print("\n📧 Testing message after connection...")
                    await brain.process_input("Discovery test message", "discovery_test")
                    await asyncio.sleep(0.5)
                    print("✅ Test message sent")
            else:
                print("⏭️  Skipping connection phase")
        else:
            print("\nℹ️  No plugin modules found. All modules are legacy (already connected by boot).")
            print("💡 Tip: To add plugin support, add JEFFREY_PLUGIN contract to your modules.")
            print("\nExample JEFFREY_PLUGIN:")
            print('"""')
            print("JEFFREY_PLUGIN = {")
            print("    'topics_in': ['percept.text', 'emotion.state'],")
            print("    'topics_out': ['plan.execute'],")
            print("    'handler': 'process',  # Method name to call")
            print("    'dependencies': []")
            print("}")
            print('"""')

        # Test de santé
        print_banner("HEALTH CHECK")

        # Vérifier que le système fonctionne toujours
        print("\n🏥 Testing brain functionality...")

        # Test simple
        test_envelope = type('Envelope', (), {'topic': 'system.health', 'payload': {'test': True}})()

        try:
            # Essayer d'émettre un événement de test
            if hasattr(brain.bus, 'emit'):
                if asyncio.iscoroutinefunction(brain.bus.emit):
                    await brain.bus.emit('system.health', test_envelope)
                else:
                    brain.bus.emit('system.health', test_envelope)
                print("✅ Neural bus is operational")
            elif hasattr(brain.bus, 'publish'):
                if asyncio.iscoroutinefunction(brain.bus.publish):
                    await brain.bus.publish('system.health', test_envelope)
                else:
                    brain.bus.publish('system.health', test_envelope)
                print("✅ Neural bus is operational (via publish)")
            else:
                print("⚠️  Neural bus emit/publish not available")

            # Vérifier le registry
            if hasattr(brain, 'registry') and hasattr(brain.registry, 'services'):
                service_count = len(brain.registry.services)
                print(f"✅ Registry has {service_count} services")
            else:
                print("⚠️  Registry not available")

        except Exception as e:
            print(f"⚠️  Health check warning: {e}")

        # Afficher les métriques finales si disponibles
        if brain.discovery and hasattr(brain.discovery, 'graph'):
            print("\n📊 Final Graph Metrics:")
            metrics = brain.discovery.graph.get_metrics()
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  - {key}: {value:.2f}")
                else:
                    print(f"  - {key}: {value}")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()

    print_banner("TEST COMPLETE")


if __name__ == "__main__":
    asyncio.run(test_discovery())
