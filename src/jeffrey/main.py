#!/usr/bin/env python3
"""
Jeffrey OS Main - Avec UVLoop pour +30% performance
"""

import logging
import platform
import sys

# Configuration du logging (éviter doublons)
if logging.getLogger().handlers:
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("jeffrey_neuralbus.log")],
)
logger = logging.getLogger(__name__)

# UVLOOP pour performance maximale (sauf Windows)
if platform.system() != "Windows":
    try:
        import uvloop

        uvloop.install()
        logger.info("✅ UVLoop installed - +30% async performance")
    except ImportError:
        logger.warning("⚠️ UVLoop not available - using standard asyncio")

import asyncio

from jeffrey.core.bus.neurobus_adapter import NeuroBusAdapter
from jeffrey.core.loops.loop_manager import LoopManager
from jeffrey.core.neuralbus.bus_v2 import NeuralBusV2 as NeuralBus


async def main():
    """Main entry point for Jeffrey OS with NeuralBus"""

    # Configuration NeuralBus
    neural_bus = NeuralBus()

    # Initialiser le bus
    logger.info("Initializing NeuralBus...")
    await neural_bus.start()

    # Créer adapter
    bus_adapter = NeuroBusAdapter(neural_bus)
    await bus_adapter.connect()

    # Créer LoopManager avec bus réel
    manager = LoopManager(event_bus=bus_adapter)

    # Démarrer
    logger.info("Starting Jeffrey OS with NeuralBus...")
    await manager.start()

    # Afficher métriques initiales
    metrics = neural_bus.get_metrics() if hasattr(neural_bus, "get_metrics") else {}
    logger.info("🧠 Jeffrey OS running with NeuralBus")
    logger.info(f"📊 Initial Performance: {metrics}")

    # Run forever
    try:
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("\n⏹️ Shutting down...")
        await manager.stop()
        await neural_bus.shutdown()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Jeffrey OS stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
