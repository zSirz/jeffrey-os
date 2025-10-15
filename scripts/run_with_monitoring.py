#!/usr/bin/env python3
"""
Lance Jeffrey avec monitoring temps réel du NeuralBus
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class MonitoredJeffrey:
    def __init__(self):
        self.manager = None
        self.shutdown_event = asyncio.Event()

    async def run(self):
        """Main run function with monitoring"""
        # Import avec uvloop si disponible
        try:
            import uvloop

            uvloop.install()
            print("✅ UVLoop active (+30% perf)")
        except:
            print("ℹ️ UVLoop not available, using standard asyncio")

        from jeffrey.core.loops.loop_manager import LoopManager

        # Créer et démarrer
        self.manager = LoopManager()
        await self.manager.start()

        print("\n" + "=" * 60)
        print("🧠 JEFFREY OS - NEURALBUS ACTIVE")
        print("=" * 60)
        print("Press Ctrl+C to stop\n")

        # Start monitoring task
        monitor_task = asyncio.create_task(self.monitor_loop())

        # Wait for shutdown signal
        await self.shutdown_event.wait()

        # Clean shutdown
        print("\n🛑 Shutting down...")
        monitor_task.cancel()
        await self.manager.stop()
        print("✅ Shutdown complete")

    async def monitor_loop(self):
        """Monitoring loop with real-time dashboard"""
        last_published = 0
        last_consumed = 0
        last_time = time.time()

        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(2)  # Update every 2 seconds

                # Get current metrics
                loop_metrics = self.manager.get_all_metrics() if self.manager else {}
                bus_metrics = (
                    self.manager.event_bus.get_metrics()
                    if (self.manager and hasattr(self.manager.event_bus, "get_metrics"))
                    else {}
                )

                # Calculate rates
                current_time = time.time()
                time_delta = current_time - last_time

                current_published = bus_metrics.get("published", 0)
                current_consumed = bus_metrics.get("consumed", 0)

                publish_rate = (current_published - last_published) / time_delta if time_delta > 0 else 0
                consume_rate = (current_consumed - last_consumed) / time_delta if time_delta > 0 else 0

                last_published = current_published
                last_consumed = current_consumed
                last_time = current_time

                # Clear screen (Unix/Mac) - Comment out if causes issues
                if sys.platform != "win32":
                    print("\033[2J\033[H", end="")

                # Dashboard
                print(f"⏰ {datetime.now().strftime('%H:%M:%S')}")
                print("=" * 60)

                print("📊 SYSTEM STATUS")
                if "system" in loop_metrics:
                    symbiosis = loop_metrics["system"].get("symbiosis_score", 0)
                    status_icon = "✅" if symbiosis > 0.5 else "⚠️"
                    print(f"  Symbiosis Score: {symbiosis:.3f} {status_icon}")
                    print(f"  Active Loops: {len(loop_metrics.get('loops', {}))}")
                    print(f"  Total Cycles: {loop_metrics['system'].get('total_cycles', 0)}")

                print("\n🚌 NEURALBUS METRICS")
                print(f"  Published: {current_published:,} ({publish_rate:.1f}/s)")
                print(f"  Consumed: {current_consumed:,} ({consume_rate:.1f}/s)")

                compressed = bus_metrics.get("compressed_count", 0)
                if compressed > 0:
                    print(f"  Compressed: {compressed:,}")

                p95 = bus_metrics.get("p95_latency_ms", 0)
                p99 = bus_metrics.get("p99_latency_ms", 0)
                print(f"  P95 Latency: {p95:.1f}ms")
                print(f"  P99 Latency: {p99:.1f}ms")

                dropped = bus_metrics.get("dropped", 0) + bus_metrics.get("adapter_dropped", 0)
                drop_rate = dropped / max(1, current_published) * 100
                print(f"  Drop Rate: {drop_rate:.2f}%")

                print("\n🔄 LOOP STATUS")
                if "loops" in loop_metrics:
                    for name, data in loop_metrics["loops"].items():
                        status = "🟢" if data.get("status") == "running" else "🔴"
                        cycles = data.get("cycles", 0)
                        p99_loop = data.get("p99_latency_ms", 0)
                        drops = data.get("bus_dropped", 0)

                        print(f"  {status} {name}: {cycles} cycles, P99={p99_loop:.1f}ms")
                        if drops > 0:
                            print(f"      ⚠️ {drops} events dropped")

                # Performance indicator
                if "system" in loop_metrics:
                    uptime = loop_metrics["system"].get("uptime", 1)
                    throughput = current_published / max(1, uptime)
                    print(f"\n⚡ Average Throughput: {throughput:.0f} msg/sec")

                # Warnings
                if p99 > 40:
                    print("\n⚠️ WARNING: High latency detected")
                if drop_rate > 0.5:
                    print("⚠️ WARNING: Elevated drop rate")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitor error: {e}")
                await asyncio.sleep(5)

    def handle_signal(self, sig, frame):
        """Handle shutdown signals"""
        print("\nReceived interrupt signal...")
        self.shutdown_event.set()


async def main():
    """Main entry point"""
    app = MonitoredJeffrey()

    # Setup signal handlers
    signal.signal(signal.SIGINT, app.handle_signal)
    signal.signal(signal.SIGTERM, app.handle_signal)

    try:
        await app.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nJeffrey OS stopped")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
