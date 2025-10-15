#!/usr/bin/env python3
"""
Benchmark script for NeuralBus performance testing
"""

import asyncio
import statistics
import time

from jeffrey.core.neuralbus import CloudEvent, EventMeta, EventPriority, neural_bus

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


async def benchmark_throughput(num_events: int = 1000):
    """Benchmark publishing throughput"""
    await neural_bus.start()

    print(f"Benchmarking with {num_events} events...")

    # Warmup
    for _ in range(10):
        event = CloudEvent(meta=EventMeta(type="bench.warmup", tenant_id="bench"), data={"test": "warmup"})
        await neural_bus.publish(event)

    # Collect metrics
    latencies: list[float] = []

    if HAS_PSUTIL:
        cpu_before = psutil.cpu_percent(interval=0.1)
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

    start_time = time.perf_counter()

    # Benchmark with different priorities
    priorities = [
        EventPriority.LOW,
        EventPriority.NORMAL,
        EventPriority.HIGH,
        EventPriority.CRITICAL,
    ]

    for i in range(num_events):
        event_start = time.perf_counter()

        event = CloudEvent(
            meta=EventMeta(type="bench.test", tenant_id="bench", priority=priorities[i % 4]),
            data={"index": i, "timestamp": time.time()},
        )

        await neural_bus.publish(event)

        latency = (time.perf_counter() - event_start) * 1000  # ms
        latencies.append(latency)

    duration = time.perf_counter() - start_time

    # Calculate statistics
    throughput = num_events / duration

    if latencies:
        p50 = statistics.quantiles(latencies, n=100)[49]
        p95 = statistics.quantiles(latencies, n=100)[94]
        p99 = statistics.quantiles(latencies, n=100)[98]
        avg_latency = statistics.mean(latencies)
    else:
        p50 = p95 = p99 = avg_latency = 0

    # Resource usage
    if HAS_PSUTIL:
        cpu_after = psutil.cpu_percent(interval=0.1)
        mem_after = process.memory_info().rss / 1024 / 1024
        cpu_usage = cpu_after - cpu_before
        mem_delta = mem_after - mem_before
    else:
        cpu_usage = mem_delta = 0

    # Print results
    print(
        f"""
╔══════════════════════════════════════════════════════╗
║            NEURALBUS BENCHMARK RESULTS               ║
╠══════════════════════════════════════════════════════╣
║ Total Events:   {num_events:>10}                      ║
║ Total Time:     {duration:>10.2f} seconds           ║
║ Throughput:     {throughput:>10.0f} events/sec      ║
╠══════════════════════════════════════════════════════╣
║ LATENCY STATISTICS (milliseconds)                    ║
╠══════════════════════════════════════════════════════╣
║ Average:        {avg_latency:>10.2f} ms            ║
║ P50:            {p50:>10.2f} ms                    ║
║ P95:            {p95:>10.2f} ms                    ║
║ P99:            {p99:>10.2f} ms                    ║
╠══════════════════════════════════════════════════════╣
║ RESOURCE USAGE                                       ║
╠══════════════════════════════════════════════════════╣
║ CPU Delta:      {cpu_usage:>10.1f} %               ║
║ Memory Delta:   {mem_delta:>10.1f} MB              ║
╚══════════════════════════════════════════════════════╝
"""
    )

    # Performance verdict
    if throughput >= 5000:
        print("🚀 EXCELLENT: Production-grade performance achieved!")
    elif throughput >= 1000:
        print("✅ GOOD: Ready for production use")
    elif throughput >= 500:
        print("⚠️  ACCEPTABLE: Consider performance optimizations")
    else:
        print("❌ SLOW: Performance tuning required")

    # Check configuration
    from jeffrey.core.neuralbus.config import config

    print("\nConfiguration:")
    print(f"  • msgpack: {'✅' if config.USE_MSGPACK else '❌'}")
    print(f"  • uvloop: {'✅' if config.USE_UVLOOP else '❌'}")
    print(f"  • Redis dedup: {'✅' if config.USE_REDIS_DEDUP else '❌'}")
    print(f"  • Dynamic batching: {'✅' if config.BATCH_DYNAMIC else '❌'}")

    await neural_bus.shutdown()


async def test_consumer_performance():
    """Test consumer processing performance"""
    await neural_bus.start()

    processed_count = 0
    processing_times = []

    # Create consumer
    consumer = neural_bus.create_consumer("BENCH_CONSUMER", "events.bench.>")

    async def benchmark_handler(event, headers):
        nonlocal processed_count
        start = time.perf_counter()

        # Simulate some processing
        await asyncio.sleep(0.001)

        processing_times.append((time.perf_counter() - start) * 1000)
        processed_count += 1

    consumer.register_handler("bench.consumer", benchmark_handler)
    await consumer.connect()

    # Start consumer
    consumer_task = asyncio.create_task(consumer.run())
    await asyncio.sleep(1)

    # Publish test events
    print("Publishing 100 events for consumer test...")
    start = time.perf_counter()

    for i in range(100):
        event = CloudEvent(meta=EventMeta(type="bench.consumer", tenant_id="bench"), data={"index": i})
        await neural_bus.publish(event)

    # Wait for processing
    await asyncio.sleep(5)

    duration = time.perf_counter() - start

    print("\nConsumer Performance:")
    print(f"  • Processed: {processed_count} events")
    print(f"  • Processing rate: {processed_count / duration:.0f} events/sec")

    if processing_times:
        print(f"  • Avg processing time: {statistics.mean(processing_times):.2f} ms")
        print(f"  • P95 processing time: {statistics.quantiles(processing_times, n=100)[94]:.2f} ms")

    # Cleanup
    await consumer.stop()
    consumer_task.cancel()
    await neural_bus.shutdown()


async def main():
    """Run all benchmarks"""
    print("\n" + "=" * 60)
    print(" NEURALBUS PERFORMANCE BENCHMARK")
    print("=" * 60)

    print("\n1️⃣  PUBLISHER THROUGHPUT TEST")
    print("-" * 40)
    await benchmark_throughput(1000)

    print("\n2️⃣  CONSUMER PERFORMANCE TEST")
    print("-" * 40)
    await test_consumer_performance()

    print("\n" + "=" * 60)
    print(" BENCHMARK COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
