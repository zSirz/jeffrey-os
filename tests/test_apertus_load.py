"""
Test de charge pour Apertus intégré
"""

import asyncio
import statistics
import time

from jeffrey.core.llm.apertus_client import ApertusClient


async def load_test(n_messages: int = 100, rate: int = 10):
    """
    Test de charge Apertus
    """
    client = ApertusClient()

    latencies = []
    errors = 0

    print(f"🚀 Starting load test: {n_messages} messages at {rate} msg/s")

    start_time = time.perf_counter()

    for i in range(n_messages):
        try:
            msg_start = time.perf_counter()

            response, metadata = await client.chat("Tu es Jeffrey", f"Message test #{i}: Réponds brièvement.")

            latency = metadata["latency_ms"]
            latencies.append(latency)

            # Contrôle du débit
            elapsed = time.perf_counter() - msg_start
            sleep_time = max(0, (1.0 / rate) - elapsed)
            await asyncio.sleep(sleep_time)

        except Exception as e:
            errors += 1
            print(f"❌ Error on message {i}: {e}")

    total_time = time.perf_counter() - start_time

    # Statistiques
    if latencies:
        print(
            f"""
    ======================================
    📊 LOAD TEST RESULTS
    ======================================
    Total messages: {n_messages}
    Total time: {total_time:.2f}s
    Actual rate: {n_messages / total_time:.2f} msg/s
    Errors: {errors} ({errors / n_messages * 100:.2f}%)

    Latency Stats (ms):
      Min: {min(latencies):.2f}
      Max: {max(latencies):.2f}
      Mean: {statistics.mean(latencies):.2f}
      P50: {statistics.median(latencies):.2f}"""
        )

        if len(latencies) > 1:
            print(
                f"""      P95: {statistics.quantiles(latencies, n=20)[18]:.2f}
      P99: {statistics.quantiles(latencies, n=100)[98]:.2f}"""
            )

        print()
    else:
        print(
            f"""
    ======================================
    📊 LOAD TEST RESULTS
    ======================================
    Total messages: {n_messages}
    Total time: {total_time:.2f}s
    Errors: {errors} (100%)
    No successful messages to compute latency stats
    """
        )

    # Check santé
    health = await client.health_check()
    print(f"    Health check: {health}")


if __name__ == "__main__":
    asyncio.run(load_test(n_messages=100, rate=10))
