#!/usr/bin/env python
"""
Monitore Apertus en temps réel
"""

import asyncio

from rich.console import Console
from rich.live import Live
from rich.table import Table

from jeffrey.core.llm.apertus_client import ApertusClient
from jeffrey.core.llm.hybrid_bridge import HybridOrchestrator


async def monitor():
    console = Console()
    client = ApertusClient()
    orchestrator = HybridOrchestrator(client)

    with Live(console=console, refresh_per_second=1) as live:
        while True:
            try:
                # Collecter les métriques
                health = await client.health_check()
                stats = orchestrator.get_stats()

                # Créer le tableau
                table = Table(title="🧠 Apertus Monitor")
                table.add_column("Metric", style="cyan")
                table.add_column("Value", style="green")

                # Santé
                table.add_row("Status", "✅ Healthy" if health["healthy"] else "❌ Unhealthy")
                table.add_row("Model", health.get("model", "N/A"))

                # Performance
                table.add_row("Avg Latency", f"{health['metrics']['avg_latency_ms']:.2f} ms")
                table.add_row("Total Requests", str(health["metrics"]["total_requests"]))
                success_rate = 0
                if health["metrics"]["total_requests"] > 0:
                    success_rate = health["metrics"]["successful_responses"] / health["metrics"]["total_requests"] * 100
                table.add_row("Success Rate", f"{success_rate:.1f}%")

                # Routing
                table.add_row("─────", "─────")
                percentages = stats.get("percentages", {})
                table.add_row("Local Routes", f"{stats['local']} ({percentages.get('local', 0):.1f}%)")
                table.add_row(
                    "External Routes",
                    f"{stats['external']} ({percentages.get('external', 0):.1f}%)",
                )
                table.add_row("Hybrid Routes", f"{stats['hybrid']} ({percentages.get('hybrid', 0):.1f}%)")

                live.update(table)
            except Exception as e:
                # En cas d'erreur, afficher un message
                error_table = Table(title="🧠 Apertus Monitor")
                error_table.add_column("Status", style="red")
                error_table.add_row(f"Error: {str(e)}")
                live.update(error_table)

            await asyncio.sleep(1)


if __name__ == "__main__":
    try:
        asyncio.run(monitor())
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped")
