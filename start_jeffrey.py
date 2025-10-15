#!/usr/bin/env python3
"""
Lanceur principal pour Jeffrey OS
"""

import asyncio
import logging
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

# Ajouter le répertoire src au path
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.INFO)
console = Console()


async def main():
    """Point d'entrée principal de Jeffrey OS"""

    # Banner de bienvenue
    welcome_text = Text()
    welcome_text.append("🤖 ", style="bold cyan")
    welcome_text.append("JEFFREY OS ", style="bold magenta")
    welcome_text.append("v1.0", style="green")

    console.print(Panel.fit(welcome_text, border_style="cyan", padding=(1, 2)))

    # Essayer d'importer l'orchestrateur
    try:
        from jeffrey.core.orchestration.ia_orchestrator_ultimate import UltimateOrchestrator

        orch = UltimateOrchestrator()
        console.print("[green]✅ Système initialisé avec succès[/green]")

        # Afficher le statut
        try:
            stats = await orch.get_orchestration_stats()
            console.print("[cyan]Tous les composants sont opérationnels[/cyan]")
            console.print(f"[dim]Professeurs actifs: {len(stats.get('professors', {}))}[/dim]")
        except Exception as e:
            console.print(f"[yellow]⚠️ Mode dégradé - {e}[/yellow]")

    except ImportError as e:
        console.print(f"[red]❌ Erreur import: {e}[/red]")
        console.print("[yellow]Mode démonstration activé[/yellow]")
        orch = None

    console.print("\n[dim]Tapez 'help' pour l'aide, 'quit' pour quitter[/dim]\n")

    # Boucle interactive
    while True:
        try:
            # Prompt avec style
            user_input = Prompt.ask("[bold cyan]Vous[/bold cyan]")

            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[yellow]Au revoir! 👋[/yellow]")
                break

            elif user_input.lower() == "help":
                console.print(
                    Panel(
                        "[cyan]Commandes disponibles:[/cyan]\n"
                        "• help - Affiche cette aide\n"
                        "• status - Affiche l'état du système\n"
                        "• clear - Efface l'écran\n"
                        "• quit/exit - Quitte l'application\n"
                        "\n[dim]Tapez n'importe quoi d'autre pour interagir avec Jeffrey[/dim]",
                        title="Aide",
                        border_style="blue",
                    )
                )

            elif user_input.lower() == "status":
                if orch:
                    try:
                        stats = await orch.get_orchestration_stats()
                        budget = stats.get("budget_status", {})
                        console.print(
                            Panel.fit(
                                f"[green]Système: Opérationnel[/green]\n"
                                f"Professeurs: {len(stats.get('professors', {}))}\n"
                                f"Budget utilisé: {budget.get('utilization_percentage', 0):.1f}%\n"
                                f"Charge système: {stats.get('total_load', 0):.1f}%",
                                title="État du Système",
                                border_style="green",
                            )
                        )
                    except Exception as e:
                        console.print(f"[yellow]Erreur stats: {e}[/yellow]")
                else:
                    console.print("[yellow]Système en mode démonstration[/yellow]")

            elif user_input.lower() == "clear":
                console.clear()

            else:
                # Traitement par l'orchestrateur
                if orch:
                    try:
                        with console.status("[cyan]Jeffrey réfléchit...[/cyan]"):
                            response = await orch.orchestrate_request(user_input, user_id="user", session_id="console")
                        if isinstance(response, dict):
                            console.print(f"[green]{response.get('response', 'Réponse reçue')}[/green]")
                        else:
                            console.print(f"[green]{response}[/green]")
                    except Exception as e:
                        console.print(f"[yellow]Erreur: {e}[/yellow]")
                else:
                    # Mode demo
                    console.print(f"[green]Jeffrey> En mode démo, j'ai reçu: '{user_input}'[/green]")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interruption détectée[/yellow]")
            break

        except Exception as e:
            console.print(f"[red]Erreur: {e}[/red]")

    # Shutdown propre
    if orch:
        try:
            await orch.shutdown_orchestrator()
        except:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]Arrêt...[/yellow]")
        sys.exit(0)
