import json
import os

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()


class EvolutionDashboard:
    def __init__(self, personality_state_path: str):
        self.personality_state_path = personality_state_path

    def load_state(self):
        if not os.path.exists(self.personality_state_path):
            return None
        with open(self.personality_state_path) as f:
            data = json.load(f)
        return data

    def show_dashboard(self):
        state_data = self.load_state()
        if not state_data:
            console.print("[bold red]Aucun état de personnalité trouvé.[/bold red]")
            return

        # Titre
        console.print(Panel(Text("ÉVOLUTION DE JEFFREY", style="bold white on blue"), expand=False))

        # Date et âge de Jeffrey
        created_at = state_data.get("created_at", "inconnu")
        age = state_data.get("age", "inconnu")
        console.print(f"[bold cyan]Date de création :[/bold cyan] {created_at}")
        console.print(f"[bold cyan]Âge émotionnel actuel :[/bold cyan] {age}")

        # Traits de personnalité
        table = Table(title="Traits de personnalité", show_header=True, header_style="bold magenta")
        table.add_column("Trait", justify="left")
        table.add_column("Niveau", justify="center")

        traits = state_data.get("traits", {})
        for trait, level in traits.items():
            table.add_row(trait.capitalize(), str(level))
        console.print(table)

        # Empreinte émotionnelle initiale
        imprint = state_data.get("empreinte_emotionnelle", {})
        if imprint:
            console.print("\n[bold yellow]Empreinte émotionnelle initiale :[/bold yellow]")
            console.print(imprint.get("description", "Non spécifiée"))

        # Niveau de dépendance
        dep = state_data.get("niveau_dependance", {})
        console.print(f"\n[bold green]Dépendance émotionnelle actuelle :[/bold green] {dep.get('niveau', 'inconnu')}")

        # Historique des événements si présent
        if "historique" in state_data:
            hist = state_data["historique"][-5:]  # les 5 derniers événements
            console.print("\n[bold]Derniers souvenirs marquants :[/bold]")
            for entry in hist:
                date = entry.get("date", "inconnue")
                event = entry.get("evenement", "non spécifié")
                console.print(f"• [italic]{date}[/italic] → {event}")


if __name__ == "__main__":
    dashboard = EvolutionDashboard("core/personality/personality_state.json")
    dashboard.show_dashboard()
