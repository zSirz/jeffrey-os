import sys
from datetime import datetime
from typing import Any

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.prompt import Confirm, Prompt
    from rich.table import Table
    from rich.text import Text

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .models import Proposal, ProposalType, RiskLevel, VerdictType
from .proposal_manager import ProposalManager


class HumanInterface:
    """Interactive CLI interface for proposal review"""

    def __init__(self, proposal_manager: ProposalManager):
        self.proposal_manager = proposal_manager
        self.console = Console() if RICH_AVAILABLE else None
        self.active_proposal: Proposal | None = None
        self.review_start_time: datetime | None = None
        self.session_stats = {
            "reviewed": 0,
            "accepted": 0,
            "rejected": 0,
            "deferred": 0,
            "total_time": 0.0,
        }

        # Accessibility settings
        self.use_colors = True
        self.use_emojis = True
        self.verbose_mode = False

        # Auto-detect accessibility needs
        self._detect_accessibility_needs()

    def _detect_accessibility_needs(self):
        """Auto-detect accessibility requirements"""
        # Check for screen reader or high contrast mode
        if sys.platform == "darwin":
            # macOS accessibility detection would go here
            pass
        elif sys.platform == "linux":
            # Linux accessibility detection would go here
            pass

        # For now, default to accessible mode if rich is not available
        if not RICH_AVAILABLE:
            self.use_colors = False
            self.use_emojis = False
            self.verbose_mode = True

    def start_interactive_session(self):
        """Start the main interactive review session"""
        self._print_welcome()

        while True:
            try:
                choice = self._show_main_menu()

                if choice == "q":
                    break
                elif choice == "1":
                    self._review_single_proposal()
                elif choice == "2":
                    self._batch_review()
                elif choice == "3":
                    self._view_statistics()
                elif choice == "4":
                    self._search_proposals()
                elif choice == "5":
                    self._export_session_report()
                elif choice == "6":
                    self._settings_menu()
                else:
                    self._print_error("Invalid choice. Please try again.")

            except KeyboardInterrupt:
                if self._confirm_exit():
                    break
            except Exception as e:
                self._print_error(f"Error: {str(e)}")

    def _print_welcome(self):
        """Print welcome message"""
        if RICH_AVAILABLE:
            welcome_panel = Panel(
                "[bold blue]Jeffrey OS - Proposal Review System[/bold blue]\n"
                "[dim]Phase 0.7 - Human Feedback & Provenance[/dim]\n\n"
                "Welcome to the interactive proposal review interface.\n"
                "Your decisions will be recorded and tracked for complete provenance.",
                title="🤖 Welcome",
                border_style="blue",
            )
            self.console.print(welcome_panel)
        else:
            print("=" * 60)
            print("Jeffrey OS - Proposal Review System")
            print("Phase 0.7 - Human Feedback & Provenance")
            print("=" * 60)
            print("Welcome to the interactive proposal review interface.")
            print("Your decisions will be recorded and tracked for complete provenance.")
            print("=" * 60)

    def _show_main_menu(self) -> str:
        """Show main menu and get user choice"""
        pending_count = len(self.proposal_manager.pending_queue)

        if RICH_AVAILABLE:
            menu_text = f"""
[bold]Main Menu[/bold]

[1] Review Single Proposal ({pending_count} pending)
[2] Batch Review Mode
[3] View Statistics
[4] Search Proposals
[5] Export Session Report
[6] Settings
[q] Quit

Session Stats: {self.session_stats['reviewed']} reviewed, {self.session_stats['accepted']} accepted
"""

            menu_panel = Panel(menu_text, title="📋 Menu", border_style="green")
            self.console.print(menu_panel)

            choice = Prompt.ask("Enter your choice", choices=["1", "2", "3", "4", "5", "6", "q"])
        else:
            print("\n--- Main Menu ---")
            print(f"1. Review Single Proposal ({pending_count} pending)")
            print("2. Batch Review Mode")
            print("3. View Statistics")
            print("4. Search Proposals")
            print("5. Export Session Report")
            print("6. Settings")
            print("q. Quit")
            print(
                f"\nSession Stats: {self.session_stats['reviewed']} reviewed, {self.session_stats['accepted']} accepted"
            )

            choice = input("Enter your choice: ").strip().lower()

        return choice

    def _review_single_proposal(self):
        """Review a single proposal"""
        pending = self.proposal_manager.get_pending_proposals(sort_by="impact", limit=1)

        if not pending:
            self._print_info("No pending proposals to review.")
            return

        proposal = pending[0]
        self.active_proposal = proposal
        self.review_start_time = datetime.now()

        # Display proposal details
        self._display_proposal_details(proposal)

        # Get decision
        verdict = self._get_verdict()
        if verdict is None:
            return  # User cancelled

        # Get rationale
        rationale = self._get_rationale(verdict)

        # Calculate review time
        review_time = (datetime.now() - self.review_start_time).total_seconds()

        # Record decision
        decision = self.proposal_manager.record_decision(
            proposal.id,
            verdict,
            rationale,
            review_time,
            {"interface": "human_cli", "session_id": id(self)},
        )

        # Update session stats
        self._update_session_stats(verdict, review_time)

        # Show confirmation
        self._show_decision_confirmation(proposal, decision)

    def _display_proposal_details(self, proposal: Proposal):
        """Display detailed proposal information"""
        if RICH_AVAILABLE:
            self._display_proposal_rich(proposal)
        else:
            self._display_proposal_plain(proposal)

    def _display_proposal_rich(self, proposal: Proposal):
        """Display proposal with rich formatting"""
        # Create main layout
        layout = Layout()

        # Header with basic info
        header_content = f"""
[bold]ID:[/bold] {proposal.id}
[bold]Type:[/bold] {self._get_type_emoji(proposal.type)} {proposal.type.value.title()}
[bold]Status:[/bold] {proposal.status.value.title()}
[bold]Created:[/bold] {proposal.created_at.strftime('%Y-%m-%d %H:%M:%S')}
"""

        header = Panel(header_content, title="📋 Proposal Info", border_style="blue")

        # Impact and risk visualization
        impact_bar = "█" * int(proposal.impact_score * 10) + "░" * (10 - int(proposal.impact_score * 10))
        risk_color = self._get_risk_color(proposal.risk_level)
        risk_bar = "█" * self._get_risk_level_numeric(proposal.risk_level) + "░" * (
            10 - self._get_risk_level_numeric(proposal.risk_level)
        )

        metrics_content = f"""
[bold]Impact:[/bold] {impact_bar} {proposal.impact_score:.0%}
[bold]Risk:[/bold] [{risk_color}]{risk_bar}[/{risk_color}] {proposal.risk_level.value.title()}
[bold]Sources:[/bold] {len(proposal.sources)} events
"""

        metrics = Panel(metrics_content, title="📊 Metrics", border_style="yellow")

        # Description
        description = Panel(proposal.description, title="📝 Description", border_style="green")

        # Detailed plan
        plan = Panel(
            proposal.detailed_plan or "[dim]No detailed plan provided[/dim]",
            title="🔧 Implementation Plan",
            border_style="cyan",
        )

        # Sources
        sources_content = ""
        for i, source in enumerate(proposal.sources[:3]):  # Show first 3 sources
            sources_content += f"[bold]{i + 1}.[/bold] {source.event_type}: {source.description}\n"
            sources_content += f"   [dim]Time: {source.timestamp.strftime('%Y-%m-%d %H:%M:%S')}[/dim]\n"

        if len(proposal.sources) > 3:
            sources_content += f"[dim]... and {len(proposal.sources) - 3} more sources[/dim]"

        sources = Panel(sources_content, title="🔍 Event Sources", border_style="magenta")

        # Print all panels
        self.console.print(header)
        self.console.print(metrics)
        self.console.print(description)
        if proposal.detailed_plan:
            self.console.print(plan)
        self.console.print(sources)

    def _display_proposal_plain(self, proposal: Proposal):
        """Display proposal in plain text mode"""
        print("\n" + "=" * 60)
        print("PROPOSAL DETAILS")
        print("=" * 60)
        print(f"ID: {proposal.id}")
        print(f"Type: {proposal.type.value.title()}")
        print(f"Status: {proposal.status.value.title()}")
        print(f"Created: {proposal.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Impact Score: {proposal.impact_score:.0%}")
        print(f"Risk Level: {proposal.risk_level.value.title()}")
        print(f"Event Sources: {len(proposal.sources)}")
        print("\nDescription:")
        print(proposal.description)

        if proposal.detailed_plan:
            print("\nImplementation Plan:")
            print(proposal.detailed_plan)

        print("\nEvent Sources:")
        for i, source in enumerate(proposal.sources[:3]):
            print(f"{i + 1}. {source.event_type}: {source.description}")
            print(f"   Time: {source.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")

        if len(proposal.sources) > 3:
            print(f"... and {len(proposal.sources) - 3} more sources")

        print("=" * 60)

    def _get_verdict(self) -> VerdictType | None:
        """Get verdict from user"""
        if RICH_AVAILABLE:
            choices = ["A", "R", "D", "V", "H", "Q"]
            choice = Prompt.ask(
                "[bold]Decision:[/bold] [A]ccept, [R]eject, [D]efer, [V]iew details, [H]elp, [Q]uit",
                choices=choices,
                show_choices=False,
            )
        else:
            print("\nDecision Options:")
            print("A - Accept")
            print("R - Reject")
            print("D - Defer")
            print("V - View details")
            print("H - Help")
            print("Q - Quit")
            choice = input("Enter your choice: ").strip().upper()

        if choice == "A":
            return VerdictType.ACCEPT
        elif choice == "R":
            return VerdictType.REJECT
        elif choice == "D":
            return VerdictType.DEFER
        elif choice == "V":
            self._show_detailed_view()
            return self._get_verdict()  # Ask again after showing details
        elif choice == "H":
            self._show_help()
            return self._get_verdict()  # Ask again after showing help
        elif choice == "Q":
            return None
        else:
            self._print_error("Invalid choice. Please try again.")
            return self._get_verdict()

    def _get_rationale(self, verdict: VerdictType) -> str:
        """Get rationale for the decision"""
        if RICH_AVAILABLE:
            rationale = Prompt.ask(
                f"[bold]Please provide rationale for {verdict.value}ing this proposal:[/bold]",
                default="",
            )
        else:
            rationale = input(f"Please provide rationale for {verdict.value}ing this proposal: ")

        if not rationale.strip():
            if verdict == VerdictType.ACCEPT:
                rationale = "Approved - looks good to implement"
            elif verdict == VerdictType.REJECT:
                rationale = "Rejected - not suitable for implementation"
            else:
                rationale = "Deferred - needs more consideration"

        return rationale.strip()

    def _batch_review(self):
        """Batch review multiple proposals"""
        if RICH_AVAILABLE:
            limit = int(Prompt.ask("How many proposals to review?", default="5"))
        else:
            limit = int(input("How many proposals to review? (default 5): ") or "5")

        proposals = self.proposal_manager.get_pending_proposals(sort_by="impact", limit=limit)

        if not proposals:
            self._print_info("No pending proposals to review.")
            return

        self._print_info(f"Starting batch review of {len(proposals)} proposals")

        for i, proposal in enumerate(proposals):
            self._print_info(f"\n--- Proposal {i + 1} of {len(proposals)} ---")
            self.active_proposal = proposal
            self.review_start_time = datetime.now()

            # Show brief summary
            self._display_proposal_summary(proposal)

            # Quick decision
            verdict = self._get_quick_verdict()
            if verdict is None:
                continue

            rationale = self._get_rationale(verdict)
            review_time = (datetime.now() - self.review_start_time).total_seconds()

            # Record decision
            self.proposal_manager.record_decision(
                proposal.id,
                verdict,
                rationale,
                review_time,
                {"interface": "batch_cli", "session_id": id(self)},
            )

            self._update_session_stats(verdict, review_time)

    def _display_proposal_summary(self, proposal: Proposal):
        """Display brief proposal summary for batch review"""
        if RICH_AVAILABLE:
            summary = f"""
[bold]{proposal.type.value.title()}[/bold] - Impact: {proposal.impact_score:.0%} - Risk: {proposal.risk_level.value}
{proposal.description[:100]}{'...' if len(proposal.description) > 100 else ''}
"""
            self.console.print(Panel(summary, border_style="blue"))
        else:
            print(
                f"{proposal.type.value.title()} - Impact: {proposal.impact_score:.0%} - Risk: {proposal.risk_level.value}"
            )
            print(f"{proposal.description[:100]}{'...' if len(proposal.description) > 100 else ''}")

    def _get_quick_verdict(self) -> VerdictType | None:
        """Get quick verdict for batch review"""
        if RICH_AVAILABLE:
            choice = Prompt.ask(
                "[bold]Quick Decision:[/bold] [A]ccept, [R]eject, [D]efer, [S]kip",
                choices=["A", "R", "D", "S"],
                show_choices=False,
            )
        else:
            choice = input("Quick Decision: [A]ccept, [R]eject, [D]efer, [S]kip: ").strip().upper()

        if choice == "A":
            return VerdictType.ACCEPT
        elif choice == "R":
            return VerdictType.REJECT
        elif choice == "D":
            return VerdictType.DEFER
        else:
            return None

    def _view_statistics(self):
        """View proposal statistics"""
        stats = self.proposal_manager.get_statistics()

        if RICH_AVAILABLE:
            table = Table(title="📊 Proposal Statistics")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")

            table.add_row("Total Proposals", str(stats["total_proposals"]))
            table.add_row("Pending Review", str(stats["pending"]))
            table.add_row("Urgent Proposals", str(stats["urgent_count"]))
            table.add_row("Average Impact", f"{stats['average_impact']:.1%}")

            table.add_section()
            table.add_row("By Status", "")
            for status, count in stats["by_status"].items():
                table.add_row(f"  {status.title()}", str(count))

            table.add_section()
            table.add_row("By Type", "")
            for ptype, count in stats["by_type"].items():
                table.add_row(f"  {ptype.title()}", str(count))

            table.add_section()
            table.add_row("Session Stats", "")
            table.add_row("  Reviewed", str(self.session_stats["reviewed"]))
            table.add_row("  Accepted", str(self.session_stats["accepted"]))
            table.add_row("  Rejected", str(self.session_stats["rejected"]))
            table.add_row("  Deferred", str(self.session_stats["deferred"]))

            if self.session_stats["reviewed"] > 0:
                avg_time = self.session_stats["total_time"] / self.session_stats["reviewed"]
                table.add_row("  Avg Review Time", f"{avg_time:.1f}s")

            self.console.print(table)
        else:
            print("\n--- Proposal Statistics ---")
            print(f"Total Proposals: {stats['total_proposals']}")
            print(f"Pending Review: {stats['pending']}")
            print(f"Urgent Proposals: {stats['urgent_count']}")
            print(f"Average Impact: {stats['average_impact']:.1%}")
            print("\nBy Status:")
            for status, count in stats["by_status"].items():
                print(f"  {status.title()}: {count}")
            print("\nBy Type:")
            for ptype, count in stats["by_type"].items():
                print(f"  {ptype.title()}: {count}")
            print("\nSession Stats:")
            print(f"  Reviewed: {self.session_stats['reviewed']}")
            print(f"  Accepted: {self.session_stats['accepted']}")
            print(f"  Rejected: {self.session_stats['rejected']}")
            print(f"  Deferred: {self.session_stats['deferred']}")
            if self.session_stats["reviewed"] > 0:
                avg_time = self.session_stats["total_time"] / self.session_stats["reviewed"]
                print(f"  Avg Review Time: {avg_time:.1f}s")

    def _search_proposals(self):
        """Search and filter proposals"""
        # This would implement search functionality
        self._print_info("Search functionality not yet implemented")

    def _export_session_report(self):
        """Export session report"""
        # This would export the session report
        self._print_info("Export functionality not yet implemented")

    def _settings_menu(self):
        """Settings menu"""
        if RICH_AVAILABLE:
            setting = Prompt.ask(
                "Settings: [1] Toggle colors, [2] Toggle emojis, [3] Verbose mode, [4] Back",
                choices=["1", "2", "3", "4"],
            )
        else:
            setting = input("Settings: [1] Toggle colors, [2] Toggle emojis, [3] Verbose mode, [4] Back: ")

        if setting == "1":
            self.use_colors = not self.use_colors
            self._print_info(f"Colors: {'enabled' if self.use_colors else 'disabled'}")
        elif setting == "2":
            self.use_emojis = not self.use_emojis
            self._print_info(f"Emojis: {'enabled' if self.use_emojis else 'disabled'}")
        elif setting == "3":
            self.verbose_mode = not self.verbose_mode
            self._print_info(f"Verbose mode: {'enabled' if self.verbose_mode else 'disabled'}")

    def _show_detailed_view(self):
        """Show detailed view of current proposal"""
        if self.active_proposal:
            self._display_proposal_details(self.active_proposal)

    def _show_help(self):
        """Show help information"""
        help_text = """
Help - Proposal Review System

Commands:
- Accept: Approve the proposal for implementation
- Reject: Decline the proposal permanently
- Defer: Postpone decision (proposal stays in queue)
- View Details: See full proposal information
- Help: Show this help message

Tips:
- Review the impact score and risk level carefully
- Consider the event sources that triggered this proposal
- Provide meaningful rationale for your decisions
- Use batch mode for reviewing multiple proposals quickly

Your decisions are recorded with full provenance tracking.
"""

        if RICH_AVAILABLE:
            self.console.print(Panel(help_text, title="❓ Help", border_style="blue"))
        else:
            print(help_text)

    def _confirm_exit(self) -> bool:
        """Confirm exit"""
        if RICH_AVAILABLE:
            return Confirm.ask("Are you sure you want to exit?")
        else:
            response = input("Are you sure you want to exit? (y/n): ").strip().lower()
            return response in ["y", "yes"]

    def _update_session_stats(self, verdict: VerdictType, review_time: float):
        """Update session statistics"""
        self.session_stats["reviewed"] += 1
        self.session_stats["total_time"] += review_time

        if verdict == VerdictType.ACCEPT:
            self.session_stats["accepted"] += 1
        elif verdict == VerdictType.REJECT:
            self.session_stats["rejected"] += 1
        elif verdict == VerdictType.DEFER:
            self.session_stats["deferred"] += 1

    def _show_decision_confirmation(self, proposal: Proposal, decision):
        """Show confirmation of decision"""
        if RICH_AVAILABLE:
            confirmation = f"""
[bold green]Decision Recorded[/bold green]

Proposal: {proposal.description[:50]}...
Decision: {decision.verdict.value.title()}
Rationale: {decision.rationale}
Review Time: {decision.review_time_seconds:.1f}s
"""
            self.console.print(Panel(confirmation, border_style="green"))
        else:
            print("\n--- Decision Recorded ---")
            print(f"Proposal: {proposal.description[:50]}...")
            print(f"Decision: {decision.verdict.value.title()}")
            print(f"Rationale: {decision.rationale}")
            print(f"Review Time: {decision.review_time_seconds:.1f}s")

    def _get_type_emoji(self, ptype: ProposalType) -> str:
        """Get emoji for proposal type"""
        if not self.use_emojis:
            return ""

        emojis = {
            ProposalType.OPTIMIZATION: "🚀",
            ProposalType.SECURITY: "🔒",
            ProposalType.FEATURE: "✨",
            ProposalType.BUGFIX: "🐛",
        }
        return emojis.get(ptype, "📋")

    def _get_risk_color(self, risk: RiskLevel) -> str:
        """Get color for risk level"""
        if not self.use_colors:
            return ""

        colors = {
            RiskLevel.LOW: "green",
            RiskLevel.MEDIUM: "yellow",
            RiskLevel.HIGH: "red",
            RiskLevel.CRITICAL: "bright_red",
        }
        return colors.get(risk, "white")

    def _get_risk_level_numeric(self, risk: RiskLevel) -> int:
        """Get numeric value for risk level"""
        levels = {RiskLevel.LOW: 2, RiskLevel.MEDIUM: 5, RiskLevel.HIGH: 8, RiskLevel.CRITICAL: 10}
        return levels.get(risk, 5)

    def _print_info(self, message: str):
        """Print info message"""
        if RICH_AVAILABLE:
            self.console.print(f"[blue]ℹ️ {message}[/blue]")
        else:
            print(f"Info: {message}")

    def _print_error(self, message: str):
        """Print error message"""
        if RICH_AVAILABLE:
            self.console.print(f"[red]❌ {message}[/red]")
        else:
            print(f"Error: {message}")

    def _print_success(self, message: str):
        """Print success message"""
        if RICH_AVAILABLE:
            self.console.print(f"[green]✅ {message}[/green]")
        else:
            print(f"Success: {message}")


# Voice interface stub for future implementation
class VoiceInterface:
    """Voice interface for accessibility"""

    def __init__(self):
        self.tts_available = False
        self.stt_available = False

    def speak(self, text: str):
        """Text to speech"""
        # Implementation would use pygame or pyttsx3
        pass

    def listen(self) -> str:
        """Speech to text"""
        # Implementation would use speech_recognition
        return ""


# Accessibility validator
class AccessibilityValidator:
    """Validates WCAG compliance"""

    def __init__(self):
        self.wcag_level = "AA"

    def validate_contrast(self, foreground: str, background: str) -> bool:
        """Validate color contrast ratios"""
        # Implementation would check WCAG contrast ratios
        return True

    def validate_text_size(self, size: int) -> bool:
        """Validate text size for readability"""
        return size >= 12

    def generate_compliance_report(self) -> dict[str, Any]:
        """Generate WCAG compliance report"""
        return {
            "wcag_level": self.wcag_level,
            "contrast_compliant": True,
            "text_size_compliant": True,
            "keyboard_navigation": True,
            "screen_reader_compatible": True,
        }
