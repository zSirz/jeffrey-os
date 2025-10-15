"""Interface CLI F2 avec monitoring système et filtrage intelligent."""

import hashlib
import itertools
import os
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import click
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

# Import avec fallback pour robustesse
try:
    from analyzer import NLP_AVAILABLE
except Exception:
    NLP_AVAILABLE = False

# Import psutil pour monitoring
try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠️ psutil non installé - monitoring désactivé (pip install psutil)")

# JSON optimisé (import json toujours nécessaire pour json.dump)
import json as json_module

from orchestrator import AdaptiveOrchestrator
from scanner import THIRD_PARTY_SEGMENTS, PuzzleScanner

try:
    import orjson

    json_loads = orjson.loads
    json_dumps = lambda obj: orjson.dumps(obj).decode('utf-8')
except ImportError:
    json_loads = json_module.loads
    json_dumps = lambda obj: json_module.dumps(obj, ensure_ascii=False)

console = Console()


class SystemMonitor:
    """Monitore CPU/RAM pendant l'analyse."""

    def __init__(self):
        self.monitoring = False
        self.process = psutil.Process() if PSUTIL_AVAILABLE else None
        self.peak_memory = 0
        self.peak_cpu = 0
        self.files_processed = 0
        self.start_time = time.time()
        self.thread = None

    def start(self):
        if not PSUTIL_AVAILABLE:
            return
        self.monitoring = True
        self.start_time = time.time()
        self.thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.monitoring = False
        if self.thread:
            self.thread.join(timeout=1)

    def _monitor_loop(self):
        while self.monitoring:
            try:
                cpu = self.process.cpu_percent(interval=0.1)
                mem = self.process.memory_info().rss / (1024 * 1024 * 1024)  # GB

                self.peak_cpu = max(self.peak_cpu, cpu)
                self.peak_memory = max(self.peak_memory, mem)

                # Auto-warning si dépassement
                if mem > 8.0:  # Si > 8GB RAM
                    console.print(
                        f"\n⚠️ Mémoire élevée: {mem:.1f}GB - Réduction des workers recommandée", style="yellow"
                    )

                time.sleep(5)  # Update toutes les 5 secondes
            except:
                pass

    def get_stats(self):
        if not PSUTIL_AVAILABLE:
            return "Monitoring non disponible"

        try:
            current_mem = self.process.memory_info().rss / (1024 * 1024 * 1024)
            current_cpu = psutil.cpu_percent(interval=0.1)
            elapsed = time.time() - self.start_time
            fps = self.files_processed / max(1, elapsed)

            return (
                f"CPU: {current_cpu:.0f}% (peak: {self.peak_cpu:.0f}%) | "
                f"RAM: {current_mem:.1f}GB (peak: {self.peak_memory:.1f}GB) | "
                f"Speed: {fps:.1f} files/sec"
            )
        except:
            return "Monitoring actif"


@click.group()
def cli():
    """Jeffrey Analyzer F1 - L'analyseur le plus rapide au monde."""
    pass


def _content_fingerprint(path: str, full_cutoff: int = 10 * 1024 * 1024) -> str | None:
    """
    Calcule une empreinte du contenu du fichier.
    - Hash SHA256 complet pour fichiers <= 10MB
    - Hash de 3 échantillons pour gros fichiers
    - None pour fichiers vides ou erreurs
    """
    try:
        stat = os.stat(path)

        # Skip fichiers vides (évite de grouper tous les vides)
        if stat.st_size == 0:
            return None

        if stat.st_size <= full_cutoff:
            # Hash complet pour petits fichiers
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(chunk)
            return f"F:{h.hexdigest()}"
        else:
            # Échantillonnage pour gros fichiers
            with open(path, 'rb') as f:
                # Lire début (64KB)
                head = f.read(65536)
                # Lire milieu (64KB)
                f.seek(max(0, stat.st_size // 2 - 32768))
                mid = f.read(65536)
                # Lire fin (64KB)
                f.seek(max(0, stat.st_size - 65536))
                tail = f.read(65536)

            h = hashlib.sha256(head + mid + tail)
            return f"P:{stat.st_size}:{h.hexdigest()}"
    except Exception:
        return None


def _is_jeffrey_path(p: str) -> bool:
    """Détecte si un chemin contient Jeffrey_ de manière cross-platform."""
    try:
        return any(part.lower().startswith('jeffrey_') for part in Path(p).parts)
    except Exception:
        return 'jeffrey_' in p.lower()


def _pick_canonical(original: str, dup_paths: list[str]) -> str:
    """
    Choisit la version à garder parmi les doublons.
    Priorités : 1) Chemins Jeffrey_*, 2) Plus récent, 3) Original
    """
    all_paths = [original] + dup_paths

    # Prioriser les chemins Jeffrey_*
    jeffrey_paths = [p for p in all_paths if _is_jeffrey_path(p)]
    if jeffrey_paths:
        all_paths = jeffrey_paths

    # Prendre le plus récent
    try:
        return max(all_paths, key=lambda p: os.stat(p).st_mtime)
    except:
        return original


def _full_sha256(path: str) -> str | None:
    """Calcule le hash SHA256 complet d'un fichier."""
    try:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


@cli.command()
@click.option('--jeffrey-only/--full-scan', default=True, help='Scanner uniquement Jeffrey (défaut) ou tout iCloud')
@click.option('--roots', multiple=True, help='Dossiers racines spécifiques')
@click.option('--limit', type=int, help='Limite de fichiers (test)')
@click.option('--resume', is_flag=True, help='Reprendre depuis checkpoint')
@click.option('--no-cache', is_flag=True, help='Désactiver le cache')
@click.option('--workers-io', type=int, help='Workers I/O (auto par défaut)')
@click.option('--workers-cpu', type=int, help='Workers CPU (auto par défaut)')
@click.option('--gpu', is_flag=True, default=True, help='Utiliser GPU si disponible')
@click.option('--visual-report', is_flag=True, help='Générer graphiques')
@click.option('--format', type=click.Choice(['json', 'jsonl', 'yaml']), default='jsonl')
@click.option(
    '--icloud-root',
    type=click.Path(exists=True, path_type=Path),
    default=Path.home() / "Library/Mobile Documents/com~apple~CloudDocs",
    help="Racine iCloud",
)
@click.option(
    '--max-file-size', type=int, default=10 * 1024 * 1024, help='Taille max des fichiers en bytes (défaut: 10MB)'
)
@click.option('--debug', is_flag=True, help='Mode debug avec logs verbeux')
@click.option('--clean', is_flag=True, help='Nettoyer les métadonnées avant analyse')
@click.option('--monitor', is_flag=True, default=False, help='Activer le monitoring système (CPU/RAM)')
@click.option('--include-third-party', is_flag=True, help='Inclure les dépendances tierces (venv, node_modules, etc.)')
@click.option('--verify', is_flag=True, help='Vérifier la pollution après finalisation')
def analyze(
    jeffrey_only,
    roots,
    limit,
    resume,
    no_cache,
    workers_io,
    workers_cpu,
    gpu,
    visual_report,
    format,
    icloud_root,
    max_file_size,
    debug,
    clean,
    monitor,
    include_third_party,
    verify,
):
    """Analyser le code avec performance maximale - Version F2."""

    console.print("[bold cyan]🏎️ JEFFREY ANALYZER F2[/bold cyan]")
    console.print("[yellow]Version F2 - Filtrage intelligent + Monitoring[/yellow]\n")

    # Auto-nettoyage si demandé
    if clean:
        console.print("[yellow]🧹 Nettoyage des anciens artefacts...[/yellow]")
        import subprocess

        try:
            result = subprocess.run(['bash', 'clean_metadata.sh'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                console.print("[green]✅ Nettoyage terminé[/green]")
            else:
                console.print(f"[yellow]⚠️ Nettoyage partiel: {result.stderr[:100]}[/yellow]")
        except:
            console.print("[yellow]⚠️ Script de nettoyage non trouvé[/yellow]")

    # Vérifier que le chemin iCloud existe
    if not icloud_root.exists():
        console.print(f"[red]❌ Chemin iCloud introuvable: {icloud_root}[/red]")
        sys.exit(1)

    # Scanner F2 avec filtrage intelligent
    scanner = PuzzleScanner(max_file_size=max_file_size, include_third_party=include_third_party)

    # Tracking du temps
    start_time = time.time()

    # Monitoring conditionnel
    monitor_obj = None
    if monitor:
        monitor_obj = SystemMonitor()
        monitor_obj.start()
        console.print("[cyan]📊 Monitoring système activé[/cyan]")

    # SECTION CORRIGÉE : Résolution des racines
    if roots:
        # Racines spécifiées manuellement
        resolved_roots = [Path(r) for r in roots]
        console.print(f"[cyan]📁 Racines manuelles: {[r.name for r in resolved_roots]}[/cyan]")
    elif jeffrey_only:
        # Mode Jeffrey-only : scanner uniquement les dossiers Jeffrey
        resolved_roots = scanner.resolve_roots(icloud_root)
        if not resolved_roots:
            console.print("[red]❌ Aucun dossier Jeffrey trouvé dans iCloud![/red]")
            console.print(
                "[yellow]Patterns recherchés: Jeffrey*, Phoenix*, architecture_analysis*, laboratory*, imports[/yellow]"
            )
            console.print(f"[yellow]Dans: {icloud_root}[/yellow]")
            sys.exit(1)
        console.print("[green]✅ Mode Jeffrey-only activé[/green]")
    else:
        # Mode scan complet (non recommandé)
        console.print("[red]⚠️ ATTENTION: Mode scan complet d'iCloud![/red]")
        console.print("[yellow]Des centaines de milliers de fichiers seront scannés...[/yellow]")
        console.print("[yellow]Utilisez --jeffrey-only pour cibler uniquement les dossiers Jeffrey[/yellow]")
        if not click.confirm("Vraiment scanner TOUT iCloud (déconseillé)?"):
            sys.exit(1)
        resolved_roots = [icloud_root]

    # NOUVEAU : Afficher les racines détectées avec estimation
    console.print(f"\n📁 Racines à scanner ({len(resolved_roots)}):")
    console.print(f"[dim]Racine iCloud: {icloud_root}[/dim]")
    for root in resolved_roots:
        try:
            # Estimation rapide limitée pour performance
            file_count = sum(1 for _ in itertools.islice(root.rglob('*'), 100))
            suffix = "+" if file_count == 100 else ""
            console.print(f"   • [cyan]{root.name}[/cyan]: ~{file_count}{suffix} fichiers")
        except Exception as e:
            console.print(f"   • [cyan]{root.name}[/cyan]")
            if debug:
                console.print(f"     [dim]Erreur estimation: {e}[/dim]")

    # Scanner les fichiers (iter_fast_files retourne des strings)
    files = list(scanner.iter_fast_files(resolved_roots, jeffrey_only=jeffrey_only))
    if limit:
        files = files[:limit]

    console.print(f"\n📊 {len(files)} fichiers à analyser")

    # Afficher les stats du scanner
    scanner.print_stats()

    # Update monitor si actif
    if monitor_obj:
        monitor_obj.files_processed = len(files)

    # NOUVEAU : Afficher la répartition des extensions
    if files:
        from pathlib import Path

        exts = Counter([Path(p).suffix.lower() for p in files[: min(1000, len(files))]])
        if exts:
            console.print("\n📈 Top extensions détectées:")
            for ext, count in exts.most_common(10):
                console.print(f"   {ext or '[sans extension]'}: {count} fichiers")

    # Afficher config avec exclusions
    table = Table(title="⚙️ Configuration F2", box=box.ROUNDED)
    table.add_column("Paramètre", style="cyan")
    table.add_column("Valeur", style="green")

    # Afficher les exclusions actives
    if not include_third_party and scanner.ignore_patterns:
        exclusions = list(scanner.ignore_patterns)[:8]
        exclusions_str = ", ".join(exclusions[:5]) + ("..." if len(exclusions) > 5 else "")
    else:
        exclusions_str = "Aucune (mode inclusion totale)"

    table.add_row("Exclusions actives", exclusions_str)
    table.add_row("Fichiers tiers inclus", "Oui ⚠️" if include_third_party else "Non ✅")
    table.add_row("Mode", "Jeffrey-only" if jeffrey_only else "Scan complet")
    table.add_row("Racines", str([r.name for r in resolved_roots]))
    table.add_row("Limite", str(limit) if limit else "Aucune")
    table.add_row("Cache", "Désactivé" if no_cache else "Activé")
    if monitor:
        table.add_row("Monitoring", "✅ Activé")
    # Sinon, rien

    # CORRIGÉ : Status GPU/NLP réel
    gpu_status = "Disponible (NLP)" if (gpu and NLP_AVAILABLE) else "Non disponible"
    table.add_row("GPU/NLP", gpu_status)

    # NOUVEAU : Affichage taille max en MB
    mb = max_file_size / (1024 * 1024)
    table.add_row("Taille max fichier", f"{mb:.1f} MB")

    table.add_row("Format", format)
    table.add_row("Debug", "Activé" if debug else "Désactivé")

    console.print(table)

    # Lancer l'orchestrateur
    orchestrator = AdaptiveOrchestrator()
    orchestrator.debug = debug  # Passer le flag debug

    # Override config si spécifié
    if workers_io:
        orchestrator.config['io_workers'] = workers_io
    if workers_cpu:
        orchestrator.config['cpu_workers'] = workers_cpu

    # Analyser avec progress bar améliorée
    try:
        if monitor:
            # Version avec monitoring temps réel
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.0f}%",
                "•",
                TextColumn("{task.fields[status]}"),
                "•",
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=1,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Analyse F2 en cours...", total=len(files) if files else None, status="Initialisation..."
                )

                # Thread pour update le monitoring
                def update_monitor_status():
                    while orchestrator.running if hasattr(orchestrator, 'running') else True:
                        if monitor_obj:
                            progress.update(task, status=monitor_obj.get_stats())
                        time.sleep(2)

                if monitor_obj:
                    status_thread = threading.Thread(target=update_monitor_status, daemon=True)
                    status_thread.start()

                # Lancer l'analyse
                success = orchestrator.run(resolved_roots, limit)

                progress.update(task, completed=len(files))
        else:
            # Version sans monitoring
            success = orchestrator.run(resolved_roots, limit)

        # Calcul du temps écoulé
        elapsed = time.time() - start_time

        # Message de fin avec Panel Rich
        console.print("\n")
        console.print(
            Panel.fit(
                "✅ ANALYSE F2 TERMINÉE AVEC SUCCÈS",
                border_style="green",
                title="Jeffrey Analyzer F2",
                subtitle=f"Durée: {elapsed:.1f}s",
            )
        )

        # Afficher les stats du scanner
        if hasattr(scanner, 'stats'):
            scanner.print_stats()

        console.print("\n💾 Résultats: Jeffrey_V1/metadata/PUZZLE_PIECES.jsonl")
        console.print("🎯 Prochaine étape: [cyan]python cli.py finalize[/cyan]")

        # Monitoring final si activé
        if monitor_obj:
            monitor_obj.stop()
            console.print(f"\n[dim]Monitoring: {monitor_obj.get_stats()}[/dim]")

        # Générer visualisations si demandé
        if visual_report:
            generate_visualizations(orchestrator)

    except KeyboardInterrupt:
        console.print("\n[yellow]⏸️ Analyse interrompue (checkpoint sauvegardé)[/yellow]")
    except Exception as e:
        console.print(f"\n[red]❌ Erreur: {e}[/red]")
        if debug:
            import traceback

            console.print("[dim]" + traceback.format_exc() + "[/dim]")
        sys.exit(1)


@cli.command()
@click.option(
    '--input', type=click.Path(exists=True, path_type=Path), default=Path("Jeffrey_V1/metadata/PUZZLE_PIECES.jsonl")
)
@click.option('--output', type=click.Path(path_type=Path), default=Path("Jeffrey_V1/metadata/PUZZLE_PLAN.json"))
def finalize(input, output):
    """Finaliser: JSONL → PUZZLE_PLAN.json pour la migration."""
    console.print("[cyan]📦 Finalisation du plan de migration...[/cyan]")

    # Import JSON optimisé
    try:
        import orjson

        json_loads = orjson.loads
        json_dumps = lambda obj: orjson.dumps(obj).decode('utf-8')
    except ImportError:
        import json

        json_loads = json.loads
        json_dumps = lambda obj: json.dumps(obj, ensure_ascii=False, indent=2)

    if not input.exists():
        console.print(f"[red]❌ Fichier introuvable: {input}[/red]")
        sys.exit(1)

    # Charger les pièces
    pieces = []
    errors = 0

    with open(input) as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                pieces.append(json_loads(line))
            except Exception as e:
                errors += 1
                if line_no <= 5:  # Afficher les premières erreurs
                    console.print(f"[yellow]⚠️ Ligne {line_no}: {str(e)[:50]}[/yellow]")

    console.print(f"📊 {len(pieces)} pièces chargées ({errors} erreurs)")

    # Initialiser stats d'abord
    stats = {
        'total_pieces': len(pieces),
        'unique_pieces': 0,
        'duplicates': 0,
        'empty_files': 0,
        'duplicate_groups': 0,
        'by_extension': {},
        'by_analysis': {'with_tests': 0, 'with_docs': 0, 'high_complexity': 0},
    }

    # Initialiser les structures de déduplication
    seen_hashes = {}  # fingerprint -> source original
    duplicate_groups = defaultdict(list)  # fingerprint -> [pieces...]
    unique_pieces = []
    case_collisions = []
    empty_files_count = 0

    # Progress pour fingerprinting
    console.print("\n[cyan]📊 Calcul des empreintes de contenu pour détecter les doublons...[/cyan]")
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        console=console,
        refresh_per_second=1,
    ) as progress:
        task = progress.add_task("[cyan]Analyse des doublons...", total=len(pieces))

        for piece in pieces:
            source = piece.get('source', '')

            # Calculer l'empreinte de contenu
            content_fp = _content_fingerprint(source)

            # Stocker l'empreinte pour tous les pieces
            piece['content_fingerprint'] = content_fp or ""

            if content_fp is None:
                # Fichier vide ou erreur
                empty_files_count += 1
                unique_pieces.append(piece)  # Garder mais ne pas dédupliquer
            elif content_fp in seen_hashes:
                # Doublon trouvé !
                stats['duplicates'] = stats.get('duplicates', 0) + 1
                duplicate_groups[content_fp].append(piece)
            else:
                # Premier exemplaire
                seen_hashes[content_fp] = source
                unique_pieces.append(piece)

            progress.update(task, advance=1)

    stats['empty_files'] = empty_files_count
    stats['unique_pieces'] = len(unique_pieces)
    stats['duplicate_groups'] = len(duplicate_groups)

    # Vérification anti-faux positifs pour gros fichiers échantillonnés
    if duplicate_groups:
        console.print("\n[cyan]🔍 Vérification des doublons de gros fichiers...[/cyan]")
        for fp, dups in list(duplicate_groups.items()):
            if fp.startswith("P:") and dups:  # Fichiers échantillonnés
                base = seen_hashes.get(fp)
                if base:
                    basesig = _full_sha256(base)
                    if basesig:
                        verified = []
                        for d in dups:
                            if _full_sha256(d['source']) == basesig:
                                verified.append(d)
                        duplicate_groups[fp] = verified
                        # Ajuster les stats
                        diff = len(dups) - len(verified)
                        if diff > 0:
                            stats['duplicates'] -= diff

    # Afficher le résumé des doublons
    if duplicate_groups:
        console.print(
            f"\n[bold yellow]🔍 Doublons détectés : {len(duplicate_groups)} groupes, {stats['duplicates']} fichiers[/bold yellow]"
        )

        # Afficher les 5 premiers groupes
        for i, (fp, duplicates) in enumerate(list(duplicate_groups.items())[:5]):
            original = seen_hashes.get(fp, "?")
            console.print(f"\n[dim]Groupe {i + 1}: {len(duplicates) + 1} copies[/dim]")
            console.print(f"  [green]Original:[/green] {Path(original).name}")
            console.print(f"  [dim]Chemin:[/dim] {original}")
            for dup in duplicates[:2]:
                console.print(f"  [yellow]Copie:[/yellow] {dup['source']}")
            if len(duplicates) > 2:
                console.print(f"  [dim]... et {len(duplicates) - 2} autres copies[/dim]")

        if len(duplicate_groups) > 5:
            console.print(f"\n[dim]... et {len(duplicate_groups) - 5} autres groupes de doublons[/dim]")
    else:
        console.print("\n[green]✅ Aucun doublon détecté[/green]")

    # Créer le rapport détaillé de doublons
    if duplicate_groups:
        duplicates_report_path = output.parent / "DUPLICATES_REPORT.json"
        console.print("\n[cyan]📝 Génération du rapport de doublons...[/cyan]")

        groups_data = []
        total_space_savable = 0

        for fp, dups in duplicate_groups.items():
            original = seen_hashes.get(fp)
            dup_paths = [d['source'] for d in dups]

            # Choisir la version canonique (à garder)
            canonical = _pick_canonical(original, dup_paths)
            all_copies = [original] + dup_paths

            # Calculer l'espace récupérable
            try:
                space_savable = sum(os.path.getsize(p) for p in all_copies if p != canonical)
            except:
                space_savable = 0

            total_space_savable += space_savable

            groups_data.append(
                {
                    "fingerprint": fp,
                    "canonical": canonical,
                    "original": original,
                    "copies": all_copies,
                    "count": len(all_copies),
                    "space_savable_bytes": space_savable,
                    "space_savable_mb": round(space_savable / (1024 * 1024), 2),
                }
            )

        # Trier par espace récupérable (plus gros d'abord)
        groups_data.sort(key=lambda x: x['space_savable_bytes'], reverse=True)

        duplicates_data = {
            "summary": {
                "total_groups": len(groups_data),
                "total_duplicate_files": stats['duplicates'],
                "total_space_savable_bytes": total_space_savable,
                "total_space_savable_mb": round(total_space_savable / (1024 * 1024), 2),
                "total_space_savable_gb": round(total_space_savable / (1024 * 1024 * 1024), 2),
                "analysis_timestamp": datetime.now().isoformat(),
            },
            "groups": groups_data,
        }

        with open(duplicates_report_path, 'w', encoding='utf-8') as f:
            json_module.dump(duplicates_data, f, indent=2, ensure_ascii=False)

        console.print(f"[green]✅ Rapport sauvegardé : {duplicates_report_path}[/green]")
        console.print(
            f"[bold green]💾 Espace récupérable : {duplicates_data['summary']['total_space_savable_gb']:.2f} GB[/bold green]"
        )

    # Détection des collisions de casse (macOS)
    seen_lower = {}
    for piece in unique_pieces:
        norm_source = piece['source'].lower()
        if norm_source in seen_lower and seen_lower[norm_source] != piece['source']:
            case_collisions.append((seen_lower[norm_source], piece['source']))
        else:
            seen_lower[norm_source] = piece['source']

    if case_collisions:
        console.print(f"\n[yellow]⚠️ {len(case_collisions)} collisions de casse détectées (macOS)[/yellow]")
        for file1, file2 in case_collisions[:5]:
            console.print(f"   • {file1} vs {file2}")

    # Mettre à jour les stats déjà initialisées

    for piece in unique_pieces:
        # Extension
        ext = Path(piece['source']).suffix
        stats['by_extension'][ext] = stats['by_extension'].get(ext, 0) + 1

        # Analyse
        analysis = piece.get('analysis', {})
        if analysis.get('has_tests'):
            stats['by_analysis']['with_tests'] += 1
        if analysis.get('has_docs'):
            stats['by_analysis']['with_docs'] += 1
        if analysis.get('complexity', 0) > 10:
            stats['by_analysis']['high_complexity'] += 1

    # Fonction helper pour pollution avec constantes centralisées
    def has_third_party_segment(path: str) -> bool:
        """Détecte segments tiers avec constantes centralisées"""
        try:
            parts = set(Path(path).parts)
            return bool(parts & THIRD_PARTY_SEGMENTS)
        except:
            return any(seg in path for seg in THIRD_PARTY_SEGMENTS)

    # Vérifier la pollution (fichiers tiers)
    tiers_count = 0
    jeffrey_count = 0

    for piece in unique_pieces:
        source = piece.get('source', '')
        # Check segments tiers avec fonction robuste
        if has_third_party_segment(source):
            tiers_count += 1
        # Check fichiers Jeffrey
        if 'Jeffrey' in source or 'jeffrey' in source.lower():
            jeffrey_count += 1

    jeffrey_ratio = (jeffrey_count / len(unique_pieces) * 100) if unique_pieces else 0

    # Calculer le ratio de pollution
    pollution_ratio = (tiers_count / len(unique_pieces) * 100) if unique_pieces else 0

    # Stocker les métriques de pollution détaillées dans le plan
    stats['pollution'] = {
        'third_party_files': tiers_count,
        'jeffrey_files': jeffrey_count,
        'total_files': len(unique_pieces),
        'third_party_ratio': round(pollution_ratio, 2),
        'jeffrey_ratio': round(jeffrey_ratio, 2),
        'scan_timestamp': datetime.now().isoformat(),
        'scanner_version': 'F2-Bulletproof-Final',
    }

    # Alerte si pollution détectée
    if tiers_count > 0:
        console.print(f"[yellow]⚠️ ATTENTION: {tiers_count} fichiers tiers détectés dans le plan![/yellow]")
        console.print("[yellow]   Relancez avec --clean pour exclure les dépendances[/yellow]")

    # Créer le plan final avec config
    plan = {
        'version': 'puzzle-2.0',  # Version F2
        'created_at': datetime.now().isoformat(),
        'config': {
            'scanner': 'F2',
            'exclusions': list(scanner.ignore_patterns) if 'scanner' in locals() else [],
            'ignore_file': '.jeffreyignore',
        },
        'stats': stats,
        'pieces': unique_pieces,
        'duplicates': list(duplicate_groups.keys())[:100],  # Limiter pour ne pas exploser la taille
    }

    # Sauvegarder
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json_dumps(plan))

    # Afficher résumé
    table = Table(title="📊 Résumé du Plan")
    table.add_column("Métrique", style="cyan")
    table.add_column("Valeur", style="green")

    table.add_row("Pièces totales", str(stats['total_pieces']))
    table.add_row("Pièces uniques", str(stats['unique_pieces']))
    table.add_row("Doublons détectés", str(stats['duplicates']))
    table.add_row("Avec tests", str(stats['by_analysis']['with_tests']))
    table.add_row("Avec documentation", str(stats['by_analysis']['with_docs']))
    table.add_row("Haute complexité", str(stats['by_analysis']['high_complexity']))

    console.print(table)

    # Afficher les stats de pollution
    if stats.get('pollution'):
        pollution_table = Table(title="🔍 Analyse de Pollution", box=box.ROUNDED)
        pollution_table.add_column("Métrique", style="cyan")
        pollution_table.add_column("Valeur", style="green" if tiers_count == 0 else "yellow")

        pollution_table.add_row("Fichiers tiers", str(stats['pollution']['third_party_files']))
        pollution_table.add_row("Fichiers Jeffrey", str(stats['pollution']['jeffrey_files']))
        pollution_table.add_row("Ratio Jeffrey", f"{stats['pollution']['jeffrey_ratio']:.1f}%")

        console.print(pollution_table)

    console.print(f"\n[green]✅ Plan F2 sauvegardé: {output}[/green]")

    # NOUVEAU : Guide pour la suite
    console.print("\n[bold cyan]🎯 Prochaines étapes:[/bold cyan]")
    console.print(
        "1. Vérifier le plan : [yellow]cat Jeffrey_V1/metadata/PUZZLE_PLAN.json | python -m json.tool | head -50[/yellow]"
    )
    console.print("2. Vérifier les stats : [yellow]jq '.stats' Jeffrey_V1/metadata/PUZZLE_PLAN.json[/yellow]")
    console.print(
        "3. Lancer la migration : [yellow]python ../jeffrey_puzzle.py migrate --plan Jeffrey_V1/metadata/PUZZLE_PLAN.json[/yellow]"
    )
    console.print("4. Si problème, reprendre : [yellow]python cli.py analyze --jeffrey-only --resume[/yellow]")


@cli.command()
def self_analyze():
    """Mode Jeffrey : auto-analyse du code."""
    console.print("[cyan]🤖 Jeffrey entre en mode introspection...[/cyan]")

    # Analyser le propre code de Jeffrey
    jeffrey_path = Path("Jeffrey_V1")
    orchestrator = AdaptiveOrchestrator()

    success = orchestrator.run([jeffrey_path])

    # Calcul du score de santé
    health_score = calculate_health_score(orchestrator)

    console.print(f"\n[bold]Score de santé: {health_score}/100[/bold]")

    if health_score > 80:
        console.print("[green]Jeffrey est en excellente santé![/green]")
    elif health_score > 60:
        console.print("[yellow]Jeffrey a besoin de quelques optimisations[/yellow]")
    else:
        console.print("[red]Jeffrey nécessite une refonte urgente![/red]")


def generate_visualizations(orchestrator):
    """Générer graphiques matplotlib/graphviz."""
    try:
        import json

        import matplotlib.pyplot as plt

        # Histogramme de complexité
        complexities = []
        for line in open(orchestrator.pieces_file):
            piece = json.loads(line)
            if 'complexity' in piece.get('analysis', {}):
                complexities.append(piece['analysis']['complexity'])

        if complexities:
            plt.figure(figsize=(10, 6))
            plt.hist(complexities, bins=30, edgecolor='black')
            plt.xlabel('Complexité Cyclomatique')
            plt.ylabel('Nombre de fichiers')
            plt.title('Distribution de la Complexité du Code')
            plt.savefig(orchestrator.output_dir / 'complexity_histogram.png')
            console.print("[green]✅ Histogramme généré[/green]")

    except ImportError:
        console.print("[yellow]⚠️ matplotlib non installé, pas de graphiques[/yellow]")


def calculate_health_score(orchestrator) -> int:
    """Calculer le score de santé global."""
    score = 100

    # Pénalités
    if orchestrator.stats.get('errors', 0) > 0:
        score -= min(20, orchestrator.stats['errors'])

    # Bonus
    cache_ratio = orchestrator.cache.stats['hits'] / max(1, orchestrator.cache.stats['misses'])
    if cache_ratio > 0.5:
        score = min(100, score + 10)

    return max(0, score)


if __name__ == '__main__':
    cli()
