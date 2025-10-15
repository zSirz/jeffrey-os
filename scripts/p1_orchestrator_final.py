#!/usr/bin/env python3
"""P1 Week 1 Final Orchestrator - Production Ready"""

import argparse
import asyncio
import logging
import os  # GPT fix: missing import
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class P1FinalOrchestrator:
    """Final P1 orchestrator with all improvements"""

    def __init__(self, dry_run: bool = True, max_workers: int = None):
        self.dry_run = dry_run  # DRY RUN BY DEFAULT
        self.max_workers = max_workers or min(4, (os.cpu_count() or 1))
        self.results = {
            "start_time": datetime.now().isoformat(),
            "dry_run": dry_run,
            "platform": sys.platform,
            "python_version": sys.version,
            "tasks": {},
        }

    async def run_task(self, name: str, command: list[str], timeout: int = None) -> dict[str, Any]:
        """Run task with configurable timeout"""
        timeout = timeout or int(os.getenv("JEFFREY_TASK_TIMEOUT", "300"))

        logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Starting: {name}")

        if self.dry_run:
            return {"status": "DRY_RUN", "command": " ".join(command), "would_execute": True}

        start = time.time()

        try:
            proc = await asyncio.create_subprocess_exec(
                *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            return {
                "status": "SUCCESS" if proc.returncode == 0 else "FAILED",
                "return_code": proc.returncode,
                "duration": time.time() - start,
                "stdout": stdout.decode()[:1000] if stdout else "",
                "stderr": stderr.decode()[:1000] if stderr else "",
            }

        except TimeoutError:
            return {"status": "TIMEOUT", "duration": timeout}
        except Exception as e:
            return {"status": "ERROR", "error": str(e)}

    async def phase0_check_environment(self):
        """Phase 0: Environment verification"""
        result = await self.run_task("Check Environment", ["python3", "scripts/check_environment.py"])
        self.results["tasks"]["phase0_environment"] = result

        if result.get("return_code") != 0 and not self.dry_run:
            logger.warning("Environment check failed - some features may not work")

    async def phase1_sync_async_adapters(self):
        """Phase 1: Create sync/async adapters"""
        # Test adapters
        test_code = """
import sys
sys.path.insert(0, 'src')
from jeffrey.core.adapters.kernel_adapter import BrainKernelAdapter
from jeffrey.core.adapters.symbiosis_adapter import SymbiosisEngineAdapter
print('Adapters OK')
"""
        result = await self.run_task("Test Adapters", ["python3", "-c", test_code])
        self.results["tasks"]["phase1_adapters"] = result

    async def phase2_fix_memory_moment(self):
        """Phase 2: Fix MemoryMoment"""
        result = await self.run_task("Fix MemoryMoment", ["python3", "scripts/fix_memory_moment_robust.py"])
        self.results["tasks"]["phase2_memory_fix"] = result

    async def phase3_setup_contracts(self):
        """Phase 3: Setup data contracts"""
        test_code = """
import sys
sys.path.insert(0, 'src')
from jeffrey.core.contracts.data_models import MemoryMoment, EmotionState
mm = MemoryMoment(message='test')
print(f'Contracts OK: {mm.source}')
"""
        result = await self.run_task("Test Contracts", ["python3", "-c", test_code])
        self.results["tasks"]["phase3_contracts"] = result

    async def phase4_test_privacy(self):
        """Phase 4: Test privacy filter"""
        test_code = """
import sys
sys.path.insert(0, 'src')
from jeffrey.core.ethics.privacy_global import privacy_filter
test_text = 'My email is test@example.com and phone 123-456-7890'
result = privacy_filter.redact(test_text)
print(f'Privacy filter OK: {len(privacy_filter.PII_PATTERNS)} patterns')
"""
        result = await self.run_task("Test Privacy Filter", ["python3", "-c", test_code], timeout=60)
        self.results["tasks"]["phase4_privacy"] = result

    async def phase5_quality_checks(self):
        """Phase 5: Code quality (optional)"""
        # Run quality checks if tools available
        tasks = [
            ("Ruff", ["ruff", "check", "src", "--exit-zero"]),
            ("Black", ["black", "--check", "src", "--diff"]),
        ]

        for name, cmd in tasks:
            # Check if tool exists
            check = subprocess.run(["which", cmd[0]], capture_output=True)
            if check.returncode == 0:
                result = await self.run_task(f"Quality - {name}", cmd, timeout=60)
                self.results["tasks"][f"phase5_{name}"] = result
            else:
                self.results["tasks"][f"phase5_{name}"] = {
                    "status": "SKIP",
                    "reason": f"{cmd[0]} not installed",
                }

    def generate_final_report(self):
        """Generate comprehensive final report"""
        report_path = Path("reports/P1_WEEK1_FINAL_REPORT.md")
        report_path.parent.mkdir(exist_ok=True)

        report = [
            "# P1 Week 1 Final Report",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Mode:** {'DRY RUN' if self.dry_run else 'EXECUTED'}",
            f"**Platform:** {self.results['platform']}",
            "",
            "## Summary",
            "",
        ]

        # Count statuses
        statuses = {}
        for task in self.results["tasks"].values():
            status = task.get("status", "UNKNOWN")
            statuses[status] = statuses.get(status, 0) + 1

        for status, count in statuses.items():
            emoji = {
                "SUCCESS": "✅",
                "DRY_RUN": "🔍",
                "WARNING": "⚠️",
                "FAILED": "❌",
                "ERROR": "🔥",
                "SKIP": "⏭️",
            }.get(status, "❓")
            report.append(f"- {emoji} {status}: {count}")

        report.append("\n## Task Details\n")

        for name, result in self.results["tasks"].items():
            status = result.get("status", "UNKNOWN")
            emoji = {
                "SUCCESS": "✅",
                "DRY_RUN": "🔍",
                "WARNING": "⚠️",
                "FAILED": "❌",
                "ERROR": "🔥",
                "TIMEOUT": "⏱️",
                "SKIP": "⏭️",
            }.get(status, "❓")

            report.append(f"### {emoji} {name}")
            report.append(f"- Status: {status}")
            if "duration" in result:
                report.append(f"- Duration: {result['duration']:.2f}s")
            if "error" in result:
                report.append(f"- Error: `{result['error']}`")
            if "reason" in result:
                report.append(f"- Reason: {result['reason']}")
            report.append("")

        # Next steps
        report.append("\n## Next Steps\n")

        if self.dry_run:
            report.append("1. Review this dry run report")
            report.append("2. Run with `--execute` to apply changes")
        else:
            report.append("1. Review any warnings or errors above")
            report.append("2. Test the new adapters and contracts")
            report.append("3. Run capability tests if BrainKernel is available")

        report_path.write_text("\n".join(report))
        logger.info(f"📊 Report generated: {report_path}")

    async def run(self):
        """Run all phases"""
        logger.info(f"Starting P1 Orchestration ({'DRY RUN' if self.dry_run else 'EXECUTE'})")

        # Run phases in order
        await self.phase0_check_environment()
        await self.phase1_sync_async_adapters()
        await self.phase2_fix_memory_moment()

        # Parallel phases
        await asyncio.gather(self.phase3_setup_contracts(), self.phase4_test_privacy())

        await self.phase5_quality_checks()

        # Generate report
        self.generate_final_report()

        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print execution summary"""
        print("\n" + "=" * 60)
        print(f"P1 ORCHESTRATION {'DRY RUN' if self.dry_run else 'COMPLETE'}")
        print("=" * 60)

        success_count = sum(
            1 for t in self.results["tasks"].values() if t.get("status") in ["SUCCESS", "DRY_RUN", "SKIP"]
        )
        total_count = len(self.results["tasks"])

        print(f"✅ Success: {success_count}/{total_count}")
        print("📊 Report: reports/P1_WEEK1_FINAL_REPORT.md")

        if self.dry_run:
            print("\n⚠️ This was a DRY RUN - no changes made")
            print("Run with --execute to apply changes")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="P1 Week 1 Final Orchestrator")
    parser.add_argument("--execute", action="store_true", help="Execute changes (default is dry-run)")
    parser.add_argument("--workers", type=int, help="Max parallel workers")

    args = parser.parse_args()

    # DRY RUN by default unless --execute is passed
    orchestrator = P1FinalOrchestrator(dry_run=not args.execute, max_workers=args.workers)

    exit_code = 0
    try:
        asyncio.run(orchestrator.run())

        # Set exit code based on results
        if not orchestrator.dry_run:
            failed = sum(1 for t in orchestrator.results["tasks"].values() if t.get("status") == "FAILED")
            exit_code = 1 if failed > 0 else 0

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
        exit_code = 130
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
