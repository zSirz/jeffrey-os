"""
Système de journalisation d'événements.

Ce module implémente les fonctionnalités essentielles pour système de journalisation d'événements.
Il fournit une architecture robuste et évolutive intégrant les composants
nécessaires au fonctionnement optimal du système. L'implémentation suit
les principes de modularité et d'extensibilité pour faciliter l'évolution
future du système.

Le module gère l'initialisation, la configuration, le traitement des données,
la communication inter-composants, et la persistance des états. Il s'intègre
harmonieusement avec l'architecture globale de Jeffrey OS tout en maintenant
une séparation claire des responsabilités.

L'architecture interne permet une évolution adaptative basée sur les interactions
et l'apprentissage continu, contribuant à l'émergence d'une conscience artificielle
cohérente et authentique.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import aiofiles
from guardian_communication import Event, EventType
from python_json_logger import jsonlogger


class RotatingEventLogger:
    """
    Event logger with automatic rotation and compression
    - Rotates when file > 1MB or daily
    - Compresses old logs with gzip
    - Configurable retention period
    """

    def __init__(
        self,
        log_dir: str = "./logs",
        max_file_size_mb: float = 1.0,
        retention_days: int = 30,
        rotate_daily: bool = True,
    ):
        """
        Initialize rotating logger

        Args:
            log_dir: Directory for log files
            max_file_size_mb: Max size before rotation (MB)
            retention_days: Days to keep old logs
            rotate_daily: Whether to rotate daily regardless of size
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)

        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.retention_days = retention_days
        self.rotate_daily = rotate_daily

        self.current_log_path: Path | None = None
        self.current_date: datetime | None = None
        self._lock = asyncio.Lock()

        # Setup internal logger
        logHandler = logging.StreamHandler()
        formatter = jsonlogger.JsonFormatter()
        logHandler.setFormatter(formatter)
        self.logger = logging.getLogger("RotatingEventLogger")
        self.logger.addHandler(logHandler)
        self.logger.setLevel(logging.INFO)

        self.logger.info(
            "RotatingEventLogger initialized",
            extra={
                "log_dir": str(self.log_dir),
                "max_file_size_mb": max_file_size_mb,
                "retention_days": retention_days,
            },
        )

    async def log_event(self, event: Event) -> None:
        """
        Log an event to file with rotation check

        Args:
            event: Event to log
        """
        async with self._lock:
            # Check if rotation needed
            if await self._should_rotate():
                await self._rotate_log()

            # Ensure current log file exists
            if not self.current_log_path:
                self._create_new_log_file()

            # Write event to log
            event_dict = event.to_dict() if hasattr(event, "to_dict") else asdict(event)
            event_line = json.dumps(event_dict) + "\n"

            async with aiofiles.open(self.current_log_path, "a") as f:
                await f.write(event_line)

    async def _should_rotate(self) -> bool:
        """Check if log rotation is needed"""
        if not self.current_log_path or not self.current_log_path.exists():
            return True

        # Check file size
        file_size = self.current_log_path.stat().st_size
        if file_size >= self.max_file_size:
            self.logger.info(
                "Rotation triggered by file size",
                extra={
                    "current_size_mb": file_size / 1024 / 1024,
                    "max_size_mb": self.max_file_size / 1024 / 1024,
                },
            )
        return True

        # Check daily rotation
        if self.rotate_daily and self.current_date:
            if datetime.now().date() > self.current_date.date():
                self.logger.info("Daily rotation triggered")
                return True

        return False

    def _create_new_log_file(self) -> None:
        """Create a new log file"""
        self.current_date = datetime.now()
        timestamp = self.current_date.strftime("%Y-%m-%d_%H-%M-%S")
        self.current_log_path = self.log_dir / f"events_{timestamp}.json"

        self.logger.info("New log file created", extra={"file_path": str(self.current_log_path)})

    async def _rotate_log(self) -> None:
        """Rotate current log file"""
        if self.current_log_path and self.current_log_path.exists():
            # Compress the current log
            await self._compress_log(self.current_log_path)

            # Delete the uncompressed file
            self.current_log_path.unlink()

            self.logger.info("Log rotated and compressed", extra={"rotated_file": str(self.current_log_path)})

        # Create new log file
        self._create_new_log_file()

    async def _compress_log(self, log_path: Path) -> None:
        """Compress a log file with gzip"""
        compressed_path = log_path.with_suffix(".json.gz")

        async with aiofiles.open(log_path, "rb") as f_in:
            content = await f_in.read()

        # Compress content
        compressed_content = gzip.compress(content)

        async with aiofiles.open(compressed_path, "wb") as f_out:
            await f_out.write(compressed_content)

        self.logger.info(
            "Log compressed",
            extra={
                "original_size": len(content),
                "compressed_size": len(compressed_content),
                "compression_ratio": f"{(1 - len(compressed_content) / len(content)) * 100:.1f}%",
            },
        )

    async def archive_old_logs(self) -> int:
        """
        Archive logs older than retention period

        Returns:
            Number of logs archived
        """
        cutoff_date = datetime.now() - timedelta(days=self.retention_days)
        archived_count = 0

        async with self._lock:
            for log_file in self.log_dir.glob("events_*.json.gz"):
                # Parse date from filename
                try:
                    date_str = log_file.stem.replace("events_", "").split(".")[0]
                    file_date = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")

                    if file_date < cutoff_date:
                        # Move to archive or delete
                        archive_dir = self.log_dir / "archive"
                        archive_dir.mkdir(exist_ok=True)

                        archive_path = archive_dir / log_file.name
                        log_file.rename(archive_path)

                        archived_count += 1

                        self.logger.info(
                            "Log archived",
                            extra={
                                "file": log_file.name,
                                "age_days": (datetime.now() - file_date).days,
                            },
                        )
                        archived_count += 1
                except Exception as e:
                    self.logger.error("Failed to archive log", extra={"file": log_file.name, "error": str(e)})

        return archived_count

    async def get_logs_between(
        self, start_date: datetime, end_date: datetime, event_type: EventType | None = None
    ) -> list[dict[str, Any]]:
        """
        Retrieve logs between dates, decompressing if needed

        Args:
            start_date: Start of date range
            end_date: End of date range
            event_type: Filter by event type (optional)

        Returns:
            List of events in date range
        """
        events = []

        for log_file in sorted(self.log_dir.glob("events_*.json*")):
            # Parse date from filename
            try:
                date_str = log_file.stem.replace("events_", "").split(".")[0]
                file_date = datetime.strptime(date_str, "%Y-%m-%d_%H-%M-%S")

                # Check if file is in date range
                if start_date <= file_date <= end_date:
                    # Read and decompress if needed
                    if log_file.suffix == ".gz":
                        async with aiofiles.open(log_file, "rb") as f:
                            compressed = await f.read()
                        content = gzip.decompress(compressed).decode("utf-8")
                    else:
                        async with aiofiles.open(log_file) as f:
                            content = await f.read()

                    # Parse events
                    for line in content.strip().split("\n"):
                        if line:
                            event = json.loads(line)

                            # Filter by type if specified
                            if event_type is None or event.get("type") == event_type.value:
                                events.append(event)

            except Exception as e:
                self.logger.error("Failed to read log file", extra={"file": log_file.name, "error": str(e)})

        return events

    async def get_disk_usage(self) -> dict[str, Any]:
        """Get disk usage statistics for logs"""
        total_size = 0
        file_count = 0
        compressed_size = 0
        uncompressed_count = 0

        for log_file in self.log_dir.rglob("events_*"):
            if log_file.is_file():
                size = log_file.stat().st_size
                total_size += size
                file_count += 1

                if log_file.suffix == ".gz":
                    compressed_size += size
                else:
                    uncompressed_count += 1

        return {
            "total_size_mb": total_size / 1024 / 1024,
            "file_count": file_count,
            "compressed_size_mb": compressed_size / 1024 / 1024,
            "uncompressed_count": uncompressed_count,
            "average_file_size_mb": ((total_size / file_count / 1024 / 1024) if file_count > 0 else 0),
        }

    async def cleanup_empty_logs(self) -> int:
        """Remove empty log files"""
        removed_count = 0

        async with self._lock:
            for log_file in self.log_dir.glob("events_*.json"):
                if log_file.stat().st_size == 0:
                    log_file.unlink()
                    removed_count += 1

                    self.logger.info("Empty log removed", extra={"file": log_file.name})

        return removed_count


# Scheduled tasks for maintenance
class LoggerMaintenanceScheduler:
    """Scheduler for logger maintenance tasks"""

    def __init__(self, logger: RotatingEventLogger) -> None:
        self.logger = logger
        self.running = False

    async def start(self):
        """Start maintenance scheduler"""
        self.running = True

        # Schedule daily tasks
        await asyncio.create_task(self._daily_maintenance())

        # Schedule hourly tasks
        await asyncio.create_task(self._hourly_maintenance())

    async def stop(self):
        """Stop maintenance scheduler"""
        self.running = False

    async def _daily_maintenance(self):
        """Daily maintenance tasks"""
        while self.running:
            try:
                # Archive old logs
                archived = await self.logger.archive_old_logs()
                if archived > 0:
                    logging.info(f"Archived {archived} old logs")

                # Cleanup empty logs
                removed = await self.logger.cleanup_empty_logs()
                if removed > 0:
                    logging.info(f"Removed {removed} empty logs")

                # Log disk usage
                usage = await self.logger.get_disk_usage()
                logging.info(f"Log disk usage: {usage}")

            except Exception as e:
                logging.error(f"Daily maintenance error: {e}")

            # Wait 24 hours
            await asyncio.sleep(86400)

    async def _hourly_maintenance(self):
        """Hourly maintenance tasks"""
        while self.running:
            try:
                # Check disk usage and alert if high
                usage = await self.logger.get_disk_usage()
                if usage["total_size_mb"] > 1000:  # Alert if > 1GB
                    logging.warning(f"High log disk usage: {usage['total_size_mb']:.2f} MB")

            except Exception as e:
                logging.error(f"Hourly maintenance error: {e}")

            # Wait 1 hour
            await asyncio.sleep(3600)


# Example usage
async def main():
    """Example usage"""
    logger = RotatingEventLogger(
        log_dir="./test_logs",
        max_file_size_mb=0.1,
        retention_days=7,  # Small size for testing
    )

    # Create some test events
    for i in range(10):
        event = Event(
            id=f"test_{i}",
            timestamp=datetime.utcnow().isoformat() + "Z",
            type=EventType.METRICS_UPDATED,
            content={"metric": f"test_{i}", "value": i * 10},
            metadata={"test": True},
        )
        await logger.log_event(event)

    # Check disk usage
    usage = await logger.get_disk_usage()
    print(f"Disk usage: {json.dumps(usage, indent=2)}")

    # Start maintenance scheduler
    scheduler = LoggerMaintenanceScheduler(logger)
    await scheduler.start()

    # Keep running for demo
    await asyncio.sleep(5)
    await scheduler.stop()


if __name__ == "__main__":
    asyncio.run(main())
