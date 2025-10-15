#!/usr/bin/env python3
"""
Rotate and purge old monitoring logs
Retention policy: 14 days
"""

import gzip
import logging
import shutil
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


def rotate_logs(logs_dir: Path = Path("logs/predictions"), retention_days: int = 14):
    """
    Compress logs older than 1 day, delete logs older than retention_days.

    Args:
        logs_dir: Directory containing prediction logs
        retention_days: Number of days to retain logs (default: 14)
    """
    logger.info(f"🔄 Starting log rotation (retention: {retention_days} days)")

    now = datetime.now()
    one_day_ago = now - timedelta(days=1)
    retention_threshold = now - timedelta(days=retention_days)

    compressed_count = 0
    deleted_count = 0

    for log_file in logs_dir.glob("predictions_*.jsonl"):
        # Skip already compressed files
        if log_file.suffix == ".gz":
            continue

        # Parse date from filename (predictions_YYYY-MM-DD.jsonl)
        try:
            date_str = log_file.stem.replace("predictions_", "")
            file_date = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            logger.warning(f"⚠️ Skipping file with invalid date format: {log_file}")
            continue

        # Delete if beyond retention
        if file_date < retention_threshold:
            log_file.unlink()
            deleted_count += 1
            logger.info(f"🗑️ Deleted old log: {log_file.name}")

            # Also delete corresponding .gz if exists
            gz_file = log_file.with_suffix(".jsonl.gz")
            if gz_file.exists():
                gz_file.unlink()
                logger.info(f"🗑️ Deleted old compressed log: {gz_file.name}")

        # Compress if older than 1 day but within retention
        elif file_date < one_day_ago:
            gz_file = log_file.with_suffix(".jsonl.gz")

            if not gz_file.exists():
                with open(log_file, 'rb') as f_in:
                    with gzip.open(gz_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)

                log_file.unlink()  # Delete uncompressed after successful compression
                compressed_count += 1
                logger.info(f"📦 Compressed: {log_file.name} → {gz_file.name}")

    logger.info("✅ Log rotation complete:")
    logger.info(f"   Compressed: {compressed_count} files")
    logger.info(f"   Deleted: {deleted_count} files")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rotate_logs()
