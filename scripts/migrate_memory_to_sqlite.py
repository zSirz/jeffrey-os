#!/usr/bin/env python3
"""Migrate JSONL memory to SQLite for better performance"""

import json
import sqlite3
import sys
import time
from pathlib import Path


def migrate_to_sqlite(jsonl_path: str, sqlite_path: str):
    """Migrate JSONL memory to SQLite"""
    jsonl_file = Path(jsonl_path)
    if not jsonl_file.exists():
        print(f"Source file not found: {jsonl_path}")
        return False

    # Create SQLite database
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    # Create table with indexes
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            type TEXT,
            timestamp REAL,
            data JSON,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_type ON memories(type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)")

    # Read JSONL and insert
    count = 0
    batch = []

    with jsonl_file.open("r") as f:
        for line in f:
            if not line.strip():
                continue

            try:
                record = json.loads(line)
                batch.append(
                    (
                        record.get("_id", str(time.time())),
                        record.get("type", "unknown"),
                        record.get("_timestamp", time.time()),
                        json.dumps(record),
                    )
                )

                # Insert in batches
                if len(batch) >= 1000:
                    cursor.executemany(
                        "INSERT OR IGNORE INTO memories (id, type, timestamp, data) VALUES (?, ?, ?, ?)",
                        batch,
                    )
                    conn.commit()
                    count += len(batch)
                    print(f"Migrated {count} records...")
                    batch = []

            except json.JSONDecodeError as e:
                print(f"Skipping invalid line: {e}")

    # Insert remaining
    if batch:
        cursor.executemany("INSERT OR IGNORE INTO memories (id, type, timestamp, data) VALUES (?, ?, ?, ?)", batch)
        conn.commit()
        count += len(batch)

    print(f"✅ Migration complete: {count} records migrated")

    # Vacuum to optimize
    cursor.execute("VACUUM")
    conn.close()

    # Backup original
    backup_path = jsonl_file.with_suffix(".jsonl.pre_migration")
    jsonl_file.rename(backup_path)
    print(f"Original file backed up to: {backup_path}")

    return True


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python migrate_memory_to_sqlite.py <jsonl_path> <sqlite_path>")
        sys.exit(1)

    success = migrate_to_sqlite(sys.argv[1], sys.argv[2])
    sys.exit(0 if success else 1)
