#!/usr/bin/env python3
"""Migration pour initialiser la DB mémoire de Jeffrey"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path


def migrate():
    """Exécuter les migrations SQL"""
    db_path = Path("data/jeffrey_memory.db")
    migrations_dir = Path("data/migrations")

    # Créer le dossier data si nécessaire
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Connexion à la DB
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    # Créer table de tracking des migrations
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS migrations (
            id INTEGER PRIMARY KEY,
            filename TEXT UNIQUE,
            applied_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """
    )

    # Appliquer chaque migration
    applied = 0
    for sql_file in sorted(migrations_dir.glob("*.sql")):
        # Vérifier si déjà appliquée
        cur.execute("SELECT 1 FROM migrations WHERE filename = ?", (sql_file.name,))
        if cur.fetchone():
            print(f"  ⏭️  Already applied: {sql_file.name}")
            continue

        # Lire et exécuter la migration
        print(f"  📝 Applying: {sql_file.name}")
        sql_content = sql_file.read_text(encoding="utf-8")
        cur.executescript(sql_content)

        # Marquer comme appliquée
        cur.execute("INSERT INTO migrations (filename) VALUES (?)", (sql_file.name,))
        applied += 1

    # Ajouter une première mémoire système
    if applied > 0:
        cur.execute(
            """
            INSERT OR IGNORE INTO memories (id, content, context, importance, source)
            VALUES (?, ?, ?, ?, ?)
        """,
            (
                "mem_system_init",
                "Jeffrey OS Bundle 2 initialized - Memory system online",
                json.dumps({"event": "bundle2_init", "version": "7.0.0"}),
                1.0,
                "system",
            ),
        )

    con.commit()

    # Stats
    cur.execute("SELECT COUNT(*) FROM memories")
    memory_count = cur.fetchone()[0]

    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cur.fetchall()]

    con.close()

    print(f"\n✅ Database migrated: {db_path}")
    print(f"   Tables: {', '.join(tables)}")
    print(f"   Memories: {memory_count}")
    print(f"   Applied: {applied} new migrations")
    print(f"   Timestamp: {datetime.now().isoformat()}")

    return db_path


if __name__ == "__main__":
    migrate()
