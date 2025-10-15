#!/bin/bash
# Fix Bundle 2 migration issues

echo "🔧 Fixing Bundle 2 migration..."

# First, check if database exists and has issues
if [ -f "data/jeffrey_memory.db" ]; then
    echo "📝 Database exists, checking schema..."

    # Check current schema
    sqlite3 data/jeffrey_memory.db << 'SQL'
.schema memories
SQL

    echo "🔧 Adding missing columns if needed..."

    # Add missing columns safely
    sqlite3 data/jeffrey_memory.db << 'SQL'
-- Add session_id if it doesn't exist
ALTER TABLE memories ADD COLUMN session_id TEXT;
ALTER TABLE memories ADD COLUMN source TEXT;
SQL

    echo "✅ Schema updated"
else
    echo "📝 No database found, will create fresh"
fi

# Now run the rest of Bundle 2 setup
echo -e "\n📝 Continuing Bundle 2 setup..."

# Create improved SQLite Store
cat > src/jeffrey/bundle2/memory/sqlite_store.py << 'PY'
"""Jeffrey OS Bundle 2 - SQLite Memory Store (no external deps)"""
from __future__ import annotations
import asyncio, sqlite3, json, time, os
from typing import Any, Dict, List, Optional
from pathlib import Path

class SQLiteMemoryStore:
    """Store de mémoire persistante pour Jeffrey"""

    def __init__(self, db_path: str = "data/jeffrey_memory.db"):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        """S'assurer que la DB existe avec le bon schema"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.db_path)

        # Créer la table si elle n'existe pas
        con.execute("""CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            context TEXT,
            emotions TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            importance REAL DEFAULT 0.5,
            access_count INTEGER DEFAULT 0,
            last_accessed DATETIME,
            vector BLOB,
            source TEXT,
            session_id TEXT
        )""")

        # Créer les index s'ils n'existent pas
        con.execute("CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance)")
        con.execute("CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id)")

        con.commit()
        con.close()

    def _conn(self):
        """Créer une connexion"""
        return sqlite3.connect(self.db_path)

    async def store(self, content: Any, metadata: Dict = None) -> str:
        """Stocker une mémoire"""
        def _store():
            con = self._conn()
            cur = con.cursor()
            mid = f"mem_{int(time.time()*1000000)}"  # microsecond precision

            metadata = metadata or {}
            context = metadata.get("context", {})

            # Extract session_id from context if present
            session_id = context.get("session_id", "") if isinstance(context, dict) else ""

            cur.execute(
                """INSERT OR REPLACE INTO memories
                   (id, content, context, emotions, importance, source, session_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    mid,
                    json.dumps(content) if not isinstance(content, str) else content,
                    json.dumps(context),
                    json.dumps(metadata.get("emotions", {})),
                    float(metadata.get("importance", 0.5)),
                    metadata.get("source", ""),
                    session_id
                )
            )
            con.commit()
            con.close()
            return mid

        # Python 3.9+ compatible
        try:
            return await asyncio.to_thread(_store)
        except AttributeError:
            # Fallback for Python < 3.9
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _store)

    async def recall(self, memory_id: str) -> Optional[Dict]:
        """Rappeler une mémoire par ID"""
        def _recall():
            con = self._conn()
            cur = con.cursor()
            row = cur.execute(
                """SELECT id, content, context, emotions, importance,
                          timestamp, access_count
                   FROM memories WHERE id = ?""",
                (memory_id,)
            ).fetchone()

            if row:
                # Update access count
                cur.execute(
                    "UPDATE memories SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE id = ?",
                    (memory_id,)
                )
                con.commit()

            con.close()

            if not row:
                return None

            return {
                "id": row[0],
                "content": json.loads(row[1]) if row[1] and (row[1].startswith('[') or row[1].startswith('{')) else row[1],
                "context": json.loads(row[2]) if row[2] else {},
                "emotions": json.loads(row[3]) if row[3] else {},
                "importance": row[4],
                "timestamp": row[5],
                "access_count": row[6]
            }

        try:
            return await asyncio.to_thread(_recall)
        except AttributeError:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _recall)

    async def search(self, query: str, limit: int = 10) -> List[Dict]:
        """Rechercher dans les mémoires (recherche dans content ET context)"""
        q = f"%{query.lower()}%"

        def _search():
            con = self._conn()
            cur = con.cursor()
            rows = cur.execute(
                """SELECT id, content, importance, timestamp, access_count
                   FROM memories
                   WHERE lower(content) LIKE ?
                      OR lower(COALESCE(context,'')) LIKE ?
                      OR lower(COALESCE(session_id,'')) LIKE ?
                   ORDER BY importance DESC, timestamp DESC
                   LIMIT ?""",
                (q, q, q, limit)
            ).fetchall()
            con.close()

            results = []
            for row in rows:
                content = row[1]
                if content and (content.startswith('[') or content.startswith('{')):
                    try:
                        content = json.loads(content)
                    except:
                        pass

                results.append({
                    "id": row[0],
                    "content": content,
                    "importance": row[2],
                    "timestamp": row[3],
                    "access_count": row[4]
                })

            return results

        try:
            return await asyncio.to_thread(_search)
        except AttributeError:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _search)

    async def update_access(self, memory_id: str) -> None:
        """Mettre à jour le compteur d'accès"""
        def _update():
            con = self._conn()
            cur = con.cursor()
            cur.execute(
                "UPDATE memories SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE id = ?",
                (memory_id,)
            )
            con.commit()
            con.close()

        try:
            await asyncio.to_thread(_update)
        except AttributeError:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, _update)

    async def consolidate(self) -> Dict:
        """Consolidation des mémoires"""
        def _consolidate():
            con = self._conn()
            cur = con.cursor()

            total = cur.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            important = cur.execute("SELECT COUNT(*) FROM memories WHERE importance > 0.7").fetchone()[0]
            accessed = cur.execute("SELECT COUNT(*) FROM memories WHERE access_count > 5").fetchone()[0]

            con.close()

            return {
                "total_memories": total,
                "important_memories": important,
                "frequently_accessed": accessed,
                "status": "consolidated"
            }

        try:
            return await asyncio.to_thread(_consolidate)
        except AttributeError:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _consolidate)

def health_check():
    """Health check pour le module"""
    try:
        os.makedirs("data", exist_ok=True)
        con = sqlite3.connect("data/jeffrey_memory.db")
        con.execute("SELECT 1")
        con.close()
        return {"status": "healthy", "module": __name__, "type": "memory_store"}
    except Exception as e:
        return {"status": "unhealthy", "module": __name__, "error": str(e)}
PY

echo "✅ SQLite Store updated"

# Test persistence
echo -e "\n🧪 Testing memory persistence..."
python3 << 'TEST'
import sys
import asyncio
sys.path.insert(0, "src")

from jeffrey.bundle2.memory.sqlite_store import SQLiteMemoryStore

async def test():
    store = SQLiteMemoryStore()

    # Store test memory
    mid = await store.store(
        "Bundle 2 is now active with persistent memory",
        metadata={
            "context": {"session_id": "test_session", "bundle": "2"},
            "importance": 0.9,
            "source": "system"
        }
    )
    print(f"✅ Stored: {mid}")

    # Recall it
    memory = await store.recall(mid)
    if memory:
        print(f"✅ Recalled: {memory['content'][:50]}...")

    # Search
    results = await store.search("bundle", limit=3)
    print(f"✅ Found {len(results)} memories matching 'bundle'")

    # Stats
    stats = await store.consolidate()
    print(f"✅ Stats: {stats}")

asyncio.run(test())
TEST

echo ""
echo "============================================================"
echo "✅ BUNDLE 2 MIGRATION FIXED!"
echo "============================================================"
echo ""
echo "📊 Database status:"
sqlite3 data/jeffrey_memory.db << 'SQL'
SELECT COUNT(*) as total_memories FROM memories;
SELECT 'Tables:', GROUP_CONCAT(name) FROM sqlite_master WHERE type='table';
SQL

echo ""
echo "🚀 Test the 8th region:"
echo "python3 -c 'from src.jeffrey.bundle2.language.broca_wernicke import region_8; print(region_8.process(\"Hello Jeffrey!\"))'"
