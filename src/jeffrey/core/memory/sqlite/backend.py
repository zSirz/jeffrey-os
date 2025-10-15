"""SQLite Backend with FTS5 for high-performance memory storage"""

import asyncio
import hashlib
import json
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from jeffrey.utils.logger import get_logger

logger = get_logger("SQLiteBackend")


class SQLiteMemoryBackend:
    """High-performance SQLite backend with FTS5 and WAL mode"""

    def __init__(self, db_path: str = "data/unified_memory.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = None
        self._writer_lock = asyncio.Lock()

        self.stats = {"writes": 0, "reads": 0, "write_time_total": 0.0, "read_time_total": 0.0}

        logger.info(f"📀 SQLite backend initialized: {self.db_path}")

    async def initialize(self):
        """Initialize database with optimal settings"""
        await self._execute_write(self._create_schema)
        await self._execute_write(self._set_pragmas)
        logger.info("✅ SQLite backend ready with FTS5")

    def _create_schema(self, conn: sqlite3.Connection):
        """Create optimized schema with indexes"""
        cursor = conn.cursor()

        # Main memory table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                ts REAL NOT NULL,
                data TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                access_count INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Indexes
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mem_type_ts
            ON memories(type, ts DESC)
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_mem_strength
            ON memories(strength DESC)
        """
        )

        # Check FTS5 availability
        self.has_fts5 = False
        try:
            # Test FTS5 by creating and dropping a test table
            cursor.execute("CREATE VIRTUAL TABLE IF NOT EXISTS _fts5_test USING fts5(c)")
            cursor.execute("DROP TABLE IF EXISTS _fts5_test")
            self.has_fts5 = True
            logger.info("✅ FTS5 support detected")
        except sqlite3.OperationalError:
            logger.warning("⚠️ FTS5 not available, using LIKE fallback")

        # Create FTS5 table only if available
        if self.has_fts5:
            try:
                cursor.execute(
                    """
                    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                    USING fts5(
                        id UNINDEXED,
                        content
                    )
                """
                )
                logger.info("FTS5 table created")
            except sqlite3.OperationalError as e:
                logger.error(f"FTS5 table creation failed: {e}")
                self.has_fts5 = False

        # Patterns table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS patterns (
                pattern TEXT PRIMARY KEY,
                count INTEGER DEFAULT 1,
                confidence REAL DEFAULT 0.5,
                last_seen REAL,
                tfidf REAL DEFAULT 0.0
            )
        """
        )

        # Emotional events table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS emotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts REAL NOT NULL,
                emotion TEXT NOT NULL,
                intensity REAL NOT NULL,
                trigger TEXT,
                resolved BOOLEAN DEFAULT 0
            )
        """
        )

        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_emotions_ts
            ON emotions(ts DESC)
        """
        )

        conn.commit()

    def _set_pragmas(self, conn: sqlite3.Connection):
        """Set optimal PRAGMAs for performance"""
        pragmas = [
            "PRAGMA journal_mode = WAL",
            "PRAGMA synchronous = NORMAL",
            "PRAGMA cache_size = -64000",  # 64MB cache
            "PRAGMA temp_store = MEMORY",
            "PRAGMA foreign_keys = ON",
        ]

        cursor = conn.cursor()
        for pragma in pragmas:
            try:
                cursor.execute(pragma)
            except sqlite3.OperationalError as e:
                logger.warning(f"Pragma failed (non-critical): {pragma} - {e}")
        conn.commit()

    @asynccontextmanager
    async def _get_connection(self):
        """Get database connection with proper handling"""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, isolation_level=None)
            self._conn.row_factory = sqlite3.Row

        yield self._conn

    async def _execute_write(self, func, *args):
        """Execute write operation with lock"""
        async with self._writer_lock:
            async with self._get_connection() as conn:
                start = time.perf_counter()
                result = func(conn, *args)
                elapsed = time.perf_counter() - start

                self.stats["writes"] += 1
                self.stats["write_time_total"] += elapsed

                return result

    async def _execute_read(self, func, *args):
        """Execute read operation"""
        async with self._get_connection() as conn:
            start = time.perf_counter()
            result = func(conn, *args)
            elapsed = time.perf_counter() - start

            self.stats["reads"] += 1
            self.stats["read_time_total"] += elapsed

            return result

    async def store_batch(self, records: list[dict[str, Any]]):
        """Store multiple records efficiently"""

        def _store(conn: sqlite3.Connection, records: list[dict]):
            cursor = conn.cursor()

            memory_data = []
            fts_data = []

            for record in records:
                if "_id" not in record:
                    record["_id"] = self._generate_id(record)

                if "_timestamp" not in record:
                    record["_timestamp"] = time.time()

                # Extract searchable text
                text_parts = []
                for key, value in record.items():
                    if not key.startswith("_") and isinstance(value, str):
                        text_parts.append(value)

                searchable_text = " ".join(text_parts)

                memory_data.append(
                    (
                        record["_id"],
                        record.get("type", "general"),
                        record["_timestamp"],
                        json.dumps(record, ensure_ascii=False),
                        record.get("strength", 1.0),
                        record.get("access_count", 0),
                    )
                )

                fts_data.append((record["_id"], searchable_text))

            # Batch insert
            cursor.executemany(
                """
                INSERT OR REPLACE INTO memories (id, type, ts, data, strength, access_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                memory_data,
            )

            # FTS5 - Check if available
            if getattr(self, "has_fts5", False):
                # FTS5 available - use DELETE + INSERT
                for fts_id, content in fts_data:
                    try:
                        cursor.execute("DELETE FROM memories_fts WHERE id = ?", (fts_id,))
                        cursor.execute(
                            "INSERT INTO memories_fts (id, content) VALUES (?, ?)",
                            (fts_id, content),
                        )
                    except sqlite3.OperationalError:
                        pass  # Ignore FTS5 errors

            conn.commit()
            return len(records)

        count = await self._execute_write(_store, records)
        logger.debug(f"📝 Stored batch of {count} records")
        return count

    async def store_one(self, record: dict[str, Any]):
        """Store single record"""
        return await self.store_batch([record])

    async def query(self, filter_dict: dict[str, Any]) -> list[dict[str, Any]]:
        """Query with optimized SQL"""

        def _query(conn: sqlite3.Connection, filter_dict: dict):
            cursor = conn.cursor()

            conditions = []
            params = []

            if "type" in filter_dict:
                conditions.append("type = ?")
                params.append(filter_dict["type"])

            if "_timestamp_gte" in filter_dict:
                conditions.append("ts >= ?")
                params.append(filter_dict["_timestamp_gte"])

            if "_timestamp_lte" in filter_dict:
                conditions.append("ts <= ?")
                params.append(filter_dict["_timestamp_lte"])

            where_clause = " AND ".join(conditions) if conditions else "1=1"
            limit = filter_dict.get("limit", 100)

            query = f"""
                SELECT data FROM memories
                WHERE {where_clause}
                ORDER BY ts DESC
                LIMIT ?
            """
            params.append(limit)

            cursor.execute(query, params)

            results = []
            for row in cursor.fetchall():
                results.append(json.loads(row["data"]))

            return results

        return await self._execute_read(_query, filter_dict)

    async def search_text(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search with automatic FTS5/LIKE fallback"""

        def _search(conn: sqlite3.Connection, query: str, limit: int):
            cursor = conn.cursor()
            results = []

            if getattr(self, "has_fts5", False):
                # Try FTS5 first
                try:
                    cursor.execute(
                        """
                        SELECT m.data, bm25(memories_fts) as rank
                        FROM memories_fts f
                        JOIN memories m ON f.id = m.id
                        WHERE memories_fts MATCH ?
                        ORDER BY bm25(memories_fts)
                        LIMIT ?
                    """,
                        (query, limit),
                    )

                    for row in cursor.fetchall():
                        data = json.loads(row["data"])
                        data["_search_rank"] = row["rank"]
                        results.append(data)

                    logger.debug(f"FTS5 search: {len(results)} results")
                    return results

                except sqlite3.OperationalError as e:
                    logger.warning(f"FTS5 failed, using LIKE: {e}")
                    self.has_fts5 = False

            # Fallback to LIKE
            escaped_query = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            like_pattern = f"%{escaped_query}%"

            cursor.execute(
                """
                SELECT data, 0 as rank
                FROM memories
                WHERE data LIKE ? ESCAPE '\\'
                ORDER BY ts DESC
                LIMIT ?
            """,
                (like_pattern, limit),
            )

            for row in cursor.fetchall():
                data = json.loads(row["data"])
                data["_search_rank"] = 0
                results.append(data)

            logger.debug(f"LIKE search: {len(results)} results")
            return results

        return await self._execute_read(_search, query, limit)

    def _generate_id(self, record: dict) -> str:
        """Generate unique ID for record"""
        content = json.dumps(record, sort_keys=True)
        hash_part = hashlib.md5(content.encode()).hexdigest()[:8]
        time_part = str(int(time.time() * 1000000))[-8:]
        return f"{hash_part}-{time_part}"

    async def vacuum(self):
        """Optimize database"""

        def _vacuum(conn: sqlite3.Connection):
            cursor = conn.cursor()
            cursor.execute("VACUUM")
            cursor.execute("ANALYZE")
            conn.commit()

        await self._execute_write(_vacuum)
        logger.info("🔧 Database optimized")

    async def get_stats(self) -> dict[str, Any]:
        """Get backend statistics"""

        def _stats(conn: sqlite3.Connection):
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) as count FROM memories")
            memory_count = cursor.fetchone()["count"]

            return {"memory_count": memory_count}

        backend_stats = await self._execute_read(_stats)

        backend_stats.update(
            {
                "total_writes": self.stats["writes"],
                "total_reads": self.stats["reads"],
                "avg_write_ms": (self.stats["write_time_total"] / max(self.stats["writes"], 1)) * 1000,
                "avg_read_ms": (self.stats["read_time_total"] / max(self.stats["reads"], 1)) * 1000,
            }
        )

        return backend_stats

    async def shutdown(self):
        """Clean shutdown"""
        if self._conn:
            await self.vacuum()
            self._conn.close()
            self._conn = None
        logger.info("📀 SQLite backend shutdown complete")
