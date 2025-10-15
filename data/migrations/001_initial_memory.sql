-- Jeffrey OS Bundle 2 - Memory Schema
-- Enhanced with relations and dream states

CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    context TEXT,  -- JSON
    emotions TEXT, -- JSON
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    importance REAL DEFAULT 0.5,
    access_count INTEGER DEFAULT 0,
    last_accessed DATETIME,
    vector BLOB,   -- Pour futurs embeddings
    source TEXT,   -- D'où vient la mémoire
    session_id TEXT -- Pour grouper par session
);

CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories(timestamp);
CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories(importance);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories(session_id);

-- Table pour les liens entre mémoires
CREATE TABLE IF NOT EXISTS memory_links (
    source_id TEXT,
    target_id TEXT,
    link_type TEXT, -- 'causes', 'relates_to', 'contradicts', etc.
    strength REAL DEFAULT 0.5,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (source_id, target_id),
    FOREIGN KEY (source_id) REFERENCES memories(id),
    FOREIGN KEY (target_id) REFERENCES memories(id)
);

-- Table pour les états de conscience
CREATE TABLE IF NOT EXISTS consciousness_states (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    state TEXT NOT NULL, -- JSON
    cycle_count INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    emotions_summary TEXT, -- JSON
    active_regions TEXT    -- JSON liste des régions actives
);

-- Table pour les rêves (consolidation nocturne)
CREATE TABLE IF NOT EXISTS dreams (
    id TEXT PRIMARY KEY,
    dream_content TEXT,    -- JSON avec les insights générés
    memories_processed TEXT, -- JSON liste des IDs traités
    patterns_found TEXT,   -- JSON patterns découverts
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
