-- Indexes pour améliorer les performances
CREATE INDEX IF NOT EXISTS idx_memories_emotion ON memories(emotion);
CREATE INDEX IF NOT EXISTS idx_memories_processed ON memories(processed);
CREATE INDEX IF NOT EXISTS idx_memories_composite ON memories(processed, timestamp DESC);

-- Index pour recherche textuelle
CREATE INDEX IF NOT EXISTS idx_memories_text_gin ON memories USING gin(to_tsvector('english', text));

-- Analyze pour mettre à jour les stats
ANALYZE memories;