-- Migration: add composite index on emotional_bonds
-- Grok optimization #2: Index composite pour get_bonds_for_memory

-- Index composite pour memory_id_a avec strength DESC (pour ordering)
CREATE INDEX IF NOT EXISTS idx_bonds_memory_a_strength
ON emotional_bonds (memory_id_a, strength DESC);

-- Index composite pour memory_id_b avec strength DESC (pour ordering)
CREATE INDEX IF NOT EXISTS idx_bonds_memory_b_strength
ON emotional_bonds (memory_id_b, strength DESC);

-- Index pour recherche rapide par paire ordonnée (déjà existant?)
CREATE INDEX IF NOT EXISTS idx_bonds_memory_pair
ON emotional_bonds (memory_id_a, memory_id_b);

COMMENT ON INDEX idx_bonds_memory_a_strength IS 'Composite index for bonds lookup by memory_id_a with strength ordering';
COMMENT ON INDEX idx_bonds_memory_b_strength IS 'Composite index for bonds lookup by memory_id_b with strength ordering';