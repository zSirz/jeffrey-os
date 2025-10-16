-- Migration: add strength constraint with safe check
-- Grok optimization #3: Migration CHECK strength safe

-- Pré-check : identifier data invalide
DO $$
DECLARE
    invalid_count INTEGER;
BEGIN
    SELECT COUNT(*)
    INTO invalid_count
    FROM emotional_bonds
    WHERE strength NOT BETWEEN 0 AND 1;

    IF invalid_count > 0 THEN
        RAISE NOTICE 'WARNING: % bonds with invalid strength', invalid_count;
        RAISE NOTICE 'Clamping values to [0, 1]...';

        -- Clamp values invalides
        UPDATE emotional_bonds
        SET strength = CASE
            WHEN strength < 0 THEN 0
            WHEN strength > 1 THEN 1
            ELSE strength
        END
        WHERE strength NOT BETWEEN 0 AND 1;

        RAISE NOTICE 'Clamped % bonds', invalid_count;
    ELSE
        RAISE NOTICE 'All strength values are valid';
    END IF;
END
$$;

-- Ajouter contrainte CHECK après nettoyage
ALTER TABLE emotional_bonds
ADD CONSTRAINT strength_range CHECK (strength BETWEEN 0 AND 1);

COMMENT ON CONSTRAINT strength_range ON emotional_bonds IS 'Ensures strength values are between 0 and 1';