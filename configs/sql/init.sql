-- Jeffrey P2 - PostgreSQL Schema

-- Extensions essentielles
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Table events
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    service VARCHAR(50) NOT NULL,
    type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes optimisés
CREATE INDEX idx_events_timestamp ON events(timestamp DESC);
CREATE INDEX idx_events_service_type ON events(service, type);
CREATE INDEX idx_events_payload_gin ON events USING GIN(payload);

-- Table metrics
CREATE TABLE IF NOT EXISTS metrics (
    time TIMESTAMPTZ NOT NULL,
    service VARCHAR(50) NOT NULL,
    metric VARCHAR(100) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    tags JSONB DEFAULT '{}'
);

CREATE INDEX idx_metrics_time ON metrics(time DESC);
CREATE INDEX idx_metrics_service ON metrics(service);

-- Permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO jeffrey;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO jeffrey;
