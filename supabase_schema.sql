-- Supabase Schema for SilentStigma Metrics Tracking
-- Run this in your Supabase SQL Editor

-- Create metrics table to store counts
CREATE TABLE IF NOT EXISTS metrics (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL DEFAULT 0
);

-- Create download_events table to track unique downloads per client
CREATE TABLE IF NOT EXISTS download_events (
    id BIGSERIAL PRIMARY KEY,
    client_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(client_id)
);

-- Create session_events table to track exploratory sessions
CREATE TABLE IF NOT EXISTS session_events (
    id BIGSERIAL PRIMARY KEY,
    client_id TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_download_events_client_id ON download_events(client_id);
CREATE INDEX IF NOT EXISTS idx_download_events_timestamp ON download_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_session_events_client_id ON session_events(client_id);
CREATE INDEX IF NOT EXISTS idx_session_events_timestamp ON session_events(timestamp);

-- Initialize default metric values
INSERT INTO metrics (key, value) VALUES 
    ('searches', 9382),
    ('downloads', 6503),
    ('exploratory_sessions', 21042)
ON CONFLICT (key) DO NOTHING;

-- Enable Row Level Security
ALTER TABLE metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE download_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_events ENABLE ROW LEVEL SECURITY;

-- Create policies for anonymous access
CREATE POLICY "Allow anonymous read on metrics" ON metrics
    FOR SELECT USING (true);

CREATE POLICY "Allow anonymous write on metrics" ON metrics
    FOR ALL USING (true);

CREATE POLICY "Allow anonymous insert on download_events" ON download_events
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow anonymous insert on session_events" ON session_events
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow anonymous select on download_events" ON download_events
    FOR SELECT USING (true);

CREATE POLICY "Allow anonymous select on session_events" ON session_events
    FOR SELECT USING (true);

