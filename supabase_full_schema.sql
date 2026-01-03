-- Complete Supabase Schema for SilenceVoice
-- This includes both metrics tracking AND main data tables (comments, videos)
-- Run this in your Supabase SQL Editor

-- ============================================
-- METRICS TABLES (for tracking usage)
-- ============================================

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

-- ============================================
-- MAIN DATA TABLES (comments and videos)
-- ============================================

-- Create comments table
CREATE TABLE IF NOT EXISTS comments (
    id TEXT PRIMARY KEY,
    video_id TEXT,
    channel_id TEXT,
    channel_name TEXT,
    author_name TEXT,
    text TEXT,
    like_count INTEGER DEFAULT 0,
    published_at TEXT,
    collected_at TIMESTAMPTZ DEFAULT NOW(),
    processed BOOLEAN DEFAULT FALSE
);

-- Create videos table
CREATE TABLE IF NOT EXISTS videos (
    video_id TEXT PRIMARY KEY,
    channel_id TEXT,
    channel_name TEXT,
    title TEXT,
    description TEXT,
    published_at TEXT,
    comment_count INTEGER DEFAULT 0,
    collected_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================
-- INDEXES (for fast queries)
-- ============================================

-- Metrics indexes
CREATE INDEX IF NOT EXISTS idx_download_events_client_id ON download_events(client_id);
CREATE INDEX IF NOT EXISTS idx_download_events_timestamp ON download_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_session_events_client_id ON session_events(client_id);
CREATE INDEX IF NOT EXISTS idx_session_events_timestamp ON session_events(timestamp);

-- Comments indexes (critical for search performance)
CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments(video_id);
CREATE INDEX IF NOT EXISTS idx_comments_channel_id ON comments(channel_id);
CREATE INDEX IF NOT EXISTS idx_comments_processed ON comments(processed);
CREATE INDEX IF NOT EXISTS idx_comments_published_at ON comments(published_at);
CREATE INDEX IF NOT EXISTS idx_comments_like_count ON comments(like_count);

-- Full-text search index for comments (PostgreSQL specific - much faster than LIKE queries)
CREATE INDEX IF NOT EXISTS idx_comments_text_gin ON comments USING gin(to_tsvector('english', text));

-- Videos indexes
CREATE INDEX IF NOT EXISTS idx_videos_channel_id ON videos(channel_id);
CREATE INDEX IF NOT EXISTS idx_videos_published_at ON videos(published_at);

-- ============================================
-- INITIALIZE DEFAULT VALUES
-- ============================================

-- Initialize default metric values
INSERT INTO metrics (key, value) VALUES 
    ('searches', 9382),
    ('downloads', 6503),
    ('exploratory_sessions', 21042)
ON CONFLICT (key) DO NOTHING;

-- ============================================
-- ROW LEVEL SECURITY (RLS)
-- ============================================

-- Enable RLS on all tables
ALTER TABLE metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE download_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE comments ENABLE ROW LEVEL SECURITY;
ALTER TABLE videos ENABLE ROW LEVEL SECURITY;

-- ============================================
-- RLS POLICIES
-- ============================================

-- Drop existing policies if they exist (to avoid errors on re-run)
DROP POLICY IF EXISTS "Allow anonymous read on metrics" ON metrics;
DROP POLICY IF EXISTS "Allow anonymous write on metrics" ON metrics;
DROP POLICY IF EXISTS "Allow anonymous insert on download_events" ON download_events;
DROP POLICY IF EXISTS "Allow anonymous select on download_events" ON download_events;
DROP POLICY IF EXISTS "Allow anonymous insert on session_events" ON session_events;
DROP POLICY IF EXISTS "Allow anonymous select on session_events" ON session_events;
DROP POLICY IF EXISTS "Allow anonymous read on comments" ON comments;
DROP POLICY IF EXISTS "Allow anonymous read on videos" ON videos;

-- Metrics policies (allow anonymous read/write for tracking)
CREATE POLICY "Allow anonymous read on metrics" ON metrics
    FOR SELECT USING (true);

CREATE POLICY "Allow anonymous write on metrics" ON metrics
    FOR ALL USING (true);

-- Download events policies
CREATE POLICY "Allow anonymous insert on download_events" ON download_events
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow anonymous select on download_events" ON download_events
    FOR SELECT USING (true);

-- Session events policies
CREATE POLICY "Allow anonymous insert on session_events" ON session_events
    FOR INSERT WITH CHECK (true);

CREATE POLICY "Allow anonymous select on session_events" ON session_events
    FOR SELECT USING (true);

-- Comments policies (allow anonymous read, but restrict writes to service role)
-- For public read access
CREATE POLICY "Allow anonymous read on comments" ON comments
    FOR SELECT USING (true);

-- Allow inserts for migration (using service role key bypasses RLS, but this allows anon too)
-- Note: In production, you may want to remove this and only use service role for writes
CREATE POLICY "Allow anonymous insert on comments" ON comments
    FOR INSERT WITH CHECK (true);

-- Videos policies (allow anonymous read)
CREATE POLICY "Allow anonymous read on videos" ON videos
    FOR SELECT USING (true);

-- Allow inserts for migration
CREATE POLICY "Allow anonymous insert on videos" ON videos
    FOR INSERT WITH CHECK (true);

-- Note: Write operations (INSERT/UPDATE/DELETE) on comments and videos
-- should be done using the service role key, not the anon key.
-- The anon key is sufficient for read operations.

