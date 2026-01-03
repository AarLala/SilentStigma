-- Quick SQL to create only the missing tables (comments and videos)
-- Run this in Supabase SQL Editor if you already have metrics tables

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

-- Enable RLS on new tables
ALTER TABLE comments ENABLE ROW LEVEL SECURITY;
ALTER TABLE videos ENABLE ROW LEVEL SECURITY;

-- RLS Policies for comments (allow anonymous read)
-- Drop first if exists to avoid errors
DROP POLICY IF EXISTS "Allow anonymous read on comments" ON comments;
CREATE POLICY "Allow anonymous read on comments" ON comments
    FOR SELECT USING (true);

-- RLS Policies for videos (allow anonymous read)
DROP POLICY IF EXISTS "Allow anonymous read on videos" ON videos;
CREATE POLICY "Allow anonymous read on videos" ON videos
    FOR SELECT USING (true);

