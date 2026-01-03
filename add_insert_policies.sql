-- Add INSERT policies for comments and videos tables
-- Run this in Supabase SQL Editor to allow inserts for migration

-- Drop existing policies if they exist
DROP POLICY IF EXISTS "Allow anonymous insert on comments" ON comments;
DROP POLICY IF EXISTS "Allow anonymous insert on videos" ON videos;

-- Comments: Allow anonymous inserts (for migration)
CREATE POLICY "Allow anonymous insert on comments" ON comments
    FOR INSERT WITH CHECK (true);

-- Videos: Allow anonymous inserts (for migration)
CREATE POLICY "Allow anonymous insert on videos" ON videos
    FOR INSERT WITH CHECK (true);

