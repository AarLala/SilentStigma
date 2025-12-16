# Supabase Setup Guide

This project uses Supabase to track three metrics:
1. **Searches** - Incremented every time a search is performed (no restrictions)
2. **Downloads** - Tracked once per computer/system (no duplicates per client)
3. **Exploratory Sessions** - Tracked once per hour per computer/system

## Setup Instructions

### 1. Create a Supabase Project

1. Go to [supabase.com](https://supabase.com) and create a new project
2. Note your project URL and anon/public key from Settings > API

### 2. Set Environment Variables

Add these to your `.env` file or set them as environment variables:

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-public-key
```

### 3. Run the SQL Schema

In your Supabase SQL Editor, run the following SQL to create the necessary tables:

```sql
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
```

### 4. Set Row Level Security (RLS)

For security, enable RLS and create policies:

```sql
-- Enable RLS on all tables
ALTER TABLE metrics ENABLE ROW LEVEL SECURITY;
ALTER TABLE download_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE session_events ENABLE ROW LEVEL SECURITY;

-- Allow anonymous reads on metrics (for displaying counts)
CREATE POLICY "Allow anonymous read on metrics" ON metrics
    FOR SELECT USING (true);

-- Allow anonymous inserts/updates on metrics (for tracking)
CREATE POLICY "Allow anonymous write on metrics" ON metrics
    FOR ALL USING (true);

-- Allow anonymous inserts on download_events
CREATE POLICY "Allow anonymous insert on download_events" ON download_events
    FOR INSERT WITH CHECK (true);

-- Allow anonymous inserts on session_events
CREATE POLICY "Allow anonymous insert on session_events" ON session_events
    FOR INSERT WITH CHECK (true);

-- Allow anonymous selects on download_events (for checking if client already downloaded)
CREATE POLICY "Allow anonymous select on download_events" ON download_events
    FOR SELECT USING (true);

-- Allow anonymous selects on session_events (for checking last session time)
CREATE POLICY "Allow anonymous select on session_events" ON session_events
    FOR SELECT USING (true);
```

## How It Works

### Search Tracking
- Every time a user performs a search, the count is incremented
- No restrictions - all searches are counted

### Download Tracking
- Uses a client ID generated from browser headers (User-Agent, Accept-Language)
- Each unique client can only be counted once
- The `download_events` table uses a UNIQUE constraint on `client_id` to prevent duplicates

### Session Tracking
- Tracks exploratory sessions (when the dashboard loads)
- Each client can only be counted once per hour
- Checks the last session timestamp before incrementing

## Testing

After setup, you can test the endpoints:

```bash
# Track a search
curl -X POST http://localhost:5000/api/track/search

# Track a download
curl -X POST http://localhost:5000/api/track/download

# Track a session
curl -X POST http://localhost:5000/api/track/session

# Get all metrics
curl http://localhost:5000/api/metrics
```

## Troubleshooting

If metrics aren't tracking:
1. Check that environment variables are set correctly
2. Verify the Supabase tables exist and have the correct schema
3. Check the Flask logs for Supabase connection errors
4. Ensure RLS policies allow the operations you need

