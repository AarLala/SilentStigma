# Testing Supabase Integration

## Step 1: Run the SQL Schema

Before testing, make sure you've set up your Supabase database:

1. Go to your Supabase project dashboard
2. Click on "SQL Editor" in the left sidebar
3. Open the file `supabase_schema.sql` from this project
4. Copy and paste the entire SQL into the editor
5. Click "Run" to execute the SQL

This will create:
- `metrics` table (stores search/download/session counts)
- `download_events` table (tracks unique downloads per client)
- `session_events` table (tracks exploratory sessions)
- All necessary indexes and security policies

## Step 2: Verify Environment Variables

Make sure your `.env` file has the correct format:

```env
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-anon-public-key-here
```

**Important:** 
- The URL should start with `https://` and end with `.supabase.co`
- The KEY should be the "anon" or "public" key from Supabase Settings > API
  - It should look like: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` (a long JWT token)
  - NOT the "service_role" key (that's secret)
  - NOT keys starting with `sb_` (those are for other Supabase services)
- No quotes around the values

**How to find the correct key:**
1. Go to your Supabase project dashboard
2. Click "Settings" (gear icon) in the left sidebar
3. Click "API" under Project Settings
4. Look for "Project API keys"
5. Copy the "anon" or "public" key (it's safe to expose this one)
6. It should be a long string starting with `eyJ...`

## Step 3: Run the Test Script

From the project root directory, run:

```bash
python test_supabase.py
```

This will test:
- ✅ Connection to Supabase
- ✅ Reading metrics table
- ✅ Search tracking functionality
- ✅ Download tracking (with duplicate prevention)
- ✅ Session tracking functionality

**Expected Output:**
```
============================================================
Testing Supabase Integration
============================================================
✓ Found SUPABASE_URL: https://xxxxx.supabase.co...
✓ Found SUPABASE_KEY: eyJhbGciOiJIUzI1NiIs...
✓ Supabase client created successfully

--- Test 1: Reading metrics table ---
✓ Metrics table exists and has 3 rows:
   - searches: 9382
   - downloads: 6503
   - exploratory_sessions: 21042

--- Test 2: Testing search tracking ---
   Current search count: 9382
✓ Search count incremented to: 9383

--- Test 3: Testing download tracking ---
✓ Test download event created for client: test_client_...
✓ Duplicate download correctly prevented (UNIQUE constraint working)
✓ Test download event cleaned up

--- Test 4: Testing session tracking ---
✓ Test session event created for client: test_session_...
✓ Session event retrieved: 2024-...
✓ Test session event cleaned up

============================================================
✅ All tests passed! Supabase integration is working correctly.
============================================================
```

## Step 4: Test via the Web Dashboard

1. **Start the Flask server:**
   ```bash
   python -m src.dashboard.app
   ```
   Or if you have a run script:
   ```bash
   python run_dashboard.py
   ```

2. **Check the server logs** - You should see:
   ```
   INFO: Supabase client initialized successfully
   ```

3. **Open the dashboard** in your browser (usually `http://localhost:5000`)

4. **Test search tracking:**
   - Go to the "Language lens" tab
   - Enter a search term (e.g., "pain")
   - Click "Explore" or press Enter
   - The search count should increment in the grey badge

5. **Test download tracking:**
   - Click any "Download" button (processed dataset, cluster dataset, etc.)
   - The download should work and be tracked
   - Each computer can only be counted once

6. **Test session tracking:**
   - Refresh the page
   - A session should be tracked (once per hour per computer)

## Step 5: Verify in Supabase Dashboard

1. Go to your Supabase project dashboard
2. Click on "Table Editor" in the left sidebar
3. Check the tables:

   **metrics table:**
   - Should show `searches`, `downloads`, `exploratory_sessions`
   - Values should increase as you use the dashboard

   **download_events table:**
   - Should show one row per unique computer that downloaded
   - Each `client_id` should be unique

   **session_events table:**
   - Should show session events with timestamps
   - Multiple rows per client are OK (one per hour)

## Troubleshooting

### "Supabase credentials not found"
- Check that your `.env` file is in the project root
- Verify the variable names are exactly `SUPABASE_URL` and `SUPABASE_KEY`
- Make sure there are no extra spaces or quotes

### "Failed to create Supabase client"
- Verify your URL and KEY are correct
- Check that your Supabase project is active
- Make sure you're using the "anon" public key, not the service role key

### "Could not read metrics table"
- Run the SQL schema from `supabase_schema.sql` in Supabase SQL Editor
- Check that the table was created successfully

### "Row Level Security policy violation"
- Make sure you ran the RLS policies from the SQL schema
- Check that the policies allow anonymous access (for public dashboard)

### Search count not updating
- Check browser console for JavaScript errors
- Verify the `/api/track/search` endpoint is being called
- Check Flask server logs for errors

### Downloads not being tracked
- Make sure you're clicking the download buttons (not direct links)
- Check that the `download_events` table has the UNIQUE constraint
- Verify the client ID is being generated correctly

## Manual API Testing

You can also test the endpoints directly:

```bash
# Get current metrics
curl http://localhost:5000/api/metrics

# Track a search
curl -X POST http://localhost:5000/api/track/search

# Track a download
curl -X POST http://localhost:5000/api/track/download

# Track a session
curl -X POST http://localhost:5000/api/track/session
```

## Next Steps

Once everything is working:
1. ✅ Your metrics will be tracked in Supabase
2. ✅ The search count will update in real-time
3. ✅ Downloads are tracked once per computer
4. ✅ Sessions are tracked once per hour per computer

The dashboard will automatically use Supabase if credentials are available, or fall back to SQLite if not.

