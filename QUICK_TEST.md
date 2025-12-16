# Quick Test Guide

## Your Supabase Key Issue

The test shows your key starts with `sb_publishable_...` which is not the correct format.

**You need the "anon" or "public" key, which looks like:**
```
eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFmaG1maXB6eW5iYXNseHB5Y2h2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTIzNDU2NzgsImV4cCI6MjAyNzk0MTY3OH0...
```

## How to Get the Correct Key

1. Go to your Supabase project: https://supabase.com/dashboard
2. Select your project
3. Click **Settings** (gear icon) → **API**
4. Under **Project API keys**, find the **"anon"** or **"public"** key
5. It should be a very long string starting with `eyJ...`
6. Copy it to your `.env` file as `SUPABASE_KEY=`

## After Updating Your .env File

1. **Run the test again:**
   ```bash
   python test_supabase.py
   ```

2. **Make sure you've run the SQL schema:**
   - Go to Supabase SQL Editor
   - Run the contents of `supabase_schema.sql`

3. **Start your Flask server and check logs:**
   ```bash
   python -m src.dashboard.app
   ```
   
   You should see: `INFO: Supabase client initialized successfully`

## Quick Verification

Once the test passes, you can verify it's working by:

1. **Check the dashboard search count** - it should show a number
2. **Perform a search** - the count should increment
3. **Check Supabase dashboard** - go to Table Editor → `metrics` table to see the values

