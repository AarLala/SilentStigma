# Testing Your App Locally with Supabase

This guide shows you how to test that your local Flask app works with your Supabase database.

## Quick Test

Run the connection test script:

```bash
python test_supabase_connection.py
```

This will test:
- ✅ Connection to Supabase
- ✅ Reading data (comments, videos, metrics)
- ✅ Writing data (updating metrics)
- ✅ App integration

## Step-by-Step Testing

### 1. Verify Environment Variables

Make sure your `.env` file has:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
USE_SUPABASE=true
```

### 2. Test Connection

```bash
python test_supabase_connection.py
```

Expected output:
```
✅ All tests passed! Your Supabase connection is working.
```

### 3. Start Your Flask App

```bash
python -m src.dashboard.app
```

Look for these messages in the console:
```
INFO: Supabase client initialized successfully
INFO: Using Supabase for data queries (faster than SQLite)
```

### 4. Test the Dashboard

1. Open your browser: http://localhost:5000/dashboard
2. Check that stats load (should show comment/video counts from Supabase)
3. Try a search query
4. Check that clusters are displayed
5. Verify metrics are tracked

### 5. Verify Data Source

To confirm you're using Supabase (not SQLite):

1. Check Flask logs - should say "Using Supabase for data queries"
2. Query Supabase directly:
   ```bash
   python test_supabase_connection.py
   ```
3. Compare counts - should match what you see in the dashboard

## Testing Specific Features

### Test Search Functionality

1. Go to dashboard
2. Enter a search query (e.g., "support", "therapy")
3. Verify results appear
4. Check browser console for errors (F12 → Console)

### Test Metrics Tracking

1. Perform a search in the dashboard
2. Check Supabase dashboard → Table Editor → `metrics` table
3. The `searches` value should increment

### Test Data Loading

1. Check dashboard stats (top of page)
2. Should show:
   - Total comments (from Supabase)
   - Total videos (from Supabase)
   - Processed comments
   - Cluster information

## Troubleshooting

### "Supabase client not initialized"

**Fix**: Check your `.env` file has correct `SUPABASE_URL` and `SUPABASE_KEY`

### "Using SQLite" instead of Supabase

**Fix**: Set `USE_SUPABASE=true` in your `.env` file

### Stats show 0 or wrong numbers

**Possible causes**:
1. Data not migrated yet - run `python migrate_to_supabase.py`
2. Wrong database - check `USE_SUPABASE` setting
3. RLS blocking reads - verify policies in Supabase

### Search not working

**Check**:
1. `processed_comments.csv` exists in `outputs/` folder
2. File has data (check file size)
3. Flask logs for errors

## Comparing SQLite vs Supabase

To verify you're using Supabase:

### Check SQLite (if USE_SUPABASE=false):
```bash
sqlite3 data/silencevoice.db "SELECT COUNT(*) FROM comments;"
```

### Check Supabase (if USE_SUPABASE=true):
```bash
python test_supabase_connection.py
```

The counts should match if migration was successful.

## Performance Comparison

You should notice:
- **Faster queries** with Supabase (especially for large datasets)
- **Better scalability** - can handle more concurrent users
- **Cloud access** - data accessible from anywhere

## Next Steps

Once local testing passes:

1. ✅ Verify all features work
2. ✅ Check performance is acceptable
3. ✅ Test with production data volumes
4. ✅ Deploy to Fly.io (see DEPLOYMENT.md)

## Quick Commands Reference

```bash
# Test Supabase connection
python test_supabase_connection.py

# Start Flask app
python -m src.dashboard.app

# Test Supabase integration (existing)
python test_supabase.py

# Verify schema
python verify_supabase_schema.py
```

