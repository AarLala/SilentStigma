# Migration Guide: SQLite to Supabase

This guide will help you migrate your SilenceVoice database from SQLite to Supabase for faster, cloud-based access.

## Why Migrate to Supabase?

1. **Faster Access**: Supabase uses PostgreSQL with optimized indexes and full-text search
2. **Cloud-Based**: Access your data from anywhere, not just local files
3. **Better Performance**: PostgreSQL is much faster than SQLite for large datasets
4. **Scalability**: Supabase can handle much larger datasets than SQLite
5. **Production Ready**: Better suited for production deployments on Fly.io

## Prerequisites

1. A Supabase account (free tier is sufficient)
2. Your existing SQLite database at `data/silencevoice.db`
3. Python environment with dependencies installed

## Step-by-Step Migration

### Step 1: Create Supabase Project

1. Go to [supabase.com](https://supabase.com) and sign up/login
2. Click **New Project**
3. Fill in:
   - **Name**: `silencevoice` (or your choice)
   - **Database Password**: Choose a strong password (save it!)
   - **Region**: Choose closest to you
4. Wait for project to be created (2-3 minutes)

### Step 2: Get Your Credentials

1. In Supabase dashboard, go to **Settings > API**
2. Copy these values:
   - **Project URL**: `https://xxxxx.supabase.co`
   - **anon public key**: Long string starting with `eyJ...`
   - **service_role key**: Another long string (keep this secret!)

### Step 3: Create Database Schema

1. In Supabase dashboard, go to **SQL Editor**
2. Click **New Query**
3. Open `supabase_full_schema.sql` from this project
4. Copy the entire contents and paste into SQL Editor
5. Click **Run** (or press Ctrl+Enter)
6. You should see "Success. No rows returned"

**Verify tables were created:**
- Go to **Table Editor**
- You should see: `comments`, `videos`, `metrics`, `download_events`, `session_events`

### Step 4: Set Up Environment Variables

Create a `.env` file in your project root (if you don't have one):

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-public-key
SUPABASE_SERVICE_KEY=your-service-role-key
```

**Important**: 
- Add `.env` to `.gitignore` (never commit secrets!)
- Use `SUPABASE_SERVICE_KEY` for migrations (has write permissions)
- Use `SUPABASE_KEY` (anon key) for the app (read-only for public data)

### Step 5: Run Migration Script

1. Make sure your SQLite database exists at `data/silencevoice.db`
2. Run the migration script:

```bash
python migrate_to_supabase.py
```

The script will:
- Check your SQLite database
- Show how many records will be migrated
- Ask for confirmation
- Migrate data in batches (for efficiency)
- Show progress with a progress bar
- Display summary when complete

**Migration Time Estimates:**
- 10,000 comments: ~1-2 minutes
- 100,000 comments: ~10-15 minutes
- 1,000,000 comments: ~1-2 hours

### Step 6: Verify Migration

1. In Supabase dashboard, go to **Table Editor**
2. Check `comments` table - should have your data
3. Check `videos` table - should have your data
4. Compare row counts with your SQLite database:

```bash
# Check SQLite counts
sqlite3 data/silencevoice.db "SELECT COUNT(*) FROM comments;"
sqlite3 data/silencevoice.db "SELECT COUNT(*) FROM videos;"

# Check Supabase counts (in SQL Editor)
SELECT COUNT(*) FROM comments;
SELECT COUNT(*) FROM videos;
```

### Step 7: Enable Supabase in Your App

Update your `.env` file to enable Supabase:

```bash
USE_SUPABASE=true
```

Or set it as an environment variable:

```bash
export USE_SUPABASE=true
```

### Step 8: Test Your Application

1. Start your Flask app:
   ```bash
   python -m src.dashboard.app
   ```

2. Check logs - you should see:
   ```
   Supabase client initialized successfully
   Using Supabase for data queries (faster than SQLite)
   ```

3. Test the dashboard:
   - Visit `http://localhost:5000/dashboard`
   - Check that stats load correctly
   - Try searching for comments
   - Verify everything works as before

## Troubleshooting

### Migration Fails with "Permission Denied"

**Problem**: Using anon key instead of service role key

**Solution**: 
- Make sure `SUPABASE_SERVICE_KEY` is set in `.env`
- Service role key has full permissions (anon key is read-only)

### Migration is Very Slow

**Problem**: Large dataset, network issues

**Solutions**:
- Migration uses batches - be patient
- Check your internet connection
- Supabase free tier has rate limits - may need to wait between batches

### "Table does not exist" Error

**Problem**: Schema wasn't created

**Solution**:
- Go to Supabase SQL Editor
- Run `supabase_full_schema.sql` again
- Verify tables exist in Table Editor

### Data Counts Don't Match

**Problem**: Some rows failed to migrate

**Solution**:
- Check migration script output for errors
- Common issues:
  - Invalid characters in text fields
  - NULL values in required fields
  - Duplicate primary keys
- Re-run migration (it will skip existing rows)

### App Can't Connect to Supabase

**Problem**: Wrong credentials or network issue

**Solution**:
1. Verify credentials in `.env`:
   ```bash
   echo $SUPABASE_URL
   echo $SUPABASE_KEY
   ```

2. Test connection:
   ```bash
   python test_supabase.py
   ```

3. Check Supabase project is active (not paused)

## Performance Comparison

### SQLite (Local)
- ✅ Fast for small datasets (< 100K rows)
- ✅ No network latency
- ❌ Slower for large datasets
- ❌ Single-file bottleneck
- ❌ Not suitable for production

### Supabase (Cloud)
- ✅ Fast for any dataset size
- ✅ Optimized indexes
- ✅ Full-text search (PostgreSQL)
- ✅ Concurrent access
- ✅ Production-ready
- ⚠️ Network latency (~50-200ms)
- ⚠️ Requires internet connection

**For most use cases, Supabase will be faster**, especially with:
- Large datasets (> 100K rows)
- Multiple concurrent users
- Complex queries
- Full-text search

## Next Steps

After successful migration:

1. **Deploy to Fly.io**: See `DEPLOYMENT.md`
2. **Monitor Usage**: Check Supabase dashboard for usage stats
3. **Optimize Queries**: Use Supabase's query analyzer
4. **Set Up Backups**: Configure automatic backups in Supabase

## Rollback Plan

If you need to rollback to SQLite:

1. Set `USE_SUPABASE=false` in `.env`
2. Your SQLite database is still at `data/silencevoice.db`
3. App will automatically use SQLite again

**Note**: Your data is safe - Supabase migration doesn't delete SQLite data.

## Support

- Supabase Docs: https://supabase.com/docs
- Migration Issues: Check script output for specific errors
- Database Issues: Check Supabase logs in dashboard

