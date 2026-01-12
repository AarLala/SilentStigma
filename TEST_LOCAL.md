# Quick Local Testing Guide

## 1. Test Locally

```bash
# Make sure you have .env file with your settings
# Then run:
python -m src.dashboard.app
```

Or with Gunicorn (production-like):
```bash
gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 4 src.dashboard.app:application
```

Then open: http://localhost:5000/dashboard

## 2. Check if Using Supabase or Local DB

### Method 1: Check Logs
When the app starts, look for these messages:
- ✅ **Using Supabase**: `"Using Supabase for data queries (faster than SQLite)"`
- ❌ **Using SQLite**: `"Supabase credentials not found"` or `"Falling back to SQLite"`

### Method 2: Check .env File
```bash
# If these are set, you're using Supabase:
USE_SUPABASE=true
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
```

### Method 3: Check Health Endpoint
```bash
curl http://localhost:5000/health
```
Response shows: `"supabase_enabled": true` or `false`

### Method 4: Check App Startup
Look at console output when starting:
- **Supabase**: `"Supabase client initialized successfully"`
- **SQLite**: No Supabase message, uses local `data/*.db` file

## Quick Test Commands

```bash
# Test search endpoint
curl "http://localhost:5000/api/search?q=pain&limit=5"

# Test health
curl http://localhost:5000/health

# Test stats
curl http://localhost:5000/api/stats
```

