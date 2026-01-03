# SilenceVoice - Deployment & Supabase Setup

This document summarizes the deployment setup for hosting SilenceVoice on Fly.io with Supabase as the database.

## 📋 What's Been Set Up

### ✅ Fly.io Configuration
- **Dockerfile**: Production-ready container with all dependencies
- **fly.toml**: Fly.io app configuration
- **.dockerignore**: Optimized build context

### ✅ Supabase Integration
- **supabase_full_schema.sql**: Complete database schema (comments, videos, metrics)
- **migrate_to_supabase.py**: Script to migrate data from SQLite to Supabase
- **Updated app.py**: Supports both SQLite (backward compatible) and Supabase

### ✅ Documentation
- **DEPLOYMENT.md**: Complete deployment guide
- **MIGRATION_GUIDE.md**: Step-by-step migration instructions
- **QUICK_DEPLOY.md**: Quick start guide

## 🚀 Quick Start

### Option 1: Use Supabase (Recommended - Faster)

1. **Set up Supabase** (see MIGRATION_GUIDE.md)
2. **Migrate data**: `python migrate_to_supabase.py`
3. **Deploy to Fly.io**: `fly deploy` (see DEPLOYMENT.md)

### Option 2: Keep SQLite (For Development)

- Just deploy: `fly deploy`
- App will use SQLite by default (slower but works)

## 🔑 Environment Variables

Required for Supabase:
```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-public-key
USE_SUPABASE=true  # Set to 'true' to use Supabase
```

Optional:
```bash
SUPABASE_SERVICE_KEY=your-service-role-key  # For migrations only
PORT=8080  # Default port
```

## 📊 Performance Comparison

| Feature | SQLite | Supabase |
|---------|--------|----------|
| Small datasets (<100K) | ✅ Fast | ✅ Fast |
| Large datasets (>100K) | ❌ Slow | ✅ Fast |
| Full-text search | ❌ Basic | ✅ PostgreSQL FTS |
| Concurrent users | ❌ Limited | ✅ Excellent |
| Cloud access | ❌ No | ✅ Yes |
| Production ready | ❌ No | ✅ Yes |

## 🎯 Key Benefits of Supabase

1. **Faster Queries**: PostgreSQL with optimized indexes
2. **Full-Text Search**: Built-in PostgreSQL FTS (much faster than LIKE)
3. **Scalability**: Handles millions of rows easily
4. **Cloud Access**: Access from anywhere, not just local files
5. **Production Ready**: Better for hosting on Fly.io

## 📁 File Structure

```
.
├── Dockerfile                 # Container definition
├── fly.toml                  # Fly.io config
├── .dockerignore             # Build optimization
├── supabase_full_schema.sql  # Complete Supabase schema
├── migrate_to_supabase.py    # Migration script
├── DEPLOYMENT.md             # Full deployment guide
├── MIGRATION_GUIDE.md        # Migration instructions
└── QUICK_DEPLOY.md           # Quick start
```

## 🔧 How It Works

### Database Selection

The app automatically chooses the database based on `USE_SUPABASE`:

- `USE_SUPABASE=false` (default): Uses SQLite from `data/silencevoice.db`
- `USE_SUPABASE=true`: Uses Supabase for all queries

### Backward Compatibility

- SQLite database is **never deleted** during migration
- You can switch back anytime by setting `USE_SUPABASE=false`
- Both databases can coexist

### Search Functionality

- Currently uses CSV files (`processed_comments.csv`) for in-memory search
- This is fast and works with both SQLite and Supabase
- Future: Can be enhanced to use Supabase full-text search

## 🐛 Troubleshooting

### App won't start
- Check logs: `fly logs`
- Verify secrets: `fly secrets list`
- Ensure Supabase credentials are correct

### Slow queries
- Verify Supabase indexes are created (run `supabase_full_schema.sql`)
- Check Supabase dashboard for query performance
- Consider upgrading Supabase plan if needed

### Migration fails
- Check you have `SUPABASE_SERVICE_KEY` set (not just anon key)
- Verify schema was created in Supabase
- Check migration script output for specific errors

## 📚 Next Steps

1. **Read MIGRATION_GUIDE.md** - Step-by-step migration
2. **Read DEPLOYMENT.md** - Complete deployment guide
3. **Run migration** - `python migrate_to_supabase.py`
4. **Deploy** - `fly deploy`

## 💡 Tips

- **Free Tier**: Both Fly.io and Supabase have generous free tiers
- **Costs**: ~$0/month for small apps, ~$25/month for larger scale
- **Performance**: Supabase is significantly faster for large datasets
- **Backup**: Supabase automatically backs up your data

## 🆘 Need Help?

- Fly.io Docs: https://fly.io/docs
- Supabase Docs: https://supabase.com/docs
- Check logs: `fly logs` or Supabase dashboard

---

**Ready to deploy?** Start with `QUICK_DEPLOY.md` for the fastest path!

