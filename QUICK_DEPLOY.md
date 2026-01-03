# Quick Deploy Guide

## TL;DR - Get Running Fast

### 1. Supabase Setup (5 minutes)

```bash
# 1. Create project at supabase.com
# 2. Copy your URL and keys from Settings > API
# 3. Run the schema in SQL Editor (copy supabase_full_schema.sql)
# 4. Migrate your data:
python migrate_to_supabase.py
```

### 2. Fly.io Setup (5 minutes)

```bash
# 1. Install flyctl: https://fly.io/docs/hands-on/install-flyctl/
# 2. Login:
fly auth login

# 3. Deploy:
fly launch

# 4. Set secrets:
fly secrets set SUPABASE_URL=https://your-project.supabase.co
fly secrets set SUPABASE_KEY=your-anon-key
fly secrets set USE_SUPABASE=true

# 5. Deploy again:
fly deploy
```

### 3. Done! 🎉

Your app is live. Get URL with:
```bash
fly open
```

## Full Details

See `DEPLOYMENT.md` for complete instructions.

