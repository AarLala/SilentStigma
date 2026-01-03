# ✅ Ready for GitHub + Fly.io Auto-Deployment

## Files Configured

✅ **Dockerfile** - Production-ready container configuration  
✅ **.dockerignore** - Optimized build context  
✅ **fly.toml** - Fly.io app configuration  
✅ **.gitignore** - Excludes .env and sensitive files  
✅ **.github/workflows/fly.yml** - GitHub Actions workflow (optional)  
✅ **requirements.txt** - Includes gunicorn for production  

## Quick Deployment Steps

### 1. Push to GitHub

```powershell
git add .
git commit -m "Configure for Fly.io deployment"
git push origin main
```

### 2. Connect GitHub to Fly.io

**Option A: Fly.io Dashboard (Easiest)**
1. Go to https://fly.io/dashboard
2. Create/select your app
3. Go to **Settings** → **GitHub Integration**
4. Click **Connect GitHub**
5. Select your repository
6. Enable **Auto Deploy**

**Option B: GitHub Actions**
1. Go to GitHub repo → **Settings** → **Secrets** → **Actions**
2. Add secret: `FLY_API_TOKEN` (get from Fly.io → Account Settings → Access Tokens)
3. Push code - auto-deploys via GitHub Actions

### 3. Set Secrets in Fly.io

```powershell
fly secrets set SUPABASE_URL=https://qfhmfipzynbaslxpychv.supabase.co
fly secrets set SUPABASE_KEY=your-anon-key
fly secrets set USE_SUPABASE=true
```

Or use Fly.io dashboard → Your App → Secrets

### 4. Deploy!

**If using Fly.io GitHub Integration:**
- Just push to GitHub - auto-deploys!

**If using GitHub Actions:**
- Push to GitHub - workflow deploys automatically

**Manual (if needed):**
```powershell
fly deploy
```

## What's Configured

- **Port**: 8080 (set in fly.toml)
- **Server**: Gunicorn (production-ready)
- **Workers**: 2 workers, 2 threads each
- **Memory**: 2048MB
- **Auto-start/stop**: Enabled (saves costs)
- **HTTPS**: Forced (automatic)

## Important Notes

- ✅ `.env` is excluded from Git (never commit secrets!)
- ✅ Dockerfile uses production server (gunicorn)
- ✅ All dependencies in requirements.txt
- ✅ NLTK data downloaded during build
- ✅ Outputs directory created automatically

## Troubleshooting

**Build fails?**
- Check `fly logs`
- Verify all files are committed to GitHub

**App won't start?**
- Check secrets: `fly secrets list`
- Check logs: `fly logs`

**Auto-deploy not working?**
- Verify GitHub integration in Fly.io dashboard
- Check GitHub Actions tab (if using workflows)

## View Your App

```powershell
fly open
```

Or visit: `https://your-app-name.fly.dev`

---

**Every push to main/master will auto-deploy! 🚀**

