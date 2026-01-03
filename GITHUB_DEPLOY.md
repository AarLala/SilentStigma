# GitHub + Fly.io Auto-Deployment Setup

## Quick Setup Steps

### 1. Push Your Code to GitHub

```powershell
# Initialize git if not already done
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - ready for Fly.io deployment"

# Add your GitHub repository (replace with your repo URL)
git remote add origin https://github.com/yourusername/silencevoice.git

# Push to GitHub
git push -u origin main
```

### 2. Connect GitHub to Fly.io

**Option A: Using Fly.io Dashboard (Recommended)**

1. Go to https://fly.io/dashboard
2. Click on your app (or create one if needed)
3. Go to **Settings** → **GitHub Integration**
4. Click **Connect GitHub**
5. Authorize Fly.io to access your repositories
6. Select your repository
7. Choose branch (usually `main` or `master`)
8. Enable **Auto Deploy** - deployments will trigger on every push

**Option B: Using GitHub Actions (Alternative)**

1. Go to your GitHub repository
2. Go to **Settings** → **Secrets and variables** → **Actions**
3. Add a new secret:
   - Name: `FLY_API_TOKEN`
   - Value: Get from Fly.io → Account Settings → Access Tokens → Create Token
4. Push your code - the workflow in `.github/workflows/fly.yml` will auto-deploy

### 3. Set Secrets in Fly.io

After connecting, set your environment variables:

```powershell
fly secrets set SUPABASE_URL=https://qfhmfipzynbaslxpychv.supabase.co
fly secrets set SUPABASE_KEY=your-anon-key-here
fly secrets set USE_SUPABASE=true
```

Or use Fly.io dashboard:
- Go to your app → **Secrets**
- Add each secret

### 4. First Deployment

**If using Fly.io GitHub Integration:**
- Just push to GitHub - it will auto-deploy!

**If using GitHub Actions:**
- Push to GitHub - the workflow will deploy automatically

**Manual deployment (if needed):**
```powershell
fly deploy
```

## What Happens on Each Push

1. Code is pushed to GitHub
2. Fly.io detects the push (or GitHub Actions triggers)
3. Builds Docker image from `Dockerfile`
4. Deploys to Fly.io
5. App restarts with new code

## Important Files for Deployment

✅ **Dockerfile** - Defines how to build your app  
✅ **fly.toml** - Fly.io configuration  
✅ **.dockerignore** - Excludes files from Docker build  
✅ **.gitignore** - Excludes sensitive files from Git  
✅ **requirements.txt** - Python dependencies  
✅ **.github/workflows/fly.yml** - GitHub Actions workflow (optional)

## Secrets Management

**Never commit these to GitHub:**
- `.env` file
- API keys
- Service role keys

**Set them in Fly.io:**
```powershell
fly secrets set KEY=value
```

## Troubleshooting

### Auto-deploy not working?

1. Check Fly.io dashboard → Your App → Deployments
2. Check GitHub repository → Actions (if using GitHub Actions)
3. Verify secrets are set: `fly secrets list`

### Build fails?

1. Check logs: `fly logs`
2. Test Dockerfile locally:
   ```powershell
   docker build -t test-app .
   docker run -p 8080:8080 test-app
   ```

### App won't start?

1. Check logs: `fly logs`
2. Verify secrets: `fly secrets list`
3. Test locally first: `python -m src.dashboard.app`

## Manual Deployment (if auto-deploy fails)

```powershell
fly deploy
```

## View Your App

```powershell
fly open
```

Or visit: `https://your-app-name.fly.dev`

---

**That's it! Every push to main/master will auto-deploy! 🚀**

