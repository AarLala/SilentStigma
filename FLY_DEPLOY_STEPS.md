# Fly.io Deployment - Step-by-Step Guide

## Prerequisites Check

First, let's check if you have everything installed:

```powershell
# Check if flyctl is installed
fly version

# If you get an error, you need to install it (see Step 1)
```

---

## Step 1: Install Fly CLI (if needed)

### Option A: Using PowerShell (Recommended for Windows)

```powershell
# Download and install flyctl
iwr https://fly.io/install.ps1 -useb | iex
```

### Option B: Using Scoop (if you have it)

```powershell
scoop install flyctl
```

### Option C: Manual Download

1. Go to: https://fly.io/docs/hands-on/install-flyctl/
2. Download the Windows installer
3. Run the installer

**Verify installation:**
```powershell
fly version
```

You should see something like: `flyctl v0.x.x windows/amd64`

---

## Step 2: Login to Fly.io

```powershell
fly auth login
```

This will:
- Open your browser
- Ask you to sign up/login to Fly.io
- Create an account if you don't have one (free tier available)
- Authenticate your CLI

**Expected output:**
```
Opening https://fly.io/app/auth/cli/...
Waiting for session transfer... complete
Successfully logged in as your-email@example.com
```

---

## Step 3: Initialize Your App

From your project root directory:

```powershell
fly launch
```

**When prompted, answer:**

1. **App name**: Press Enter for auto-generated name, OR type a custom name (e.g., `silencevoice`)
   - ⚠️ App names must be globally unique on Fly.io
   - If taken, try: `silencevoice-yourname` or `silencevoice-2025`

2. **Region**: Choose closest to you (e.g., `iad` for US East, `lhr` for London)
   - Type the region code and press Enter

3. **Postgres?**: Type `n` and press Enter (we're using Supabase)

4. **Redis?**: Type `n` and press Enter (optional, not needed)

5. **Deploy now?**: Type `n` and press Enter (we'll deploy after setting secrets)

**Expected output:**
```
Creating app in [region]...
...
Your app is ready! Deploy with: fly deploy
```

This creates a `fly.toml` file (we already have one, so it may ask to overwrite - say `n` to keep ours).

---

## Step 4: Set Environment Variables (Secrets)

Set your Supabase credentials as secrets:

```powershell
# Set Supabase URL
fly secrets set SUPABASE_URL=https://qfhmfipzynbaslxpychv.supabase.co

# Set Supabase anon key
fly secrets set SUPABASE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InFmaG1maXB6eW5iYXNseHB5Y2h2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjU5MjY3NTQsImV4cCI6MjA4MTUwMjc1NH0.UreclGzrUd2XmvwYODMM3XjixA_YcdpBpiR96DN1Q88

# Enable Supabase usage
fly secrets set USE_SUPABASE=true
```

**Verify secrets are set:**
```powershell
fly secrets list
```

You should see all three secrets listed.

---

## Step 5: Review fly.toml (Optional)

Check that `fly.toml` looks correct:

```powershell
Get-Content fly.toml
```

It should have:
- `app = "your-app-name"`
- `primary_region = "your-region"`
- `internal_port = 8080`
- Memory and CPU settings

---

## Step 6: Deploy Your App

```powershell
fly deploy
```

**What happens:**
1. Builds Docker image (this takes 5-10 minutes first time)
2. Uploads to Fly.io
3. Deploys to your region
4. Starts your app

**Expected output:**
```
==> Building image
...
==> Creating release
...
==> Monitoring deployment
...
✓ Deployment successful!
```

**First deployment takes longer** (5-10 minutes) because it needs to:
- Download base Python image
- Install all dependencies
- Build the Docker image
- Upload everything

---

## Step 7: Verify Deployment

### Check App Status

```powershell
fly status
```

Should show:
```
App
  Name     = your-app-name
  Owner    = your-email
  Hostname = your-app-name.fly.dev
  Status   = running
```

### View Logs

```powershell
fly logs
```

Look for:
```
INFO: Supabase client initialized successfully
INFO: Using Supabase for data queries (faster than SQLite)
```

### Open Your App

```powershell
fly open
```

This opens your app in the browser at: `https://your-app-name.fly.dev`

Or manually visit: `https://your-app-name.fly.dev/dashboard`

---

## Step 8: Test Your Deployed App

1. **Visit the dashboard**: `https://your-app-name.fly.dev/dashboard`
2. **Check stats load** - should show your 166K+ comments
3. **Try a search** - test functionality
4. **Check logs** if anything fails:
   ```powershell
   fly logs
   ```

---

## Troubleshooting

### "App name already taken"

**Fix**: Choose a different name:
```powershell
fly launch --name silencevoice-unique-name
```

### "Deployment failed"

**Check logs:**
```powershell
fly logs
```

**Common issues:**
- Missing secrets → Run `fly secrets set` commands again
- Build errors → Check Dockerfile is correct
- Port issues → Verify `fly.toml` has `internal_port = 8080`

### "Can't connect to Supabase"

**Verify secrets:**
```powershell
fly secrets list
```

**Check Supabase project is active:**
- Go to Supabase dashboard
- Make sure project isn't paused

### "App won't start"

**SSH into the container:**
```powershell
fly ssh console
```

**Check environment:**
```powershell
env | grep SUPABASE
```

### View Detailed Logs

```powershell
# Real-time logs
fly logs --follow

# Last 100 lines
fly logs -n 100
```

---

## Useful Commands

```powershell
# View app info
fly info

# Scale app (if needed)
fly scale count 2

# Restart app
fly apps restart your-app-name

# View metrics
fly metrics

# SSH into container
fly ssh console

# View secrets (values hidden)
fly secrets list

# Update a secret
fly secrets set KEY=new-value

# Remove a secret
fly secrets unset KEY
```

---

## Next Steps After Deployment

1. **Set up custom domain** (optional):
   - In Fly.io dashboard → Your App → Domains
   - Add your domain
   - Update DNS records as instructed

2. **Monitor usage**:
   - Check `fly metrics` for resource usage
   - Monitor Supabase dashboard for database usage

3. **Set up backups** (optional):
   - Supabase automatically backs up your database
   - Consider setting up Fly.io volume backups if needed

---

## Cost Estimate

**Free Tier:**
- 3 shared-cpu-1x VMs with 256MB RAM
- 160GB outbound data transfer
- Usually enough for small-medium apps

**If you need more:**
- ~$1.94/month per 1GB RAM VM
- Check `fly.toml` for current VM size

---

## Quick Reference

```powershell
# 1. Install (if needed)
iwr https://fly.io/install.ps1 -useb | iex

# 2. Login
fly auth login

# 3. Launch app
fly launch

# 4. Set secrets
fly secrets set SUPABASE_URL=https://your-project.supabase.co
fly secrets set SUPABASE_KEY=your-anon-key
fly secrets set USE_SUPABASE=true

# 5. Deploy
fly deploy

# 6. Open app
fly open
```

---

**That's it! Your app should now be live on Fly.io! 🚀**

