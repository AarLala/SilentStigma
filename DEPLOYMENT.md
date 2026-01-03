# Deployment Guide for SilenceVoice on Fly.io

This guide will help you deploy SilenceVoice to Fly.io and migrate your database to Supabase for faster access.

## Prerequisites

1. **Fly.io Account**: Sign up at [fly.io](https://fly.io)
2. **Supabase Account**: Sign up at [supabase.com](https://supabase.com)
3. **Fly CLI**: Install from [fly.io/docs/hands-on/install-flyctl](https://fly.io/docs/hands-on/install-flyctl/)

## Step 1: Set Up Supabase

### 1.1 Create a Supabase Project

1. Go to [supabase.com](https://supabase.com) and create a new project
2. Wait for the project to be fully provisioned (takes a few minutes)
3. Go to **Settings > API** and note:
   - **Project URL** (e.g., `https://xxxxx.supabase.co`)
   - **anon public key** (for client-side access)
   - **service_role key** (for server-side operations - keep this secret!)

### 1.2 Create Database Schema

1. In your Supabase dashboard, go to **SQL Editor**
2. Copy and paste the contents of `supabase_full_schema.sql`
3. Click **Run** to execute the SQL
4. Verify tables were created by checking **Table Editor**

The schema includes:
- `comments` table (main data)
- `videos` table (video metadata)
- `metrics` table (usage tracking)
- `download_events` table (download tracking)
- `session_events` table (session tracking)
- All necessary indexes for fast queries

### 1.3 Migrate Data from SQLite to Supabase

1. Make sure you have a `.env` file with your Supabase credentials:
   ```bash
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-anon-public-key
   SUPABASE_SERVICE_KEY=your-service-role-key
   ```

2. Run the migration script:
   ```bash
   python migrate_to_supabase.py
   ```

3. The script will:
   - Check your SQLite database
   - Migrate all comments and videos to Supabase
   - Show progress and summary

**Note**: This may take a while if you have a large dataset. The script uses batch inserts for efficiency.

## Step 2: Deploy to Fly.io

### 2.1 Initialize Fly.io App

1. Login to Fly.io:
   ```bash
   fly auth login
   ```

2. Initialize your app (from project root):
   ```bash
   fly launch
   ```

   When prompted:
   - **App name**: Choose a unique name (or leave blank for auto-generated)
   - **Region**: Choose closest to your users (e.g., `iad` for US East)
   - **Postgres?**: No (we're using Supabase)
   - **Redis?**: No (optional, for caching)

3. The `fly.toml` file should already be configured. If you need to adjust settings, edit it.

### 2.2 Set Environment Variables

Set your Supabase credentials as secrets in Fly.io:

```bash
fly secrets set SUPABASE_URL=https://your-project.supabase.co
fly secrets set SUPABASE_KEY=your-anon-public-key
fly secrets set USE_SUPABASE=true
```

**Important**: Never commit your `.env` file or secrets to git!

### 2.3 Deploy

Deploy your application:

```bash
fly deploy
```

The first deployment may take a few minutes as it builds the Docker image.

### 2.4 Verify Deployment

1. Check app status:
   ```bash
   fly status
   ```

2. View logs:
   ```bash
   fly logs
   ```

3. Open your app:
   ```bash
   fly open
   ```

## Step 3: Post-Deployment

### 3.1 Verify Database Connection

1. Check logs to ensure Supabase connection is working:
   ```bash
   fly logs | grep -i supabase
   ```

2. You should see: `"Supabase client initialized successfully"` and `"Using Supabase for data queries"`

### 3.2 Test the Application

1. Visit your app URL (shown by `fly open`)
2. Test the dashboard:
   - Check that stats load correctly
   - Try a search query
   - Verify clusters are displayed
   - Test export functionality

### 3.3 Monitor Performance

1. Check Fly.io metrics:
   ```bash
   fly metrics
   ```

2. Monitor Supabase usage in your Supabase dashboard

## Configuration Options

### Scaling

To scale your app:

```bash
# Scale to 2 instances
fly scale count 2

# Scale memory
fly scale vm shared-cpu-2x --memory 4096
```

### Environment Variables

You can update secrets anytime:

```bash
fly secrets set KEY=value
```

View current secrets:

```bash
fly secrets list
```

### Custom Domain

1. Add a domain in Fly.io dashboard
2. Update DNS records as instructed
3. Fly.io will automatically provision SSL certificates

## Troubleshooting

### App Won't Start

1. Check logs:
   ```bash
   fly logs
   ```

2. Common issues:
   - Missing environment variables → Set secrets
   - Database connection errors → Verify Supabase credentials
   - Port binding issues → Ensure PORT=8080 is set

### Database Connection Issues

1. Verify Supabase credentials are correct
2. Check Supabase project is active (not paused)
3. Ensure RLS policies allow reads (see `supabase_full_schema.sql`)
4. Check Supabase logs in dashboard

### Performance Issues

1. **Slow queries**: Check Supabase indexes are created
2. **High memory usage**: Scale up memory in `fly.toml`
3. **Rate limiting**: Supabase free tier has limits; consider upgrading

### Rollback

If something goes wrong:

```bash
# View deployment history
fly releases

# Rollback to previous version
fly releases rollback
```

## Cost Optimization

### Fly.io

- Free tier: 3 shared-cpu-1x VMs with 256MB RAM
- Paid: ~$1.94/month per 1GB RAM VM
- Use `auto_stop_machines = true` in `fly.toml` to save costs

### Supabase

- Free tier: 500MB database, 2GB bandwidth
- Paid: Starts at $25/month for more resources
- Monitor usage in Supabase dashboard

## Security Best Practices

1. **Never commit secrets**: Use `fly secrets` for sensitive data
2. **Use service role key only server-side**: Never expose in client code
3. **Enable RLS**: Row Level Security is enabled in the schema
4. **Rate limiting**: Already configured in the app
5. **HTTPS**: Fly.io provides free SSL certificates

## Next Steps

- Set up monitoring (e.g., Sentry for error tracking)
- Configure backups for Supabase
- Set up CI/CD for automatic deployments
- Add custom domain
- Optimize database queries based on usage patterns

## Support

- Fly.io Docs: https://fly.io/docs
- Supabase Docs: https://supabase.com/docs
- Project Issues: Check GitHub issues

