# SilenceVoice Dashboard

**Web Dashboard for Mental Health Stigma Analysis**

Interactive web interface for exploring mental health stigma patterns from YouTube comments.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
USE_SUPABASE=true
```

### 3. Run Dashboard

```bash
python -m src.dashboard.app
```

Visit: http://localhost:5000/dashboard

## Deployment

### Render.com

1. Connect GitHub repository to Render
2. Set environment variables in Render dashboard
3. Auto-deploys on push to main

See `render.yaml` for configuration.

## Technology Stack

- **Web**: Flask, Gunicorn
- **Database**: Supabase (PostgreSQL)
- **Data**: Pandas (CSV processing)

## Project Structure

```
src/
  dashboard/     # Web dashboard (production)
outputs/         # Processed data files (CSV, JSON)
config.yaml      # Configuration
```

## License

Research use only.
