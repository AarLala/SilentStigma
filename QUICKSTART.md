# SilenceVoice Quick Start Guide

This guide will help you get up and running with SilenceVoice in minutes.

## Prerequisites

- Python 3.8 or higher
- YouTube Data API v3 key (free tier available)
- 4GB+ RAM
- Internet connection

## Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Note: spaCy is optional and not currently used in the codebase
# If you want to install it later, you can skip it for now
```

## Step 2: Get YouTube API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select existing)
3. Enable YouTube Data API v3
4. Create credentials (API Key)
5. Copy your API key

## Step 3: Configure Environment

Create a `.env` file in the project root:

```
YOUTUBE_API_KEY=your_api_key_here
```

**Important**: Never commit your `.env` file to version control!

## Step 4: Run the Pipeline

### Option A: Full Pipeline (Recommended for First Run)

```bash
# This will collect data and run all processing steps
python -m src.pipeline --collect
```

This will:
1. Collect comments from configured channels
2. Process and clean the text
3. Generate semantic embeddings
4. Cluster discourse patterns
5. Create visualizations
6. Extract patterns

**Time**: 2-4 hours for ~100K comments (depending on your system)

### Option B: Step-by-Step

```bash
# Step 1: Collect data
python -m src.pipeline --step collect

# Step 2: Process texts
python -m src.pipeline --step process

# Step 3: Encode semantics
python -m src.pipeline --step encode

# Step 4: Cluster discourse
python -m src.pipeline --step cluster

# Step 5: Visualize
python -m src.pipeline --step visualize

# Step 6: Extract patterns
python -m src.pipeline --step patterns
```

## Step 5: Explore Results

### Start the Dashboard

```bash
python -m src.dashboard.app
```

Then open your browser to: http://127.0.0.1:5000

### View Output Files

Check the `outputs/` directory:

- `stigma_landscape.html` - Open in browser for interactive visualization
- `cluster_results.csv` - All comments with cluster assignments
- `cluster_statistics.csv` - Summary statistics per cluster
- `cluster_*_patterns.json` - Detailed pattern analyses

## Common Tasks

### Find More Channels

```bash
# Find high-engagement channels
python -m src.utils.find_high_engagement_channels --min-comments 50 --top-n 15

# Search for specific channels
python -m src.utils.find_channels "mental health advocacy"
```

### Re-run Analysis on Existing Data

If you already have data collected:

```bash
# Skip collection, just process
python -m src.pipeline --step process
python -m src.pipeline --step encode
python -m src.pipeline --step cluster
python -m src.pipeline --step visualize
python -m src.pipeline --step patterns
```

### Force Re-collection

```bash
# Force re-collection even if data exists
python -m src.pipeline --collect --force
```

## Troubleshooting

### Import Errors

```bash
# Make sure all dependencies are installed
pip install -r requirements.txt --upgrade
```

### API Rate Limits

If you hit YouTube API rate limits:
- Wait a few minutes and try again
- Reduce `max_results_per_channel` in `config.yaml`
- Increase `request_delay` in `config.yaml`

### Memory Issues

For large datasets:
- Process in smaller batches
- Reduce `max_results_per_channel` in `config.yaml`
- Close other applications

### Database Errors

```bash
# If database is corrupted, delete and re-collect
rm data/silencevoice.db
python -m src.pipeline --collect
```

## Next Steps

- Read the [README.md](README.md) for full documentation
- Check [MVP_DESCRIPTION.md](MVP_DESCRIPTION.md) for technical details
- Review [MVP_IMPLEMENTATION_GUIDE.md](MVP_IMPLEMENTATION_GUIDE.md) for implementation details

## Expected Runtime

| Dataset Size | Collection | Processing | Total |
|-------------|------------|------------|-------|
| 10K comments | 30 min | 15 min | ~45 min |
| 50K comments | 2 hours | 45 min | ~3 hours |
| 100K comments | 3-4 hours | 1.5 hours | ~5 hours |

*Times are approximate and depend on system performance and API response times*

## Tips

1. **Start Small**: Test with 1-2 channels first
2. **Monitor API Quota**: YouTube API has daily quotas
3. **Save Progress**: Each step saves intermediate results
4. **Use Dashboard**: The web interface makes exploration easier
5. **Export Data**: Use the dashboard export features to get CSV files

## Getting Help

- Check the documentation files
- Review error messages in the console
- Verify your API key is correct
- Ensure all dependencies are installed

---

**Ready to start?** Run `python -m src.pipeline --collect` and grab a coffee! ☕

