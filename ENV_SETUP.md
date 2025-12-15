# Environment Setup Guide

This guide provides detailed instructions for setting up your SilenceVoice development environment.

## System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB+ recommended
- **Disk Space**: 2GB+ for models and data
- **Internet**: Required for data collection and model downloads

## Step 1: Python Installation

### Check Python Version

```bash
python --version
# Should be 3.8 or higher
```

### Install Python (if needed)

- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **macOS**: `brew install python3` or download from python.org
- **Linux**: `sudo apt-get install python3 python3-pip` (Ubuntu/Debian)

## Step 2: Virtual Environment

### Create Virtual Environment

```bash
# Navigate to project directory
cd SilenceVoice

# Create virtual environment
python -m venv venv
```

### Activate Virtual Environment

**Windows (PowerShell)**:
```powershell
venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**:
```cmd
venv\Scripts\activate.bat
```

**macOS/Linux**:
```bash
source venv/bin/activate
```

You should see `(venv)` in your prompt when activated.

### Deactivate (when done)

```bash
deactivate
```

## Step 3: Install Dependencies

### Install from requirements.txt

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Install spaCy Language Model (Optional)

**Note**: spaCy is not currently used in the codebase, so this step is optional. If you want to install it for future use:

```bash
# For Windows with Python 3.8, use pre-built wheel:
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Or install spaCy with pre-built wheels (if available):
pip install spacy --only-binary :all:
python -m spacy download en_core_web_sm
```

### Verify Installation

```bash
python -c "import sentence_transformers; import hdbscan; import umap; import keybert; print('All dependencies installed!')"
```

## Step 4: YouTube API Setup

### Get API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project:
   - Click "Select a project" → "New Project"
   - Enter project name (e.g., "SilenceVoice")
   - Click "Create"
3. Enable YouTube Data API v3:
   - Go to "APIs & Services" → "Library"
   - Search for "YouTube Data API v3"
   - Click "Enable"
4. Create API Key:
   - Go to "APIs & Services" → "Credentials"
   - Click "Create Credentials" → "API Key"
   - Copy your API key
   - (Optional) Restrict key to YouTube Data API v3

### Set API Key

Create a `.env` file in the project root:

```bash
# .env
YOUTUBE_API_KEY=your_api_key_here
```

**Important**: 
- Never commit `.env` to version control
- The `.env` file is already in `.gitignore`

### Verify API Key

```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print('API Key:', 'SET' if os.getenv('YOUTUBE_API_KEY') else 'NOT SET')"
```

## Step 5: Project Structure

Ensure your project structure looks like this:

```
SilenceVoice/
├── .env                    # Your API key (create this)
├── config.yaml             # Configuration file
├── requirements.txt        # Dependencies
├── README.md
├── data/                   # Created automatically
├── outputs/                # Created automatically
└── src/
    ├── __init__.py
    ├── data_collector.py
    ├── text_processor.py
    ├── semantic_encoder.py
    ├── clustering.py
    ├── visualization.py
    ├── pattern_extraction.py
    ├── pipeline.py
    ├── dashboard/
    │   ├── __init__.py
    │   ├── app.py
    │   └── templates/
    │       └── index.html
    └── utils/
        ├── __init__.py
        ├── find_channels.py
        └── find_high_engagement_channels.py
```

## Step 6: Test Installation

### Quick Test

```bash
# Test imports
python -c "from src.data_collector import YouTubeDataCollector; print('✓ Data collector OK')"
python -c "from src.text_processor import TextProcessor; print('✓ Text processor OK')"
python -c "from src.semantic_encoder import SemanticEncoder; print('✓ Encoder OK')"
python -c "from src.clustering import DiscourseClusterer; print('✓ Clusterer OK')"
python -c "from src.visualization import StigmaLandscapeVisualizer; print('✓ Visualizer OK')"
python -c "from src.pattern_extraction import PatternExtractor; print('✓ Pattern extractor OK')"
python -c "from src.pipeline import SilenceVoicePipeline; print('✓ Pipeline OK')"
```

### Test Dashboard

```bash
python -m src.dashboard.app
```

Then visit http://127.0.0.1:5000 (should show an empty dashboard if no data yet)

## Common Issues

### Issue: "ModuleNotFoundError"

**Solution**: 
```bash
# Make sure virtual environment is activated
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Issue: "spacy model not found"

**Solution**:
```bash
python -m spacy download en_core_web_sm
```

### Issue: "YOUTUBE_API_KEY not found"

**Solution**:
1. Check `.env` file exists in project root
2. Verify API key is correct (no quotes, no spaces)
3. Restart terminal/IDE after creating `.env`

### Issue: "Permission denied" (Windows)

**Solution**:
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Issue: "Out of memory"

**Solution**:
- Reduce dataset size in `config.yaml`
- Close other applications
- Process in smaller batches

## Development Setup (Optional)

### Install Development Tools

```bash
pip install pytest black flake8 mypy
```

### Code Formatting

```bash
black src/
```

### Type Checking

```bash
mypy src/
```

## Environment Variables Reference

| Variable | Required | Description |
|----------|----------|-------------|
| `YOUTUBE_API_KEY` | Yes | YouTube Data API v3 key |

## Next Steps

Once your environment is set up:

1. Read [QUICKSTART.md](QUICKSTART.md) to run your first analysis
2. Review [config.yaml](config.yaml) to customize settings
3. Check [README.md](README.md) for full documentation

## Verification Checklist

- [ ] Python 3.8+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] spaCy model downloaded (`python -m spacy download en_core_web_sm`)
- [ ] YouTube API key obtained
- [ ] `.env` file created with API key
- [ ] Project structure verified
- [ ] Test imports successful
- [ ] Dashboard starts without errors

---

**Setup complete?** Proceed to [QUICKSTART.md](QUICKSTART.md) to run your first analysis!

