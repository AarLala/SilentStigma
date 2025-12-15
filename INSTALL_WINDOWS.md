# Windows Installation Guide

This guide helps resolve common Windows installation issues, particularly for Python 3.8.

## Issue: spaCy Build Failures

If you encounter errors building spaCy (cymem, murmurhash compilation errors), **don't worry** - spaCy is **not required** for SilenceVoice to work. It's been made optional in the requirements.

### Solution 1: Skip spaCy (Recommended)

Simply install dependencies without spaCy:

```powershell
# Install all dependencies except spaCy
pip install sentence-transformers transformers torch numpy scikit-learn pandas
pip install hdbscan umap-learn nltk keybert langdetect
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
pip install flask flask-cors plotly python-dotenv tqdm pyyaml python-dateutil sqlalchemy
```

### Solution 2: Install spaCy with Pre-built Wheels

If you want spaCy for future use:

```powershell
# Try installing pre-built wheels only
pip install spacy --only-binary :all:

# If that fails, try installing from conda-forge (if you have conda)
conda install -c conda-forge spacy
```

### Solution 3: Install Windows Build Tools

If you need to build spaCy from source:

1. Install **Microsoft C++ Build Tools**: https://visualstudio.microsoft.com/visual-cpp-build-tools/
2. Select "C++ build tools" workload
3. Install and restart terminal
4. Then try: `pip install spacy`

## Common Windows Issues

### Issue: "Cannot open include file: 'io.h'"

This means Windows SDK headers are missing. Solutions:

1. **Skip spaCy** (recommended - it's not used)
2. Install Windows SDK: https://developer.microsoft.com/en-us/windows/downloads/windows-sdk/
3. Install Visual Studio Build Tools (see Solution 3 above)

### Issue: "pip is being invoked by an old script wrapper"

Use Python's module invocation instead:

```powershell
python -m pip install -r requirements.txt
```

### Issue: Permission Errors

Run PowerShell as Administrator, or use user installation:

```powershell
pip install --user -r requirements.txt
```

## Minimal Installation (Without spaCy)

If you just want to get started quickly:

```powershell
# Core dependencies only
python -m pip install sentence-transformers transformers torch numpy scikit-learn pandas
python -m pip install hdbscan umap-learn keybert langdetect
python -m pip install google-api-python-client flask flask-cors plotly python-dotenv pyyaml tqdm sqlalchemy
```

## Verification

After installation, verify everything works:

```powershell
python verify_setup.py
```

If spaCy is missing, that's fine - the verification will show it's optional.

## Next Steps

Once dependencies are installed:

1. Create `.env` file with your YouTube API key
2. Run: `python -m src.pipeline --collect`

---

**Note**: spaCy is completely optional. The SilenceVoice platform works perfectly without it.

