# SilenceVoice

**Unsupervised NLP Research Platform for Mental Health Stigma Analysis**

SilenceVoice is a research platform that analyzes mental health stigma through large-scale language pattern analysis of public YouTube comments. The platform processes hundreds of thousands of comments to identify naturally emerging discourse patterns without imposing predefined categories or making individual-level inferences.

## Core Principles

- **Unsupervised Analysis**: No predefined labels or categories
- **Aggregate-Level Only**: No individual classification or tracking
- **Public Data Only**: Only analyzes publicly available YouTube comments
- **Research-Oriented**: Designed for computational social science, not clinical use

## Features

- **Data Collection**: Automated collection of public YouTube comments from mental health advocacy channels
- **Text Processing**: Advanced cleaning, filtering, and semantic deduplication
- **Semantic Encoding**: State-of-the-art sentence embeddings using Sentence-BERT
- **Clustering**: HDBSCAN-based discourse pattern clustering
- **Visualization**: Interactive UMAP-based landscape visualization
- **Pattern Extraction**: Automatic keyword and pattern extraction using KeyBERT
- **Web Dashboard**: Interactive Flask-based dashboard for exploring results

## Technology Stack

- **Language**: Python 3.8+
- **NLP Framework**: sentence-transformers (Sentence-BERT)
- **Clustering**: HDBSCAN
- **Visualization**: UMAP + Plotly
- **Pattern Extraction**: KeyBERT
- **Web Framework**: Flask
- **Database**: SQLite

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd SilenceVoice

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
# Note: spaCy is optional and not currently used in the codebase
```

### 2. Configuration

Create a `.env` file in the project root:

```
YOUTUBE_API_KEY=your_youtube_api_key_here
```

Get your API key from: https://console.cloud.google.com/

### 3. Run Pipeline

```bash
# Collect data and run full pipeline
python -m src.pipeline --collect

# Or run steps individually
python -m src.pipeline --step collect
python -m src.pipeline --step process
python -m src.pipeline --step encode
python -m src.pipeline --step cluster
python -m src.pipeline --step visualize
python -m src.pipeline --step patterns
```

### 4. Start Dashboard

```bash
python -m src.dashboard.app
```

Then open http://127.0.0.1:5000 in your browser.

## Project Structure

```
SilenceVoice/
├── config.yaml              # Main configuration
├── requirements.txt         # Python dependencies
├── .env                     # Environment variables (create manually)
├── data/                    # SQLite database
├── outputs/                 # Analysis results
└── src/
    ├── data_collector.py   # YouTube data collection
    ├── text_processor.py   # Text cleaning & filtering
    ├── semantic_encoder.py # Sentence-BERT encoding
    ├── clustering.py       # HDBSCAN clustering
    ├── visualization.py    # UMAP visualization
    ├── pattern_extraction.py # Pattern analysis
    ├── pipeline.py         # Main orchestrator
    ├── dashboard/          # Web dashboard
    └── utils/              # Utility scripts
```

## Usage Examples

### Find High-Engagement Channels

```bash
python -m src.utils.find_high_engagement_channels --min-comments 50 --top-n 15
```

### Search for Channels

```bash
python -m src.utils.find_channels "mental health advocacy"
```

## Output Files

After running the pipeline, you'll find:

- `data/silencevoice.db` - SQLite database with collected comments
- `outputs/processed_comments.csv` - Cleaned and processed comments
- `outputs/embeddings.npy` - Semantic embeddings
- `outputs/cluster_results.csv` - Comments with cluster labels
- `outputs/cluster_statistics.csv` - Cluster statistics
- `outputs/umap_projection.csv` - 2D UMAP coordinates
- `outputs/stigma_landscape.html` - Interactive visualization
- `outputs/cluster_*_patterns.json` - Pattern analyses per cluster

## Ethical Considerations

⚠️ **Important**: This platform is designed for research purposes only. It:

- Only analyzes publicly available data
- Performs aggregate-level analysis only
- Does not make individual-level classifications
- Does not provide clinical diagnoses
- Does not track or identify individuals

See the dashboard for the full ethical notice.

## Documentation

- [Quick Start Guide](QUICKSTART.md)
- [MVP Description](MVP_DESCRIPTION.md)
- [Implementation Guide](MVP_IMPLEMENTATION_GUIDE.md)
- [Environment Setup](ENV_SETUP.md)

## Requirements

- Python 3.8+
- YouTube Data API v3 key
- 4GB+ RAM (8GB+ recommended for large datasets)
- Internet connection for data collection

## License

This project is for research purposes. Please ensure compliance with YouTube's Terms of Service and API usage policies.

## Support

For issues or questions, please refer to the documentation files or check the implementation guide.

## Version

**Platform Version**: 1.0.0

---

**Note**: This platform is not intended for clinical use. All analysis is performed at the aggregate level for research purposes only.

