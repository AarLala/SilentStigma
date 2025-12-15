# SilenceVoice MVP - Technical Description

## Overview

SilenceVoice is an unsupervised NLP research platform designed to analyze mental health stigma through large-scale language pattern analysis of public YouTube comments. The platform identifies naturally emerging discourse patterns without imposing predefined categories or making individual-level inferences.

## Architecture

### Pipeline Flow

```
YouTube Comments (Public)
    ↓
[Data Collection] → SQLite Database
    ↓
[Text Processing] → Cleaned DataFrame
    ↓
[Semantic Encoding] → Embeddings (n×384)
    ↓
[Clustering (HDBSCAN)] → Cluster Labels
    ↓
[UMAP Projection] → 2D Coordinates
    ↓
[Pattern Extraction] → Cluster Analyses
    ↓
[Dashboard] → Interactive Visualization
```

### Core Components

#### 1. Data Collection (`data_collector.py`)

- **Purpose**: Collect public YouTube comments from mental health advocacy channels
- **Technology**: YouTube Data API v3
- **Storage**: SQLite database
- **Features**:
  - Channel-based collection
  - Comment thread extraction (including replies)
  - Rate limiting and error handling
  - Video metadata tracking

#### 2. Text Processing (`text_processor.py`)

- **Purpose**: Clean, filter, and deduplicate comments
- **Features**:
  - Language detection and filtering (English by default)
  - Text normalization (URL removal, whitespace)
  - Spam detection (heuristic-based)
  - Semantic deduplication using sentence embeddings
  - Length and quality filtering

#### 3. Semantic Encoding (`semantic_encoder.py`)

- **Purpose**: Convert text to semantic embeddings
- **Model**: `all-MiniLM-L6-v2` (Sentence-BERT)
- **Output**: 384-dimensional normalized embeddings
- **Features**:
  - Batch processing for efficiency
  - Embedding caching
  - Normalized for cosine similarity

#### 4. Clustering (`clustering.py`)

- **Purpose**: Identify discourse pattern clusters
- **Algorithm**: HDBSCAN (Hierarchical Density-Based Spatial Clustering)
- **Features**:
  - Variable density cluster detection
  - Noise/outlier identification
  - Cluster statistics computation
  - Optional PCA dimensionality reduction

#### 5. Visualization (`visualization.py`)

- **Purpose**: Create interactive visualizations
- **Technology**: UMAP + Plotly
- **Features**:
  - 2D UMAP projection
  - Interactive Plotly visualizations
  - Cluster highlighting
  - HTML export

#### 6. Pattern Extraction (`pattern_extraction.py`)

- **Purpose**: Extract patterns and themes from clusters
- **Technology**: KeyBERT
- **Patterns**:
  - Keywords/keyphrases
  - Coping strategies (help-seeking, self-care, support networks)
  - Stigma indicators (shame, fear, dismissal, normalization)
  - Emotional language (positive, negative, anxiety, depression)

#### 7. Pipeline Orchestrator (`pipeline.py`)

- **Purpose**: Coordinate all processing steps
- **Features**:
  - Step-by-step execution
  - CLI interface
  - Error handling and logging
  - Intermediate result saving

#### 8. Dashboard (`dashboard/`)

- **Purpose**: Web interface for exploring results
- **Technology**: Flask + Plotly.js
- **Features**:
  - Statistics overview
  - Interactive UMAP visualization
  - Cluster exploration
  - Pattern analysis
  - CSV export

## Data Flow

### Input
- Public YouTube comments from mental health advocacy channels
- Channel IDs configured in `config.yaml`

### Processing
1. **Collection**: Comments stored in SQLite database
2. **Cleaning**: Text normalization, filtering, deduplication
3. **Encoding**: Semantic embeddings generated
4. **Clustering**: Discourse patterns identified
5. **Projection**: 2D visualization coordinates
6. **Analysis**: Patterns extracted per cluster

### Output
- SQLite database with raw and processed comments
- CSV files with processed data and cluster assignments
- Embeddings file (`.npy`)
- Interactive HTML visualizations
- JSON files with pattern analyses

## Technical Specifications

### Models

- **Sentence-BERT**: `all-MiniLM-L6-v2`
  - Embedding dimension: 384
  - Normalized for cosine similarity
  - Fast inference (~1000 texts/second)

### Clustering

- **Algorithm**: HDBSCAN
- **Parameters**:
  - `min_cluster_size`: 50
  - `min_samples`: 10
  - `metric`: euclidean (or cosine)

### Visualization

- **Algorithm**: UMAP
- **Parameters**:
  - `n_neighbors`: 15
  - `n_components`: 2
  - `min_dist`: 0.1
  - `metric`: cosine

### Performance

- **100K comments**: ~3-4 hours total
- **Memory**: ~2-4GB RAM
- **Storage**: ~500MB for 100K comments

## Ethical Safeguards

### Code-Level
1. **No Individual Tracking**: Only aggregate statistics
2. **Public Data Only**: Verifies data source is public
3. **No Diagnosis**: Explicit warnings in code and UI
4. **Aggregate Analysis**: All analysis at cluster level only

### UI Safeguards
- Ethical notice on dashboard
- Clear disclaimers about research use only
- No individual comment attribution in exports

## Configuration

All settings in `config.yaml`:

- **YouTube API**: Rate limits, max results
- **Channels**: List of channel IDs to analyze
- **Processing**: Text filtering parameters
- **Models**: Model names and settings
- **Clustering**: HDBSCAN parameters
- **UMAP**: Visualization parameters
- **Dashboard**: Server settings

## Database Schema

### Comments Table
```sql
CREATE TABLE comments (
    id TEXT PRIMARY KEY,
    video_id TEXT,
    channel_id TEXT,
    channel_name TEXT,
    author_name TEXT,
    text TEXT,
    like_count INTEGER,
    published_at TEXT,
    collected_at TEXT,
    processed BOOLEAN DEFAULT 0
);
```

### Videos Table
```sql
CREATE TABLE videos (
    video_id TEXT PRIMARY KEY,
    channel_id TEXT,
    channel_name TEXT,
    title TEXT,
    description TEXT,
    published_at TEXT,
    comment_count INTEGER,
    collected_at TEXT
);
```

## API Endpoints

### Dashboard API

- `GET /` - Main dashboard page
- `GET /api/stats` - Overall statistics
- `GET /api/clusters` - Cluster information
- `GET /api/cluster/<id>` - Cluster details
- `GET /api/cluster/<id>/patterns` - Cluster patterns
- `GET /api/export/cluster/<id>` - Export cluster CSV
- `GET /api/export/all` - Export all data CSV
- `GET /api/visualization` - UMAP projection data

## File Structure

```
outputs/
├── processed_comments.csv      # Cleaned comments
├── embeddings.npy              # Semantic embeddings
├── cluster_results.csv         # Comments with cluster labels
├── cluster_statistics.csv      # Cluster statistics
├── umap_projection.csv         # 2D coordinates
├── stigma_landscape.html       # Interactive visualization
└── cluster_*_patterns.json    # Pattern analyses
```

## Limitations

1. **Language**: English only (configurable)
2. **Data Source**: YouTube comments only
3. **Scale**: Optimized for 10K-500K comments
4. **Real-time**: Batch processing, not real-time
5. **Clinical Use**: Not intended for clinical applications

## Future Enhancements

- Multi-language support
- Additional data sources
- Real-time processing
- Advanced pattern detection
- Comparative analysis across time periods
- Integration with other social media platforms

## References

- Sentence-BERT: https://www.sbert.net/
- HDBSCAN: https://hdbscan.readthedocs.io/
- UMAP: https://umap-learn.readthedocs.io/
- KeyBERT: https://maartengr.github.io/KeyBERT/
- YouTube Data API: https://developers.google.com/youtube/v3

---

**Version**: 1.0.0  
**Last Updated**: 2024

