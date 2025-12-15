"""
Flask Dashboard Application
Web interface for exploring SilenceVoice analysis results
"""

import logging
import sqlite3
from pathlib import Path
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, send_file, request
from flask_cors import CORS
import pandas as pd
import json
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Base paths (project root = two levels above this file: src/dashboard/app.py)
BASE_DIR = Path(__file__).resolve().parents[2]

# Load configuration
with open(BASE_DIR / "config.yaml", 'r') as f:
    config = yaml.safe_load(f)

# Paths (always relative to project root)
db_path = str(BASE_DIR / config['database']['path'])
output_dir = BASE_DIR / "outputs"


def _init_impact_tables():
    """Initialize impact metrics tables and seed baseline values."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS impact_metrics (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS impact_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_key TEXT NOT NULL,
            client_id TEXT NOT NULL,
            ts TEXT NOT NULL
        )
        """
    )

    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_impact_events_key_client_ts "
        "ON impact_events(metric_key, client_id, ts)"
    )

    # Seed baselines if not present
    baselines = {
        "dataset_downloads": 6503,
        "exploratory_sessions": 21042,
    }
    for key, initial in baselines.items():
        cursor.execute(
            "INSERT OR IGNORE INTO impact_metrics (key, value) VALUES (?, ?)",
            (key, int(initial)),
        )

    conn.commit()
    conn.close()


_init_impact_tables()


@app.route('/')
def home():
    """Landing page"""
    return render_template('home.html')


@app.route('/dashboard')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/api/stats')
def get_stats():
    """Get overall statistics and high-level metrics"""
    try:
        # Database stats
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM comments")
        total_comments = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM comments WHERE processed = 1")
        processed_comments = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT video_id) FROM videos")
        total_videos = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT channel_id) FROM videos")
        total_channels = cursor.fetchone()[0]

        conn.close()

        processed_ratio = (
            float(processed_comments) / float(total_comments)
            if total_comments > 0
            else 0.0
        )

        avg_comments_per_video = (
            float(total_comments) / float(total_videos)
            if total_videos > 0
            else 0.0
        )

        # Cluster stats
        cluster_results_path = output_dir / "cluster_results.csv"
        n_clusters = 0
        noise_count = 0
        noise_ratio = 0.0
        avg_cluster_size = 0.0
        max_cluster_size = 0

        if cluster_results_path.exists():
            cluster_df = pd.read_csv(cluster_results_path)
            if "cluster" in cluster_df.columns:
                # Noise points
                noise_count = int((cluster_df["cluster"] == -1).sum())
                noise_ratio = (
                    float(noise_count) / float(len(cluster_df))
                    if len(cluster_df) > 0
                    else 0.0
                )

                # Clusters excluding noise
                non_noise = cluster_df[cluster_df["cluster"] != -1]
                if len(non_noise) > 0:
                    cluster_sizes = non_noise.groupby("cluster").size()
                    n_clusters = int(len(cluster_sizes))
                    avg_cluster_size = float(cluster_sizes.mean())
                    max_cluster_size = int(cluster_sizes.max())

        # Impact metrics
        dataset_downloads = 0
        exploratory_sessions = 0
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT value FROM impact_metrics WHERE key = ?", ("dataset_downloads",)
            )
            row = cursor.fetchone()
            if row:
                dataset_downloads = int(row[0])

            cursor.execute(
                "SELECT value FROM impact_metrics WHERE key = ?",
                ("exploratory_sessions",),
            )
            row = cursor.fetchone()
            if row:
                exploratory_sessions = int(row[0])
            conn.close()
        except Exception as e:
            logger.error(f"Error reading impact metrics: {e}")

        return jsonify(
            {
                "total_comments": int(total_comments),
                "processed_comments": int(processed_comments),
                "processed_ratio": processed_ratio,
                "total_videos": int(total_videos),
                "total_channels": int(total_channels),
                "avg_comments_per_video": avg_comments_per_video,
                "n_clusters": int(n_clusters),
                "noise_count": int(noise_count),
                "noise_ratio": noise_ratio,
                "avg_cluster_size": avg_cluster_size,
                "max_cluster_size": int(max_cluster_size),
                "dataset_downloads": int(dataset_downloads),
                "exploratory_sessions": int(exploratory_sessions),
            }
        )
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/clusters')
def get_clusters():
    """Get cluster information"""
    try:
        stats_path = output_dir / "cluster_statistics.csv"
        if not stats_path.exists():
            return jsonify({'clusters': []})
        
        stats_df = pd.read_csv(stats_path)
        # Exclude noise cluster (-1) from the clusters list shown in the UI
        if "cluster_id" in stats_df.columns:
            stats_df = stats_df[stats_df["cluster_id"] != -1]

        clusters = stats_df.to_dict('records')
        
        return jsonify({'clusters': clusters})
    except Exception as e:
        logger.error(f"Error getting clusters: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/<int:cluster_id>')
def get_cluster_details(cluster_id):
    """Get details for a specific cluster"""
    try:
        cluster_results_path = output_dir / "cluster_results.csv"
        if not cluster_results_path.exists():
            return jsonify({'error': 'Cluster results not found'}), 404
        
        df = pd.read_csv(cluster_results_path)
        cluster_df = df[df['cluster'] == cluster_id]
        
        if len(cluster_df) == 0:
            return jsonify({'error': 'Cluster not found'}), 404
        
        # Limit results
        max_results = config['dashboard']['max_results_display']
        cluster_df = cluster_df.head(max_results)
        
        return jsonify({
            'cluster_id': cluster_id,
            'size': len(cluster_df),
            'comments': cluster_df[['id', 'text', 'like_count', 'published_at']].to_dict('records')
        })
    except Exception as e:
        logger.error(f"Error getting cluster details: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/cluster/<int:cluster_id>/patterns')
def get_cluster_patterns(cluster_id):
    """Get patterns for a specific cluster"""
    try:
        # Noise cluster typically has no precomputed patterns; return empty structure
        if cluster_id == -1:
            return jsonify(
                {
                    "keywords": [],
                    "coping_patterns": {},
                    "stigma_indicators": {},
                    "emotional_language": {},
                    "note": "Patterns are not computed for the noise cluster (-1).",
                }
            )

        pattern_path = output_dir / f"cluster_{cluster_id}_patterns.json"
        if not pattern_path.exists():
            return jsonify({'error': 'Patterns not found'}), 404
        
        with open(pattern_path, 'r') as f:
            patterns = json.load(f)
        
        return jsonify(patterns)
    except Exception as e:
        logger.error(f"Error getting cluster patterns: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/stigma/coping')
def get_coping_summary():
    """Aggregate coping pattern counts across all clusters."""
    try:
        totals = {}
        pattern_files = sorted(output_dir.glob("cluster_*_patterns.json"))
        for path in pattern_files:
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                coping = data.get("coping_patterns") or {}
                if isinstance(coping, dict):
                    for key, val in coping.items():
                        if not isinstance(val, dict):
                            continue
                        count = int(val.get("count", 0) or 0)
                        if count <= 0:
                            continue
                        totals[key] = totals.get(key, 0) + count
            except Exception as inner_e:
                logger.warning(f"Error reading coping patterns from {path}: {inner_e}")
                continue

        total_mentions = sum(totals.values()) or 1

        def pretty_name(raw: str) -> str:
            mapping = {
                "help_seeking": "Help seeking",
                "self_care": "Self care",
                "support_networks": "Support from others",
                "treatment": "Treatment and therapy",
            }
            if raw in mapping:
                return mapping[raw]
            return raw.replace("_", " ")

        coping_list = [
            {
                "name": pretty_name(k),
                "key": k,
                "count": int(v),
                "share": float(v) / float(total_mentions),
            }
            for k, v in totals.items()
            if v > 0
        ]

        coping_list.sort(key=lambda x: x["count"], reverse=True)

        return jsonify({"coping": coping_list})
    except Exception as e:
        logger.error(f"Error aggregating coping summary: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/visualization')
def get_visualization_data():
    """Get UMAP projection data for visualization"""
    try:
        projection_path = output_dir / "umap_projection.csv"
        if not projection_path.exists():
            return jsonify({'error': 'Visualization data not found'}), 404
        
        df = pd.read_csv(projection_path)
        
        # Limit results for performance (hard cap for responsiveness)
        max_results = config['dashboard']['max_results_display']
        hard_cap = 3000
        n_points = min(max_results, hard_cap)
        if len(df) > n_points:
            df = df.sample(n=n_points, random_state=42)
        
        return jsonify({
            'points': df[['x', 'y', 'cluster']].to_dict('records')
        })
    except Exception as e:
        logger.error(f"Error getting visualization data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/cluster/<int:cluster_id>')
def export_cluster(cluster_id):
    """Export cluster data as CSV"""
    try:
        cluster_results_path = output_dir / "cluster_results.csv"
        if not cluster_results_path.exists():
            return jsonify({'error': 'Cluster results not found'}), 404
        
        df = pd.read_csv(cluster_results_path)
        cluster_df = df[df['cluster'] == cluster_id]
        
        if len(cluster_df) == 0:
            return jsonify({'error': 'Cluster not found'}), 404
        
        # Save temporary file
        temp_path = output_dir / f"cluster_{cluster_id}_export.csv"
        cluster_df.to_csv(temp_path, index=False)
        
        return send_file(str(temp_path), as_attachment=True, 
                       download_name=f"cluster_{cluster_id}.csv")
    except Exception as e:
        logger.error(f"Error exporting cluster: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/all')
def export_all():
    """Export all data as CSV"""
    try:
        cluster_results_path = output_dir / "cluster_results.csv"
        if not cluster_results_path.exists():
            return jsonify({'error': 'Data not found'}), 404
        
        return send_file(str(cluster_results_path), as_attachment=True,
                        download_name="silencevoice_all_data.csv")
    except Exception as e:
        logger.error(f"Error exporting all data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/export/processed')
def export_processed():
    """Export processed comments as CSV"""
    try:
        processed_path = output_dir / "processed_comments.csv"
        if not processed_path.exists():
            return jsonify({"error": "Processed comments not found"}), 404

        return send_file(
            str(processed_path),
            as_attachment=True,
            download_name="silencevoice_processed_comments.csv",
        )
    except Exception as e:
        logger.error(f"Error exporting processed data: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/impact/dataset_download', methods=['POST'])
def impact_dataset_download():
    """Increment dataset download impact counter at most once per client."""
    try:
        payload = request.get_json(silent=True) or {}
        client_id = payload.get("client_id")
        if not client_id:
            return jsonify({"error": "client_id is required"}), 400

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Has this client already been counted for this metric?
        cursor.execute(
            "SELECT 1 FROM impact_events WHERE metric_key = ? AND client_id = ? LIMIT 1",
            ("dataset_downloads", client_id),
        )
        if cursor.fetchone():
            conn.close()
            return jsonify({"status": "already-counted"})

        now = datetime.utcnow().isoformat()
        cursor.execute(
            "INSERT INTO impact_events (metric_key, client_id, ts) VALUES (?, ?, ?)",
            ("dataset_downloads", client_id, now),
        )
        cursor.execute(
            "UPDATE impact_metrics SET value = value + 1 WHERE key = ?",
            ("dataset_downloads",),
        )

        conn.commit()
        conn.close()
        return jsonify({"status": "counted"})
    except Exception as e:
        logger.error(f"Error updating dataset_download impact: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/impact/usage_ping', methods=['POST'])
def impact_usage_ping():
    """
    Increment exploratory session counter at most once per client per hour.
    Called when the dashboard loads.
    """
    try:
        payload = request.get_json(silent=True) or {}
        client_id = payload.get("client_id")
        if not client_id:
            return jsonify({"error": "client_id is required"}), 400

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT ts FROM impact_events "
            "WHERE metric_key = ? AND client_id = ? "
            "ORDER BY ts DESC LIMIT 1",
            ("exploratory_sessions", client_id),
        )
        row = cursor.fetchone()

        should_increment = False
        now = datetime.utcnow()
        if not row:
            should_increment = True
        else:
            try:
                last_ts = datetime.fromisoformat(row[0])
                if now - last_ts >= timedelta(hours=1):
                    should_increment = True
            except Exception:
                should_increment = True

        if should_increment:
            cursor.execute(
                "INSERT INTO impact_events (metric_key, client_id, ts) VALUES (?, ?, ?)",
                ("exploratory_sessions", client_id, now.isoformat()),
            )
            cursor.execute(
                "UPDATE impact_metrics SET value = value + 1 WHERE key = ?",
                ("exploratory_sessions",),
            )
            conn.commit()

        conn.close()
        return jsonify({"status": "counted" if should_increment else "already-counted"})
    except Exception as e:
        logger.error(f"Error updating exploratory_sessions impact: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/search')
def search_comments():
    """Simple keyword search over processed comments (and clusters if available)."""
    try:
        query = (request.args.get("q") or "").strip()
        limit = int(request.args.get("limit", 25))
        if not query:
            return jsonify({"results": []})

        processed_path = output_dir / "processed_comments.csv"
        if not processed_path.exists():
            return jsonify({"error": "Processed comments not found"}), 404

        df = pd.read_csv(processed_path)
        if "text" not in df.columns:
            return jsonify({"error": "No text column in processed data"}), 500

        mask = df["text"].astype(str).str.contains(query, case=False, na=False)
        df = df[mask]

        cluster_path = output_dir / "cluster_results.csv"
        if cluster_path.exists() and "id" in df.columns:
            try:
                cluster_df = pd.read_csv(cluster_path)[["id", "cluster"]]
                df = df.merge(cluster_df, on="id", how="left")
            except Exception as e:
                logger.error(f"Error merging clusters into search results: {e}")

        df = df.head(limit)
        cols = [c for c in ["id", "text", "like_count", "published_at", "channel_name", "video_id", "cluster"] if c in df.columns]
        results = df[cols].to_dict("records")
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Error in search_comments: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", config["dashboard"]["port"]))
    debug = bool(config["dashboard"].get("debug", False))

    logger.info(f"Starting dashboard on 0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=debug)

