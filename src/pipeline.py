"""
Pipeline Orchestrator
Main pipeline that coordinates all processing steps
"""

import logging
import argparse
import sqlite3
from typing import Optional, Tuple
from pathlib import Path
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data_collector import YouTubeDataCollector
from src.text_processor import TextProcessor
from src.semantic_encoder import SemanticEncoder
from src.clustering import DiscourseClusterer
from src.visualization import StigmaLandscapeVisualizer
from src.pattern_extraction import PatternExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SilenceVoicePipeline:
    """Main pipeline orchestrator for SilenceVoice platform"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize pipeline
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Create output directory
        self.output_dir = Path("outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize components
        logger.info("Initializing pipeline components...")
        self.text_processor = TextProcessor(self.config)
        self.encoder = SemanticEncoder(
            model_name=self.config['models']['sentence_transformer']
        )
        self.clusterer = DiscourseClusterer(self.config)
        self.visualizer = StigmaLandscapeVisualizer(self.config)
        self.pattern_extractor = PatternExtractor(self.config)
        
        # Database path
        self.db_path = self.config['database']['path']
        
        logger.info("Pipeline initialized")
    
    def collect_data(self, force: bool = False):
        """
        Collect data from YouTube channels
        
        Args:
            force: Force collection even if data exists
        """
        # Initialize collector first to ensure database is created
        collector = YouTubeDataCollector(config_path="config.yaml")
        
        # Check if data exists
        if not force:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            try:
                cursor.execute("SELECT COUNT(*) FROM comments")
                count = cursor.fetchone()[0]
                conn.close()
                
                if count > 0:
                    logger.info(f"Database already contains {count} comments. Use --force to re-collect.")
                    return
            except sqlite3.OperationalError:
                # Table doesn't exist yet, which is fine - we'll create it
                conn.close()
        
        logger.info("Starting data collection...")
        collector.collect_from_channels()
        logger.info("Data collection complete")
    
    def process_texts(self) -> pd.DataFrame:
        """
        Process and clean texts from database
        
        Returns:
            Processed DataFrame
        """
        # Check if processed comments CSV already exists
        processed_csv_path = self.output_dir / "processed_comments.csv"
        if processed_csv_path.exists():
            logger.info(f"Found existing processed comments file: {processed_csv_path}")
            logger.info("Loading processed comments from CSV...")
            df = pd.read_csv(processed_csv_path)
            logger.info(f"Loaded {len(df)} processed comments from CSV")
            return df
        
        logger.info("Loading comments from database...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load unprocessed comments
        df = pd.read_sql_query(
            "SELECT * FROM comments WHERE processed = 0",
            conn
        )
        
        conn.close()
        
        if len(df) == 0:
            logger.info("No unprocessed comments found. Loading all comments...")
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query("SELECT * FROM comments", conn)
            conn.close()
        
        if len(df) == 0:
            logger.warning("No comments found in database")
            return pd.DataFrame()
        
        logger.info(f"Processing {len(df)} comments...")
        
        # Process comments
        comments_list = df.to_dict('records')
        processed_df = self.text_processor.process_batch(comments_list)
        
        if len(processed_df) == 0:
            logger.warning("No comments passed processing filters")
            return pd.DataFrame()
        
        # Update database processed flags
        logger.info("Updating database processed flags...")
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for comment_id in processed_df['id'].values:
            cursor.execute("UPDATE comments SET processed = 1 WHERE id = ?", (comment_id,))
        
        conn.commit()
        conn.close()
        
        # Save processed data
        output_path = self.output_dir / "processed_comments.csv"
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed comments to {output_path}")
        
        return processed_df
    
    def encode_semantics(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Encode texts into semantic embeddings
        
        Args:
            df: DataFrame with processed texts
            
        Returns:
            Tuple of (DataFrame, embeddings array)
        """
        if len(df) == 0:
            logger.warning("No data to encode")
            return df, np.array([])
        
        # Check for existing embeddings
        embedding_path = self.output_dir / "embeddings.npy"
        if embedding_path.exists():
            logger.info("Loading existing embeddings...")
            embeddings = self.encoder.load_embeddings(str(embedding_path))
            
            # Ensure alignment
            if len(embeddings) == len(df):
                logger.info("Using existing embeddings")
                return df, embeddings
        
        logger.info("Encoding texts to embeddings...")
        texts = df['text'].tolist()
        embeddings = self.encoder.encode_texts(texts, batch_size=32)
        
        # Save embeddings
        self.encoder.save_embeddings(embeddings, str(embedding_path))
        
        return df, embeddings
    
    def cluster_discourse(self, embeddings: np.ndarray, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Cluster discourse patterns
        
        Args:
            embeddings: Semantic embeddings
            df: DataFrame with texts
            
        Returns:
            Tuple of (cluster labels, cluster statistics DataFrame)
        """
        if len(embeddings) == 0:
            logger.warning("No embeddings to cluster")
            return np.array([]), pd.DataFrame()
        
        logger.info("Clustering discourse patterns...")
        labels = self.clusterer.fit(embeddings)
        
        # Add labels to DataFrame
        df = df.copy()
        df['cluster'] = labels
        
        # Get cluster statistics
        stats_df = self.clusterer.get_cluster_statistics(labels, df)
        
        # Save cluster results
        cluster_results_path = self.output_dir / "cluster_results.csv"
        df.to_csv(cluster_results_path, index=False)
        logger.info(f"Saved cluster results to {cluster_results_path}")
        
        # Save cluster statistics
        stats_path = self.output_dir / "cluster_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"Saved cluster statistics to {stats_path}")
        
        return labels, stats_df
    
    def visualize_landscape(self, embeddings: np.ndarray, labels: np.ndarray, df: pd.DataFrame):
        """
        Create visualization of discourse landscape
        
        Args:
            embeddings: Semantic embeddings
            labels: Cluster labels
            df: DataFrame with texts
        """
        if len(embeddings) == 0:
            logger.warning("No data to visualize")
            return
        
        logger.info("Creating UMAP projection...")
        projected = self.visualizer.fit_transform(embeddings)
        
        # Save projection
        projection_path = self.output_dir / "umap_projection.csv"
        projection_df = pd.DataFrame({
            'x': projected[:, 0],
            'y': projected[:, 1],
            'cluster': labels
        })
        projection_df.to_csv(projection_path, index=False)
        logger.info(f"Saved UMAP projection to {projection_path}")
        
        # Create interactive visualization
        logger.info("Creating interactive visualization...")
        output_path = self.output_dir / "stigma_landscape.html"
        self.visualizer.create_interactive_plot(projected, labels, df, str(output_path))
        logger.info(f"Saved visualization to {output_path}")
    
    def extract_patterns(self, df: pd.DataFrame, labels: np.ndarray):
        """
        Extract patterns from clusters
        
        Args:
            df: DataFrame with texts
            labels: Cluster labels
        """
        if len(df) == 0:
            logger.warning("No data to analyze")
            return
        
        logger.info("Extracting patterns from clusters...")
        
        df = df.copy()
        df['cluster'] = labels
        
        # Analyze all clusters
        analyses_df = self.pattern_extractor.analyze_all_clusters(df, cluster_column='cluster')
        
        # Save analyses
        analyses_path = self.output_dir / "cluster_analyses.csv"
        analyses_df.to_csv(analyses_path, index=False)
        logger.info(f"Saved cluster analyses to {analyses_path}")
        
        # Save individual cluster pattern files
        unique_clusters = sorted([c for c in df['cluster'].unique() if c != -1])
        
        for cluster_id in unique_clusters:
            cluster_df = df[df['cluster'] == cluster_id]
            cluster_texts = cluster_df['text'].tolist()
            
            analysis = self.pattern_extractor.analyze_cluster(cluster_texts)
            
            # Save as JSON
            import json
            pattern_path = self.output_dir / f"cluster_{cluster_id}_patterns.json"
            with open(pattern_path, 'w') as f:
                json.dump(analysis, f, indent=2)
        
        logger.info(f"Saved pattern files for {len(unique_clusters)} clusters")
    
    def run_full_pipeline(self, collect: bool = False, force_collect: bool = False):
        """
        Run the complete pipeline
        
        Args:
            collect: Whether to collect data first
            force_collect: Force data collection even if data exists
        """
        logger.info("=" * 60)
        logger.info("Starting SilenceVoice Pipeline")
        logger.info("=" * 60)
        
        # Step 1: Collect data (optional)
        if collect:
            self.collect_data(force=force_collect)
        
        # Step 2: Process texts
        processed_df = self.process_texts()
        if len(processed_df) == 0:
            logger.error("No processed comments available. Exiting.")
            return
        
        # Step 3: Encode semantics
        df, embeddings = self.encode_semantics(processed_df)
        
        # Step 4: Cluster discourse
        labels, stats_df = self.cluster_discourse(embeddings, df)
        
        # Step 5: Visualize landscape
        self.visualize_landscape(embeddings, labels, df)
        
        # Step 6: Extract patterns
        self.extract_patterns(df, labels)
        
        logger.info("=" * 60)
        logger.info("Pipeline Complete!")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {self.output_dir}")
    
    def run_step(self, step: str, collect: bool = False, force_collect: bool = False):
        """
        Run a specific pipeline step
        
        Args:
            step: Step name (collect, process, encode, cluster, visualize, patterns)
            collect: Whether to collect data first (for process step)
            force_collect: Force data collection
        """
        if step == "collect":
            self.collect_data(force=force_collect)
        
        elif step == "process":
            if collect:
                self.collect_data(force=force_collect)
            self.process_texts()
        
        elif step == "encode":
            processed_df = self.process_texts()
            self.encode_semantics(processed_df)
        
        elif step == "cluster":
            processed_df = self.process_texts()
            df, embeddings = self.encode_semantics(processed_df)
            self.cluster_discourse(embeddings, df)
        
        elif step == "visualize":
            processed_df = self.process_texts()
            df, embeddings = self.encode_semantics(processed_df)
            labels, _ = self.cluster_discourse(embeddings, df)
            self.visualize_landscape(embeddings, labels, df)
        
        elif step == "patterns":
            processed_df = self.process_texts()
            df, embeddings = self.encode_semantics(processed_df)
            labels, _ = self.cluster_discourse(embeddings, df)
            self.extract_patterns(df, labels)
        
        else:
            logger.error(f"Unknown step: {step}")
            logger.info("Available steps: collect, process, encode, cluster, visualize, patterns")


def main():
    """Main entry point for pipeline"""
    parser = argparse.ArgumentParser(description="SilenceVoice Pipeline")
    parser.add_argument("--step", type=str, choices=['all', 'collect', 'process', 'encode', 
                                                      'cluster', 'visualize', 'patterns'],
                       default='all', help="Pipeline step to run")
    parser.add_argument("--collect", action='store_true', 
                       help="Collect data before processing")
    parser.add_argument("--force", action='store_true',
                       help="Force data collection even if data exists")
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to configuration file")
    
    args = parser.parse_args()
    
    pipeline = SilenceVoicePipeline(config_path=args.config)
    
    if args.step == 'all':
        pipeline.run_full_pipeline(collect=args.collect, force_collect=args.force)
    else:
        pipeline.run_step(args.step, collect=args.collect, force_collect=args.force)


if __name__ == "__main__":
    main()

