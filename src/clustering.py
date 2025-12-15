"""
Clustering Module
Clusters semantic embeddings using HDBSCAN
"""

import logging
from typing import Optional, Tuple, Dict
import numpy as np
import pandas as pd
import hdbscan
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DiscourseClusterer:
    """Clusters discourse patterns using HDBSCAN"""
    
    def __init__(self, config: Dict):
        """
        Initialize clusterer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        clustering_config = config.get('clustering', {})
        
        self.min_cluster_size = clustering_config.get('min_cluster_size', 50)
        self.min_samples = clustering_config.get('min_samples', 10)
        self.cluster_selection_epsilon = clustering_config.get('cluster_selection_epsilon', 0.0)
        self.metric = clustering_config.get('metric', 'euclidean')
        
        logger.info(f"Initialized HDBSCAN clusterer: min_cluster_size={self.min_cluster_size}, "
                   f"min_samples={self.min_samples}, metric={self.metric}")
    
    def reduce_dimensionality(self, embeddings: np.ndarray, n_components: int = 100) -> np.ndarray:
        """
        Optionally reduce dimensionality using PCA
        
        Args:
            embeddings: Input embeddings
            n_components: Number of components to keep
            
        Returns:
            Reduced embeddings
        """
        if embeddings.shape[1] <= n_components:
            return embeddings
        
        from sklearn.decomposition import PCA
        
        logger.info(f"Reducing dimensionality from {embeddings.shape[1]} to {n_components}")
        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(embeddings)
        
        logger.info(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")
        
        return reduced
    
    def fit(self, embeddings: np.ndarray, reduce_dim: bool = False) -> np.ndarray:
        """
        Fit HDBSCAN clusterer and return cluster labels
        
        Args:
            embeddings: Input embeddings
            reduce_dim: Whether to reduce dimensionality first
            
        Returns:
            Array of cluster labels (-1 indicates noise/outliers)
        """
        if len(embeddings) == 0:
            return np.array([])
        
        # Optionally reduce dimensionality
        if reduce_dim:
            embeddings = self.reduce_dimensionality(embeddings)
        
        # Standardize if using euclidean metric
        if self.metric == 'euclidean':
            scaler = StandardScaler()
            embeddings = scaler.fit_transform(embeddings)
        
        logger.info(f"Fitting HDBSCAN on {len(embeddings)} embeddings...")
        
        # Create and fit clusterer
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            core_dist_n_jobs=-1
        )
        
        labels = clusterer.fit_predict(embeddings)
        
        # Log cluster statistics
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        
        logger.info(f"Clustering complete: {n_clusters} clusters found, {n_noise} noise points")
        
        if n_clusters > 0:
            cluster_sizes = pd.Series(labels).value_counts().sort_index()
            logger.info(f"Cluster sizes: {dict(cluster_sizes)}")
        
        return labels
    
    def get_cluster_statistics(self, labels: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute statistics for each cluster
        
        Args:
            labels: Cluster labels
            df: DataFrame with text data
            
        Returns:
            DataFrame with cluster statistics
        """
        df = df.copy()
        df['cluster'] = labels
        
        stats = []
        
        for cluster_id in sorted(set(labels)):
            if cluster_id == -1:
                cluster_name = "Noise"
            else:
                cluster_name = f"Cluster {cluster_id}"
            
            cluster_df = df[df['cluster'] == cluster_id]
            
            stats.append({
                'cluster_id': cluster_id,
                'cluster_name': cluster_name,
                'size': len(cluster_df),
                'avg_text_length': cluster_df['text'].str.len().mean() if 'text' in cluster_df.columns else 0,
            })
        
        stats_df = pd.DataFrame(stats)
        return stats_df
    
    def get_cluster_members(self, cluster_id: int, labels: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
        """
        Get all members of a specific cluster
        
        Args:
            cluster_id: Cluster ID to retrieve
            labels: Cluster labels
            df: DataFrame with data
            
        Returns:
            DataFrame with cluster members
        """
        df = df.copy()
        df['cluster'] = labels
        return df[df['cluster'] == cluster_id].reset_index(drop=True)

