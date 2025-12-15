"""
Visualization Module
Creates interactive visualizations using UMAP and Plotly
"""

import logging
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import umap
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StigmaLandscapeVisualizer:
    """Creates visualizations of the stigma discourse landscape"""
    
    def __init__(self, config: Dict):
        """
        Initialize visualizer
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        umap_config = config.get('umap', {})
        
        self.n_neighbors = umap_config.get('n_neighbors', 15)
        self.n_components = umap_config.get('n_components', 2)
        self.min_dist = umap_config.get('min_dist', 0.1)
        self.metric = umap_config.get('metric', 'cosine')
        self.random_state = umap_config.get('random_state', 42)
        
        logger.info(f"Initialized UMAP visualizer: n_neighbors={self.n_neighbors}, "
                   f"min_dist={self.min_dist}, metric={self.metric}")
    
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Create 2D UMAP projection of embeddings
        
        Args:
            embeddings: Input embeddings
            
        Returns:
            2D projected coordinates
        """
        if len(embeddings) == 0:
            return np.array([])
        
        logger.info(f"Creating UMAP projection for {len(embeddings)} embeddings...")
        
        reducer = umap.UMAP(
            n_neighbors=self.n_neighbors,
            n_components=self.n_components,
            min_dist=self.min_dist,
            metric=self.metric,
            random_state=self.random_state
        )
        
        projected = reducer.fit_transform(embeddings)
        
        logger.info(f"UMAP projection complete. Shape: {projected.shape}")
        
        return projected
    
    def create_interactive_plot(self, projected: np.ndarray, labels: np.ndarray, 
                                df: pd.DataFrame, output_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive Plotly visualization
        
        Args:
            projected: 2D projected coordinates
            labels: Cluster labels
            df: DataFrame with text data
            output_path: Optional path to save HTML
            
        Returns:
            Plotly figure
        """
        if len(projected) == 0:
            logger.warning("No data to visualize")
            return go.Figure()
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'x': projected[:, 0],
            'y': projected[:, 1],
            'cluster': labels,
            'text': df['text'].values if 'text' in df.columns else [''] * len(projected),
        })
        
        # Add cluster names
        plot_df['cluster_name'] = plot_df['cluster'].apply(
            lambda x: f"Cluster {x}" if x != -1 else "Noise"
        )
        
        # Create figure
        fig = go.Figure()
        
        # Get unique clusters
        unique_clusters = sorted(set(labels))
        
        # Color palette
        colors = px.colors.qualitative.Set3
        
        for i, cluster_id in enumerate(unique_clusters):
            cluster_data = plot_df[plot_df['cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            color = colors[i % len(colors)]
            name = f"Cluster {cluster_id}" if cluster_id != -1 else "Noise"
            
            # Create hover text
            hover_text = cluster_data['text'].apply(
                lambda x: (x[:100] + '...') if len(x) > 100 else x
            )
            
            fig.add_trace(go.Scatter(
                x=cluster_data['x'],
                y=cluster_data['y'],
                mode='markers',
                name=name,
                text=hover_text,
                hovertemplate='<b>%{fullData.name}</b><br>%{text}<extra></extra>',
                marker=dict(
                    size=5,
                    color=color,
                    opacity=0.7,
                    line=dict(width=0.5, color='white')
                )
            ))
        
        # Update layout
        fig.update_layout(
            title='Mental Health Stigma Discourse Landscape',
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            hovermode='closest',
            width=1200,
            height=800,
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # Save if path provided
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Saved visualization to {output_path}")
        
        return fig
    
    def create_cluster_comparison_plot(self, projected: np.ndarray, labels: np.ndarray,
                                      df: pd.DataFrame, cluster_ids: List[int],
                                      output_path: Optional[str] = None) -> go.Figure:
        """
        Create comparison plot highlighting specific clusters
        
        Args:
            projected: 2D projected coordinates
            labels: Cluster labels
            df: DataFrame with text data
            cluster_ids: List of cluster IDs to highlight
            output_path: Optional path to save HTML
            
        Returns:
            Plotly figure
        """
        if len(projected) == 0:
            logger.warning("No data to visualize")
            return go.Figure()
        
        # Create DataFrame
        plot_df = pd.DataFrame({
            'x': projected[:, 0],
            'y': projected[:, 1],
            'cluster': labels,
            'text': df['text'].values if 'text' in df.columns else [''] * len(projected),
        })
        
        # Mark highlighted clusters
        plot_df['highlighted'] = plot_df['cluster'].isin(cluster_ids)
        
        fig = go.Figure()
        
        # Add non-highlighted points (gray)
        non_highlighted = plot_df[~plot_df['highlighted']]
        if len(non_highlighted) > 0:
            fig.add_trace(go.Scatter(
                x=non_highlighted['x'],
                y=non_highlighted['y'],
                mode='markers',
                name='Other',
                marker=dict(
                    size=3,
                    color='lightgray',
                    opacity=0.3
                ),
                showlegend=False
            ))
        
        # Add highlighted clusters
        colors = px.colors.qualitative.Set3
        for i, cluster_id in enumerate(cluster_ids):
            cluster_data = plot_df[plot_df['cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            color = colors[i % len(colors)]
            
            hover_text = cluster_data['text'].apply(
                lambda x: (x[:100] + '...') if len(x) > 100 else x
            )
            
            fig.add_trace(go.Scatter(
                x=cluster_data['x'],
                y=cluster_data['y'],
                mode='markers',
                name=f'Cluster {cluster_id}',
                text=hover_text,
                hovertemplate='<b>Cluster %{fullData.name}</b><br>%{text}<extra></extra>',
                marker=dict(
                    size=6,
                    color=color,
                    opacity=0.8,
                    line=dict(width=1, color='black')
                )
            ))
        
        # Update layout
        fig.update_layout(
            title=f'Cluster Comparison: {", ".join([f"Cluster {c}" for c in cluster_ids])}',
            xaxis_title='UMAP Dimension 1',
            yaxis_title='UMAP Dimension 2',
            hovermode='closest',
            width=1200,
            height=800,
            template='plotly_white'
        )
        
        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            fig.write_html(output_path)
            logger.info(f"Saved comparison plot to {output_path}")
        
        return fig

