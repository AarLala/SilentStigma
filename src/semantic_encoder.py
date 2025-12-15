"""
Semantic Encoder Module
Encodes text into semantic embeddings using Sentence-BERT
"""

import logging
from typing import List, Optional
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SemanticEncoder:
    """Encodes texts into semantic embeddings using Sentence-BERT"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[str] = None):
        """
        Initialize semantic encoder
        
        Args:
            model_name: Name of Sentence-BERT model
            cache_dir: Directory to cache model files
        """
        self.model_name = model_name
        logger.info(f"Loading Sentence-BERT model: {model_name}")
        
        self.model = SentenceTransformer(model_name, cache_folder=cache_dir)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")
    
    def encode_texts(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Encode texts into semantic embeddings
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings (n_samples, embedding_dim)
        """
        if not texts:
            return np.array([])
        
        logger.info(f"Encoding {len(texts)} texts...")
        
        # Encode with normalization for cosine similarity
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        logger.info(f"Encoding complete. Shape: {embeddings.shape}")
        
        return embeddings
    
    def encode_dataframe(self, df: pd.DataFrame, text_column: str = 'text', batch_size: int = 32) -> pd.DataFrame:
        """
        Encode text column in DataFrame
        
        Args:
            df: DataFrame with text column
            text_column: Name of text column
            batch_size: Batch size for encoding
            
        Returns:
            DataFrame with added 'embedding' column (as list)
        """
        if text_column not in df.columns:
            raise ValueError(f"Column '{text_column}' not found in DataFrame")
        
        texts = df[text_column].tolist()
        embeddings = self.encode_texts(texts, batch_size=batch_size)
        
        # Convert to list of arrays for DataFrame storage
        df = df.copy()
        df['embedding'] = [emb for emb in embeddings]
        
        return df
    
    def save_embeddings(self, embeddings: np.ndarray, filepath: str):
        """
        Save embeddings to disk
        
        Args:
            embeddings: Numpy array of embeddings
            filepath: Path to save file (.npy format)
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        np.save(filepath, embeddings)
        logger.info(f"Saved embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> np.ndarray:
        """
        Load embeddings from disk
        
        Args:
            filepath: Path to embeddings file
            
        Returns:
            Numpy array of embeddings
        """
        embeddings = np.load(filepath)
        logger.info(f"Loaded embeddings from {filepath}. Shape: {embeddings.shape}")
        return embeddings

