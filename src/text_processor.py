"""
Text Processing Module
Cleans, filters, and deduplicates YouTube comments
"""

import logging
import re
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set seed for langdetect reproducibility
DetectorFactory.seed = 0


class TextProcessor:
    """Processes and cleans text data for analysis"""
    
    def __init__(self, config: Dict):
        """
        Initialize text processor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.processing_config = config.get('processing', {})
        
        # Initialize sentence transformer for semantic deduplication
        model_name = config.get('models', {}).get('sentence_transformer', 'all-MiniLM-L6-v2')
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.sentence_model = SentenceTransformer(model_name)
        
        # Processing parameters
        self.min_length = self.processing_config.get('min_comment_length', 10)
        self.max_length = self.processing_config.get('max_comment_length', 1000)
        self.supported_languages = self.processing_config.get('supported_languages', ['en'])
        self.remove_urls = self.processing_config.get('remove_urls', True)
        self.remove_emojis = self.processing_config.get('remove_emojis', False)
        self.dedup_threshold = self.processing_config.get('deduplication_threshold', 0.95)
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing URLs, normalizing whitespace, etc.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs if configured
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove emojis if configured
        if self.remove_emojis:
            emoji_pattern = re.compile(
                "["
                "\U0001F600-\U0001F64F"  # emoticons
                "\U0001F300-\U0001F5FF"  # symbols & pictographs
                "\U0001F680-\U0001F6FF"  # transport & map symbols
                "\U0001F1E0-\U0001F1FF"  # flags
                "\U00002702-\U000027B0"
                "\U000024C2-\U0001F251"
                "]+", flags=re.UNICODE
            )
            text = emoji_pattern.sub('', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
    
    def detect_language(self, text: str) -> str:
        """
        Detect language of text
        
        Args:
            text: Text string
            
        Returns:
            Language code (ISO 639-1) or 'unknown'
        """
        try:
            if len(text) < 3:
                return 'unknown'
            return detect(text)
        except Exception as e:
            logger.debug(f"Language detection failed: {e}")
            return 'unknown'
    
    def filter_by_language(self, texts: List[str]) -> np.ndarray:
        """
        Filter texts by supported languages
        
        Args:
            texts: List of text strings
            
        Returns:
            Boolean array indicating which texts are in supported languages
        """
        flags = []
        for text in texts:
            lang = self.detect_language(text)
            flags.append(lang in self.supported_languages)
        return np.array(flags)
    
    def filter_by_length(self, texts: List[str]) -> np.ndarray:
        """
        Filter texts by length constraints
        
        Args:
            texts: List of text strings
            
        Returns:
            Boolean array indicating which texts meet length requirements
        """
        flags = []
        for text in texts:
            length = len(text)
            flags.append(self.min_length <= length <= self.max_length)
        return np.array(flags)
    
    def detect_spam(self, texts: List[str], threshold: float = 0.3) -> np.ndarray:
        """
        Detect spam comments using heuristics
        
        Args:
            texts: List of text strings
            threshold: Threshold for spam detection (0-1)
            
        Returns:
            Boolean array indicating which texts are likely spam
        """
        spam_flags = []
        
        for text in texts:
            if not text:
                spam_flags.append(True)
                continue
            
            # Check excessive capitalization
            if len(text) > 0:
                caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
                if caps_ratio > 0.7 and len(text) > 20:
                    spam_flags.append(True)
                    continue
            
            # Check excessive punctuation
            punct_count = sum(1 for c in text if c in '!?.')
            if len(text) > 0 and punct_count / len(text) > 0.2:
                spam_flags.append(True)
                continue
            
            # Check repetitive characters
            if len(text) > 10:
                import itertools
                max_repeat = max(len(list(g)) for _, g in itertools.groupby(text))
                if max_repeat > 5:
                    spam_flags.append(True)
                    continue
            
            spam_flags.append(False)
        
        return np.array(spam_flags)
    
    def deduplicate_semantic(self, texts: List[str], threshold: float = None, batch_size: int = 1000) -> np.ndarray:
        """
        Deduplicate texts using semantic similarity (memory-efficient batch processing)
        
        Args:
            texts: List of text strings
            threshold: Similarity threshold for duplicates (default from config)
            batch_size: Batch size for processing (default 1000 to avoid memory issues)
            
        Returns:
            Boolean array indicating which texts to keep (True) or remove (False)
        """
        if not texts:
            return np.array([])
        
        threshold = threshold or self.dedup_threshold
        
        # For very large datasets, skip semantic deduplication or use smaller batches
        if len(texts) > 50000:
            logger.warning(f"Large dataset ({len(texts)} texts). Using batch-based deduplication with batch_size={batch_size}")
            batch_size = min(batch_size, 500)  # Smaller batches for very large datasets
        
        logger.info(f"Computing semantic embeddings for {len(texts)} texts...")
        embeddings = self.sentence_model.encode(texts, show_progress_bar=True, batch_size=32)
        
        logger.info(f"Computing similarity in batches (batch_size={batch_size})...")
        
        # Mark duplicates: keep first occurrence, mark others as duplicates
        keep_flags = np.ones(len(texts), dtype=bool)
        n_batches = (len(texts) + batch_size - 1) // batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(texts))
            
            if start_idx >= len(texts):
                break
            
            # Compute similarity for this batch against all remaining texts
            batch_embeddings = embeddings[start_idx:end_idx]
            
            # Only compare with texts that haven't been marked as duplicates yet
            remaining_indices = np.where(keep_flags)[0]
            if len(remaining_indices) == 0:
                break
            
            # Compute similarity between batch and remaining texts
            batch_similarities = cosine_similarity(
                batch_embeddings,
                embeddings[remaining_indices]
            )
            
            # Process each item in the batch
            for i, global_idx in enumerate(range(start_idx, end_idx)):
                if not keep_flags[global_idx]:
                    continue
                
                # Find position of current index in remaining_indices
                pos_in_remaining = np.where(remaining_indices == global_idx)[0]
                if len(pos_in_remaining) == 0:
                    continue
                pos = pos_in_remaining[0]
                
                # Find similar texts (only those after current index)
                similar_mask = (batch_similarities[i] >= threshold)
                similar_positions = np.where(similar_mask)[0]
                
                # Filter to only include texts after current index
                for similar_pos in similar_positions:
                    similar_global_idx = remaining_indices[similar_pos]
                    if similar_global_idx > global_idx and keep_flags[similar_global_idx]:
                        keep_flags[similar_global_idx] = False
        
        duplicates_removed = (~keep_flags).sum()
        logger.info(f"Removed {duplicates_removed} semantic duplicates")
        
        return keep_flags
    
    def process_batch(self, comments: List[Dict]) -> pd.DataFrame:
        """
        Process a batch of comments through the full pipeline
        
        Args:
            comments: List of comment dictionaries with 'text' field
            
        Returns:
            Processed DataFrame with cleaned and filtered comments
        """
        if not comments:
            return pd.DataFrame()
        
        logger.info(f"Processing {len(comments)} comments...")
        
        # Convert to DataFrame
        df = pd.DataFrame(comments)
        
        if 'text' not in df.columns:
            logger.error("Comments must have 'text' field")
            return pd.DataFrame()
        
        # Clean texts
        logger.info("Cleaning texts...")
        df['text_cleaned'] = df['text'].apply(self.clean_text)
        
        # Filter by length
        logger.info("Filtering by length...")
        length_mask = self.filter_by_length(df['text_cleaned'].tolist())
        df = df[length_mask].reset_index(drop=True)
        logger.info(f"After length filtering: {len(df)} comments")
        
        if len(df) == 0:
            return df
        
        # Filter by language
        logger.info("Filtering by language...")
        lang_mask = self.filter_by_language(df['text_cleaned'].tolist())
        df = df[lang_mask].reset_index(drop=True)
        logger.info(f"After language filtering: {len(df)} comments")
        
        if len(df) == 0:
            return df
        
        # Detect spam
        logger.info("Detecting spam...")
        spam_mask = ~self.detect_spam(df['text_cleaned'].tolist())
        df = df[spam_mask].reset_index(drop=True)
        logger.info(f"After spam filtering: {len(df)} comments")
        
        if len(df) == 0:
            return df
        
        # Semantic deduplication (skip if dataset is too large to avoid memory issues)
        skip_dedup = self.processing_config.get('skip_semantic_dedup_if_large', True)
        if skip_dedup and len(df) > 50000:
            logger.warning(f"Large dataset ({len(df)} texts). Skipping semantic deduplication to avoid memory issues.")
            logger.info("Consider running deduplication separately on smaller batches if needed.")
        else:
            logger.info("Deduplicating semantically...")
            dedup_mask = self.deduplicate_semantic(df['text_cleaned'].tolist())
            df = df[dedup_mask].reset_index(drop=True)
            logger.info(f"After deduplication: {len(df)} comments")
        
        # Use cleaned text as final text
        df['text'] = df['text_cleaned']
        df = df.drop(columns=['text_cleaned'], errors='ignore')
        
        # Add processing metadata
        df['processed_at'] = pd.Timestamp.now().isoformat()
        
        logger.info(f"Processing complete. Final count: {len(df)} comments")
        
        return df

