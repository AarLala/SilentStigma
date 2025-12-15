"""
Pattern Extraction Module
Extracts patterns, keywords, and themes from clusters
"""

import logging
from typing import List, Dict
import pandas as pd
from keybert import KeyBERT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PatternExtractor:
    """Extracts patterns and themes from text clusters"""
    
    def __init__(self, config: Dict):
        """
        Initialize pattern extractor
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        keybert_config = config.get('keybert', {})
        
        # Initialize KeyBERT
        model_name = config.get('models', {}).get('sentence_transformer', 'all-MiniLM-L6-v2')
        logger.info(f"Loading KeyBERT model: {model_name}")
        self.keybert = KeyBERT(model_name)
        
        self.top_n = keybert_config.get('top_n', 10)
        self.diversity = keybert_config.get('diversity', 0.5)
        self.use_mmr = keybert_config.get('use_mmr', True)
        
        # Define pattern keywords
        self._init_pattern_keywords()
    
    def _init_pattern_keywords(self):
        """Initialize keyword lists for pattern detection"""
        # Coping patterns
        self.coping_keywords = {
            'help_seeking': ['therapy', 'counseling', 'psychologist', 'psychiatrist', 'treatment', 
                           'medication', 'meds', 'help', 'support group', 'support'],
            'self_care': ['exercise', 'meditation', 'mindfulness', 'yoga', 'breathing', 
                         'self care', 'self-care', 'routine', 'sleep', 'diet'],
            'support_networks': ['family', 'friends', 'loved ones', 'community', 'support',
                                'talking', 'sharing', 'opening up'],
            'treatment': ['therapy', 'medication', 'treatment', 'counseling', 'psychiatrist',
                         'psychologist', 'meds', 'prescription']
        }
        
        # Stigma indicators
        self.stigma_keywords = {
            'shame': ['ashamed', 'embarrassed', 'shame', 'guilty', 'weak', 'failure'],
            'fear': ['afraid', 'fear', 'scared', 'worried', 'anxiety', 'panic'],
            'dismissal': ['just', 'only', 'overreacting', 'dramatic', 'attention seeking',
                         'not real', 'fake', 'excuse'],
            'normalization': ['normal', 'everyone', 'common', 'typical', 'usual'],
            'self_blame': ['my fault', 'my problem', 'I should', 'I deserve', 'I caused']
        }
        
        # Emotional language
        self.emotion_keywords = {
            'positive': ['hope', 'better', 'improving', 'grateful', 'thankful', 'proud',
                        'progress', 'recovery', 'healing'],
            'negative': ['depressed', 'hopeless', 'suicidal', 'worthless', 'empty',
                        'numb', 'broken', 'damaged'],
            'anxiety': ['anxious', 'panic', 'worried', 'stressed', 'overwhelmed',
                       'racing thoughts', 'fear'],
            'depression': ['depressed', 'sad', 'hopeless', 'empty', 'numb', 'tired',
                          'exhausted', 'no energy']
        }
    
    def extract_keywords(self, texts: List[str], top_n: int = None) -> List[Dict]:
        """
        Extract keywords using KeyBERT
        
        Args:
            texts: List of text strings
            top_n: Number of keywords to extract
            
        Returns:
            List of keyword-score dictionaries
        """
        if not texts:
            return []
        
        top_n = top_n or self.top_n
        
        # Combine texts
        combined_text = ' '.join(texts)
        
        # Extract keywords
        if self.use_mmr:
            keywords = self.keybert.extract_keywords(
                combined_text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n,
                use_mmr=True,
                diversity=self.diversity
            )
        else:
            keywords = self.keybert.extract_keywords(
                combined_text,
                keyphrase_ngram_range=(1, 2),
                stop_words='english',
                top_n=top_n
            )
        
        # Format as list of dicts
        result = [{'keyword': kw, 'score': score} for kw, score in keywords]
        
        return result
    
    def extract_coping_patterns(self, texts: List[str]) -> Dict:
        """
        Extract coping strategy patterns
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with coping pattern counts
        """
        if not texts:
            return {}
        
        combined_text = ' '.join(texts).lower()
        
        patterns = {}
        for pattern_name, keywords in self.coping_keywords.items():
            count = sum(1 for keyword in keywords if keyword.lower() in combined_text)
            patterns[pattern_name] = {
                'count': count,
                'frequency': count / len(texts) if texts else 0
            }
        
        return patterns
    
    def extract_stigma_indicators(self, texts: List[str]) -> Dict:
        """
        Extract stigma indicator patterns
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with stigma indicator counts
        """
        if not texts:
            return {}
        
        combined_text = ' '.join(texts).lower()
        
        indicators = {}
        for indicator_name, keywords in self.stigma_keywords.items():
            count = sum(1 for keyword in keywords if keyword.lower() in combined_text)
            indicators[indicator_name] = {
                'count': count,
                'frequency': count / len(texts) if texts else 0
            }
        
        return indicators
    
    def extract_emotional_language(self, texts: List[str]) -> Dict:
        """
        Extract emotional language patterns
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with emotion pattern counts
        """
        if not texts:
            return {}
        
        combined_text = ' '.join(texts).lower()
        
        emotions = {}
        for emotion_name, keywords in self.emotion_keywords.items():
            count = sum(1 for keyword in keywords if keyword.lower() in combined_text)
            emotions[emotion_name] = {
                'count': count,
                'frequency': count / len(texts) if texts else 0
            }
        
        return emotions
    
    def analyze_cluster(self, cluster_texts: List[str]) -> Dict:
        """
        Perform comprehensive analysis of a cluster
        
        Args:
            cluster_texts: List of texts in the cluster
            
        Returns:
            Dictionary with complete analysis
        """
        if not cluster_texts:
            return {}
        
        logger.info(f"Analyzing cluster with {len(cluster_texts)} texts...")
        
        analysis = {
            'size': len(cluster_texts),
            'keywords': self.extract_keywords(cluster_texts),
            'coping_patterns': self.extract_coping_patterns(cluster_texts),
            'stigma_indicators': self.extract_stigma_indicators(cluster_texts),
            'emotional_language': self.extract_emotional_language(cluster_texts)
        }
        
        return analysis
    
    def analyze_all_clusters(self, df: pd.DataFrame, cluster_column: str = 'cluster') -> pd.DataFrame:
        """
        Analyze all clusters in a DataFrame
        
        Args:
            df: DataFrame with text and cluster columns
            cluster_column: Name of cluster column
            
        Returns:
            DataFrame with cluster analyses
        """
        if cluster_column not in df.columns:
            raise ValueError(f"Column '{cluster_column}' not found in DataFrame")
        
        if 'text' not in df.columns:
            raise ValueError("Column 'text' not found in DataFrame")
        
        analyses = []
        unique_clusters = sorted([c for c in df[cluster_column].unique() if c != -1])
        
        logger.info(f"Analyzing {len(unique_clusters)} clusters...")
        
        for cluster_id in unique_clusters:
            cluster_df = df[df[cluster_column] == cluster_id]
            cluster_texts = cluster_df['text'].tolist()
            
            analysis = self.analyze_cluster(cluster_texts)
            analysis['cluster_id'] = cluster_id
            
            analyses.append(analysis)
        
        # Convert to DataFrame
        analyses_df = pd.DataFrame(analyses)
        
        return analyses_df

