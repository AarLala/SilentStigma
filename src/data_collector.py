"""
YouTube Data Collector Module
Collects public comments from YouTube channels using the YouTube Data API v3
"""

import logging
import sqlite3
import time
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import yaml
import os
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class YouTubeDataCollector:
    """Collects YouTube comments from specified channels"""
    
    def __init__(self, api_key: Optional[str] = None, config_path: str = "config.yaml"):
        """
        Initialize the YouTube data collector
        
        Args:
            api_key: YouTube API key (if None, reads from YOUTUBE_API_KEY env var)
            config_path: Path to configuration file
        """
        # Load API key
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        if not self.api_key:
            raise ValueError("YouTube API key is required. Set YOUTUBE_API_KEY environment variable or pass api_key parameter.")
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize YouTube API client
        self.youtube = build('youtube', 'v3', developerKey=self.api_key)
        
        # Database configuration
        db_path = self.config['database']['path']
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        
        # YouTube API configuration
        self.max_results_per_channel = self.config['youtube']['max_results_per_channel']
        self.max_comments_per_video = self.config['youtube']['max_comments_per_video']
        self.request_delay = self.config['youtube']['request_delay']
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create comments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS comments (
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
            )
        """)
        
        # Create videos table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                channel_id TEXT,
                channel_name TEXT,
                title TEXT,
                description TEXT,
                published_at TEXT,
                comment_count INTEGER,
                collected_at TEXT
            )
        """)
        
        # Create indexes
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_video_id ON comments(video_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_channel_id ON comments(channel_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_comments_processed ON comments(processed)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_videos_channel_id ON videos(channel_id)")
        
        conn.commit()
        conn.close()
        logger.info(f"Database initialized at {self.db_path}")
    
    def get_channel_videos(self, channel_id: str, max_results: int = None) -> List[Dict]:
        """
        Fetch videos from a YouTube channel
        
        Args:
            channel_id: YouTube channel ID
            max_results: Maximum number of videos to fetch
            
        Returns:
            List of video metadata dictionaries
        """
        videos = []
        max_results = max_results or self.max_results_per_channel
        
        try:
            # Get uploads playlist ID
            channel_response = self.youtube.channels().list(
                part='contentDetails',
                id=channel_id
            ).execute()
            
            if not channel_response.get('items'):
                logger.warning(f"Channel {channel_id} not found or inaccessible")
                return videos
            
            uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
            
            # Fetch videos from uploads playlist
            next_page_token = None
            total_fetched = 0
            
            while total_fetched < max_results:
                request = self.youtube.playlistItems().list(
                    part='snippet,contentDetails',
                    playlistId=uploads_playlist_id,
                    maxResults=min(50, max_results - total_fetched),
                    pageToken=next_page_token
                )
                
                response = request.execute()
                
                for item in response.get('items', []):
                    video_id = item['contentDetails']['videoId']
                    snippet = item['snippet']
                    
                    videos.append({
                        'video_id': video_id,
                        'channel_id': channel_id,
                        'title': snippet.get('title', ''),
                        'description': snippet.get('description', ''),
                        'published_at': snippet.get('publishedAt', ''),
                    })
                    
                    total_fetched += 1
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                
                time.sleep(self.request_delay)
            
            logger.info(f"Fetched {len(videos)} videos from channel {channel_id}")
            
        except HttpError as e:
            logger.error(f"Error fetching videos from channel {channel_id}: {e}")
        
        return videos
    
    def get_video_comments(self, video_id: str, max_comments: int = None) -> List[Dict]:
        """
        Fetch comments from a YouTube video (including replies)
        
        Args:
            video_id: YouTube video ID
            max_comments: Maximum number of comments to fetch
            
        Returns:
            List of comment dictionaries
        """
        comments = []
        max_comments = max_comments or self.max_comments_per_video
        
        try:
            # Fetch top-level comments
            next_page_token = None
            total_fetched = 0
            
            while total_fetched < max_comments:
                request = self.youtube.commentThreads().list(
                    part='snippet,replies',
                    videoId=video_id,
                    maxResults=min(100, max_comments - total_fetched),
                    order='relevance',
                    pageToken=next_page_token
                )
                
                response = request.execute()
                
                for item in response.get('items', []):
                    top_level = item['snippet']['topLevelComment']['snippet']
                    
                    # Add top-level comment
                    comments.append({
                        'id': item['snippet']['topLevelComment']['id'],
                        'video_id': video_id,
                        'text': top_level.get('textDisplay', ''),
                        'author_name': top_level.get('authorDisplayName', ''),
                        'like_count': top_level.get('likeCount', 0),
                        'published_at': top_level.get('publishedAt', ''),
                    })
                    
                    total_fetched += 1
                    
                    # Add replies if available
                    if 'replies' in item:
                        for reply in item['replies']['comments']:
                            reply_snippet = reply['snippet']
                            comments.append({
                                'id': reply['id'],
                                'video_id': video_id,
                                'text': reply_snippet.get('textDisplay', ''),
                                'author_name': reply_snippet.get('authorDisplayName', ''),
                                'like_count': reply_snippet.get('likeCount', 0),
                                'published_at': reply_snippet.get('publishedAt', ''),
                            })
                            total_fetched += 1
                            
                            if total_fetched >= max_comments:
                                break
                    
                    if total_fetched >= max_comments:
                        break
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                
                time.sleep(self.request_delay)
            
        except HttpError as e:
            if e.resp.status == 403:
                logger.warning(f"Comments disabled for video {video_id}")
            else:
                logger.error(f"Error fetching comments from video {video_id}: {e}")
        
        return comments
    
    def save_comments(self, comments: List[Dict], channel_id: str, channel_name: str):
        """
        Save comments to SQLite database
        
        Args:
            comments: List of comment dictionaries
            channel_id: YouTube channel ID
            channel_name: Channel name
        """
        if not comments:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        collected_at = datetime.now().isoformat()
        
        for comment in comments:
            try:
                cursor.execute("""
                    INSERT OR IGNORE INTO comments 
                    (id, video_id, channel_id, channel_name, author_name, text, like_count, published_at, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    comment['id'],
                    comment['video_id'],
                    channel_id,
                    channel_name,
                    comment.get('author_name', ''),
                    comment.get('text', ''),
                    comment.get('like_count', 0),
                    comment.get('published_at', ''),
                    collected_at
                ))
            except sqlite3.Error as e:
                logger.error(f"Error saving comment {comment.get('id')}: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(comments)} comments from channel {channel_name}")
    
    def save_videos(self, videos: List[Dict], channel_id: str, channel_name: str):
        """
        Save video metadata to SQLite database
        
        Args:
            videos: List of video dictionaries
            channel_id: YouTube channel ID
            channel_name: Channel name
        """
        if not videos:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        collected_at = datetime.now().isoformat()
        
        for video in videos:
            try:
                # Get comment count
                try:
                    video_response = self.youtube.videos().list(
                        part='statistics',
                        id=video['video_id']
                    ).execute()
                    
                    comment_count = 0
                    if video_response.get('items'):
                        comment_count = int(video_response['items'][0]['statistics'].get('commentCount', 0))
                except:
                    comment_count = 0
                
                cursor.execute("""
                    INSERT OR REPLACE INTO videos 
                    (video_id, channel_id, channel_name, title, description, published_at, comment_count, collected_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    video['video_id'],
                    channel_id,
                    channel_name,
                    video.get('title', ''),
                    video.get('description', ''),
                    video.get('published_at', ''),
                    comment_count,
                    collected_at
                ))
            except sqlite3.Error as e:
                logger.error(f"Error saving video {video.get('video_id')}: {e}")
        
        conn.commit()
        conn.close()
        logger.info(f"Saved {len(videos)} videos from channel {channel_name}")
    
    def collect_from_channels(self):
        """Main collection orchestrator - collects comments from all configured channels"""
        channels = self.config['channels']
        total_comments = 0
        
        logger.info(f"Starting data collection from {len(channels)} channels")
        
        for channel_config in channels:
            channel_id = channel_config['id']
            channel_name = channel_config['name']
            
            logger.info(f"Collecting from channel: {channel_name} ({channel_id})")
            
            # Fetch videos
            videos = self.get_channel_videos(channel_id)
            if not videos:
                logger.warning(f"No videos found for channel {channel_name}")
                continue
            
            # Save video metadata
            self.save_videos(videos, channel_id, channel_name)
            
            # Fetch comments from each video
            for video in videos:
                video_id = video['video_id']
                comments = self.get_video_comments(video_id)
                
                if comments:
                    self.save_comments(comments, channel_id, channel_name)
                    total_comments += len(comments)
                
                time.sleep(self.request_delay)
            
            logger.info(f"Completed collection from {channel_name}")
        
        logger.info(f"Data collection complete. Total comments collected: {total_comments}")
        return total_comments


if __name__ == "__main__":
    collector = YouTubeDataCollector()
    collector.collect_from_channels()

