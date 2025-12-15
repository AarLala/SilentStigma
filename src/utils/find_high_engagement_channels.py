"""
Utility script to find YouTube channels with high comment engagement
"""

import argparse
import os
import sys
import yaml
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from collections import defaultdict

load_dotenv()


def get_channel_videos(youtube, channel_id, max_videos=50):
    """Get videos from a channel"""
    try:
        # Get uploads playlist
        channel_response = youtube.channels().list(
            part='contentDetails',
            id=channel_id
        ).execute()
        
        if not channel_response.get('items'):
            return []
        
        uploads_playlist_id = channel_response['items'][0]['contentDetails']['relatedPlaylists']['uploads']
        
        # Get videos
        videos = []
        next_page_token = None
        
        while len(videos) < max_videos:
            request = youtube.playlistItems().list(
                part='contentDetails',
                playlistId=uploads_playlist_id,
                maxResults=min(50, max_videos - len(videos)),
                pageToken=next_page_token
            )
            
            response = request.execute()
            
            for item in response.get('items', []):
                videos.append(item['contentDetails']['videoId'])
            
            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        
        return videos
    except HttpError as e:
        print(f"Error getting videos for channel {channel_id}: {e}")
        return []


def get_video_comment_count(youtube, video_id):
    """Get comment count for a video"""
    try:
        response = youtube.videos().list(
            part='statistics',
            id=video_id
        ).execute()
        
        if response.get('items'):
            return int(response['items'][0]['statistics'].get('commentCount', 0))
        return 0
    except HttpError as e:
        return 0


def analyze_channel_engagement(youtube, channel_id, channel_name, max_videos=20):
    """Analyze engagement for a channel"""
    videos = get_channel_videos(youtube, channel_id, max_videos)
    
    if not videos:
        return None
    
    total_comments = 0
    video_counts = []
    
    for video_id in videos:
        count = get_video_comment_count(youtube, video_id)
        total_comments += count
        video_counts.append(count)
    
    avg_comments = total_comments / len(videos) if videos else 0
    
    return {
        'channel_id': channel_id,
        'channel_name': channel_name,
        'videos_analyzed': len(videos),
        'total_comments': total_comments,
        'avg_comments_per_video': avg_comments,
        'max_comments': max(video_counts) if video_counts else 0
    }


def find_high_engagement_channels(queries: list, min_comments: int = 50, 
                                 top_n: int = 15, max_videos: int = 20):
    """
    Find channels with high comment engagement
    
    Args:
        queries: List of search queries
        min_comments: Minimum average comments per video
        top_n: Number of top channels to return
        max_videos: Maximum videos to analyze per channel
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("Error: YOUTUBE_API_KEY environment variable not set")
        sys.exit(1)
    
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    # Find channels from queries
    all_channels = []
    
    for query in queries:
        try:
            request = youtube.search().list(
                part='snippet',
                q=query,
                type='channel',
                maxResults=20
            )
            
            response = request.execute()
            
            for item in response.get('items', []):
                channel_id = item['id']['channelId']
                channel_name = item['snippet'].get('title', 'N/A')
                
                # Check if already added
                if not any(c['channel_id'] == channel_id for c in all_channels):
                    all_channels.append({
                        'channel_id': channel_id,
                        'channel_name': channel_name
                    })
        except HttpError as e:
            print(f"Error searching for '{query}': {e}")
            continue
    
    print(f"\nAnalyzing {len(all_channels)} channels for engagement...\n")
    
    # Analyze engagement
    results = []
    for i, channel in enumerate(all_channels, 1):
        print(f"[{i}/{len(all_channels)}] Analyzing {channel['channel_name']}...")
        
        engagement = analyze_channel_engagement(
            youtube, 
            channel['channel_id'], 
            channel['channel_name'],
            max_videos=max_videos
        )
        
        if engagement and engagement['avg_comments_per_video'] >= min_comments:
            results.append(engagement)
    
    # Sort by average comments
    results.sort(key=lambda x: x['avg_comments_per_video'], reverse=True)
    results = results[:top_n]
    
    # Print results
    print("\n" + "=" * 80)
    print(f"Top {len(results)} High-Engagement Channels")
    print("=" * 80)
    
    for result in results:
        print(f"\nChannel: {result['channel_name']}")
        print(f"ID: {result['channel_id']}")
        print(f"Avg Comments/Video: {result['avg_comments_per_video']:.1f}")
        print(f"Total Comments: {result['total_comments']}")
        print(f"Videos Analyzed: {result['videos_analyzed']}")
    
    # Generate YAML format for config.yaml
    print("\n" + "=" * 80)
    print("YAML Format for config.yaml:")
    print("=" * 80)
    print("\nchannels:")
    for result in results:
        print(f"  - name: \"{result['channel_name']}\"")
        print(f"    id: \"{result['channel_id']}\"")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find YouTube channels with high comment engagement"
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["mental health", "therapy", "mental health advocacy", "depression", "anxiety"],
        help="Search queries (default: mental health related)"
    )
    parser.add_argument(
        "--min-comments",
        type=int,
        default=50,
        help="Minimum average comments per video (default: 50)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="Number of top channels to return (default: 15)"
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=20,
        help="Maximum videos to analyze per channel (default: 20)"
    )
    
    args = parser.parse_args()
    
    find_high_engagement_channels(
        args.queries,
        min_comments=args.min_comments,
        top_n=args.top_n,
        max_videos=args.max_videos
    )

