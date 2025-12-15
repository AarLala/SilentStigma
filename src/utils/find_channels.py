"""
Utility script to find YouTube channels by search query
"""

import argparse
import os
import sys
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

load_dotenv()


def find_channels(query: str, max_results: int = 10):
    """
    Search for YouTube channels by query
    
    Args:
        query: Search query
        max_results: Maximum number of results
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        print("Error: YOUTUBE_API_KEY environment variable not set")
        sys.exit(1)
    
    youtube = build('youtube', 'v3', developerKey=api_key)
    
    try:
        # Search for channels
        request = youtube.search().list(
            part='snippet',
            q=query,
            type='channel',
            maxResults=max_results
        )
        
        response = request.execute()
        
        print(f"\nFound {len(response.get('items', []))} channels for query: '{query}'\n")
        print("-" * 80)
        
        for item in response.get('items', []):
            snippet = item['snippet']
            channel_id = item['id']['channelId']
            title = snippet.get('title', 'N/A')
            description = snippet.get('description', 'N/A')[:100]
            subscriber_count = snippet.get('subscriberCount', 'N/A')
            
            print(f"Channel: {title}")
            print(f"ID: {channel_id}")
            print(f"Description: {description}...")
            print(f"Subscribers: {subscriber_count}")
            print("-" * 80)
        
    except HttpError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find YouTube channels by search query")
    parser.add_argument("query", type=str, help="Search query")
    parser.add_argument("--max-results", type=int, default=10, help="Maximum number of results")
    
    args = parser.parse_args()
    
    find_channels(args.query, args.max_results)

