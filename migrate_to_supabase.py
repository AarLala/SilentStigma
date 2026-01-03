"""
Migration script to move data from SQLite to Supabase
Run this script to migrate your existing SQLite database to Supabase.

Usage:
    python migrate_to_supabase.py

Requirements:
    - SQLite database at data/silencevoice.db
    - Supabase credentials in .env file (SUPABASE_URL and SUPABASE_KEY)
    - Supabase schema already created (run supabase_full_schema.sql first)
"""

import os
import sys
import sqlite3
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client, Client
from tqdm import tqdm
import time

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

# Load environment variables
load_dotenv()

# Get Supabase credentials
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

# For data migration, we NEED service role key for writes (bypasses RLS)
# The anon key will be blocked by RLS policies for INSERT operations
SUPABASE_SERVICE_KEY = os.getenv('SUPABASE_SERVICE_KEY')

if not SUPABASE_SERVICE_KEY:
    print("⚠️  WARNING: SUPABASE_SERVICE_KEY not found!")
    print("   Migration requires service role key to bypass RLS policies.")
    print("   Get it from: Supabase Dashboard > Settings > API > service_role key")
    print("   Add to .env: SUPABASE_SERVICE_KEY=your-service-role-key")
    response = input("\nContinue with anon key anyway? (may fail due to RLS) (y/n): ")
    if response.lower() != 'y':
        sys.exit(1)
    SUPABASE_SERVICE_KEY = SUPABASE_KEY

def migrate_comments(supabase: Client, db_path: str, batch_size: int = 1000):
    """Migrate comments from SQLite to Supabase."""
    print("\n" + "="*60)
    print("Migrating Comments")
    print("="*60)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute("SELECT COUNT(*) FROM comments")
    total = cursor.fetchone()[0]
    print(f"Total comments to migrate: {total:,}")
    
    if total == 0:
        print("No comments to migrate.")
        conn.close()
        return
    
    # Check if Supabase already has data
    try:
        result = supabase.table('comments').select('id', count='exact').limit(1).execute()
        existing_count = result.count if hasattr(result, 'count') else 0
        if existing_count > 0:
            response = input(f"\n⚠️  Supabase already has {existing_count} comments. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Migration cancelled.")
                conn.close()
                return
    except Exception as e:
        print(f"⚠️  Could not check existing comments: {e}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            conn.close()
            return
    
    # Fetch all comments
    cursor.execute("SELECT id, video_id, channel_id, channel_name, author_name, text, like_count, published_at, collected_at, processed FROM comments")
    
    migrated = 0
    errors = 0
    batch = []
    
    with tqdm(total=total, desc="Migrating comments") as pbar:
        for row in cursor:
            try:
                comment = {
                    'id': row[0],
                    'video_id': row[1],
                    'channel_id': row[2],
                    'channel_name': row[3],
                    'author_name': row[4],
                    'text': row[5] or '',
                    'like_count': row[6] or 0,
                    'published_at': row[7],
                    'collected_at': row[8] or None,
                    'processed': bool(row[9]) if row[9] is not None else False
                }
                
                batch.append(comment)
                
                # Insert in batches for better performance
                if len(batch) >= batch_size:
                    try:
                        supabase.table('comments').insert(batch).execute()
                        migrated += len(batch)
                        batch = []
                        pbar.update(batch_size)
                        time.sleep(0.1)  # Small delay to avoid rate limiting
                    except Exception as e:
                        print(f"\n⚠️  Error inserting batch: {e}")
                        errors += len(batch)
                        batch = []
                        
            except Exception as e:
                print(f"\n⚠️  Error processing comment {row[0]}: {e}")
                errors += 1
                continue
        
        # Insert remaining batch
        if batch:
            try:
                supabase.table('comments').insert(batch).execute()
                migrated += len(batch)
                pbar.update(len(batch))
            except Exception as e:
                print(f"\n⚠️  Error inserting final batch: {e}")
                errors += len(batch)
    
    conn.close()
    print(f"\n✅ Comments migration complete!")
    print(f"   Migrated: {migrated:,}")
    print(f"   Errors: {errors:,}")
    return migrated, errors


def migrate_videos(supabase: Client, db_path: str, batch_size: int = 500):
    """Migrate videos from SQLite to Supabase."""
    print("\n" + "="*60)
    print("Migrating Videos")
    print("="*60)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get total count
    cursor.execute("SELECT COUNT(*) FROM videos")
    total = cursor.fetchone()[0]
    print(f"Total videos to migrate: {total:,}")
    
    if total == 0:
        print("No videos to migrate.")
        conn.close()
        return
    
    # Check if Supabase already has data
    try:
        result = supabase.table('videos').select('video_id', count='exact').limit(1).execute()
        existing_count = result.count if hasattr(result, 'count') else 0
        if existing_count > 0:
            response = input(f"\n⚠️  Supabase already has {existing_count} videos. Continue anyway? (y/n): ")
            if response.lower() != 'y':
                print("Migration cancelled.")
                conn.close()
                return
    except Exception as e:
        print(f"⚠️  Could not check existing videos: {e}")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Migration cancelled.")
            conn.close()
            return
    
    # Fetch all videos
    cursor.execute("SELECT video_id, channel_id, channel_name, title, description, published_at, comment_count, collected_at FROM videos")
    
    migrated = 0
    errors = 0
    batch = []
    
    with tqdm(total=total, desc="Migrating videos") as pbar:
        for row in cursor:
            try:
                video = {
                    'video_id': row[0],
                    'channel_id': row[1],
                    'channel_name': row[2],
                    'title': row[3] or '',
                    'description': row[4] or '',
                    'published_at': row[5],
                    'comment_count': row[6] or 0,
                    'collected_at': row[7] or None
                }
                
                batch.append(video)
                
                # Insert in batches
                if len(batch) >= batch_size:
                    try:
                        supabase.table('videos').insert(batch).execute()
                        migrated += len(batch)
                        batch = []
                        pbar.update(batch_size)
                        time.sleep(0.1)
                    except Exception as e:
                        print(f"\n⚠️  Error inserting batch: {e}")
                        errors += len(batch)
                        batch = []
                        
            except Exception as e:
                print(f"\n⚠️  Error processing video {row[0]}: {e}")
                errors += 1
                continue
        
        # Insert remaining batch
        if batch:
            try:
                supabase.table('videos').insert(batch).execute()
                migrated += len(batch)
                pbar.update(len(batch))
            except Exception as e:
                print(f"\n⚠️  Error inserting final batch: {e}")
                errors += len(batch)
    
    conn.close()
    print(f"\n✅ Videos migration complete!")
    print(f"   Migrated: {migrated:,}")
    print(f"   Errors: {errors:,}")
    return migrated, errors


def main():
    """Main migration function."""
    print("="*60)
    print("SilenceVoice - SQLite to Supabase Migration")
    print("="*60)
    
    # Check Supabase credentials
    if not SUPABASE_URL:
        print("❌ ERROR: SUPABASE_URL not found in environment variables")
        print("   Make sure you have SUPABASE_URL in your .env file")
        sys.exit(1)
    
    if not SUPABASE_KEY:
        print("❌ ERROR: SUPABASE_KEY not found in environment variables")
        print("   Make sure you have SUPABASE_KEY in your .env file")
        sys.exit(1)
    
    # Check SQLite database
    BASE_DIR = Path(__file__).resolve().parent
    db_path = BASE_DIR / "data" / "silencevoice.db"
    
    if not db_path.exists():
        print(f"❌ ERROR: SQLite database not found at {db_path}")
        print("   Make sure your database file exists")
        sys.exit(1)
    
    print(f"✓ Found SQLite database: {db_path}")
    print(f"✓ Supabase URL: {SUPABASE_URL[:30]}...")
    
    # Create Supabase client
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("✓ Supabase client created successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to create Supabase client: {e}")
        sys.exit(1)
    
    # Test connection
    try:
        result = supabase.table('metrics').select('key').limit(1).execute()
        print("✓ Supabase connection successful")
    except Exception as e:
        print(f"❌ ERROR: Could not connect to Supabase: {e}")
        print("   Make sure you've run supabase_full_schema.sql in your Supabase SQL Editor")
        sys.exit(1)
    
    # Confirm migration
    print("\n⚠️  WARNING: This will migrate data from SQLite to Supabase.")
    print("   Make sure you've already run supabase_full_schema.sql in Supabase!")
    response = input("\nContinue with migration? (y/n): ")
    if response.lower() != 'y':
        print("Migration cancelled.")
        sys.exit(0)
    
    # Run migrations
    try:
        comments_migrated, comments_errors = migrate_comments(supabase, str(db_path))
        videos_migrated, videos_errors = migrate_videos(supabase, str(db_path))
        
        print("\n" + "="*60)
        print("Migration Summary")
        print("="*60)
        print(f"Comments: {comments_migrated:,} migrated, {comments_errors:,} errors")
        print(f"Videos: {videos_migrated:,} migrated, {videos_errors:,} errors")
        print("\n✅ Migration complete!")
        print("\nNext steps:")
        print("1. Update your app to use Supabase (see updated app.py)")
        print("2. Set USE_SUPABASE=true in your environment variables")
        print("3. Test the application to ensure everything works")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: Migration failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

