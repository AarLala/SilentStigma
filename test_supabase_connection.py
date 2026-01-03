"""
Test script to verify Supabase connection and data access works locally.
This tests both reading and writing to your Supabase database.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client
from datetime import datetime

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
USE_SUPABASE = os.getenv('USE_SUPABASE', 'false').lower() == 'true'

def test_connection():
    """Test basic Supabase connection."""
    print("=" * 60)
    print("Testing Supabase Connection")
    print("=" * 60)
    
    if not SUPABASE_URL:
        print("❌ ERROR: SUPABASE_URL not found in .env")
        return False
    
    if not SUPABASE_KEY:
        print("❌ ERROR: SUPABASE_KEY not found in .env")
        return False
    
    print(f"✓ SUPABASE_URL: {SUPABASE_URL[:40]}...")
    print(f"✓ SUPABASE_KEY: {SUPABASE_KEY[:30]}...")
    print(f"✓ USE_SUPABASE: {USE_SUPABASE}")
    
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✓ Supabase client created successfully\n")
        return supabase
    except Exception as e:
        print(f"❌ ERROR: Failed to create Supabase client: {e}")
        return None


def test_read_operations(supabase):
    """Test reading data from Supabase."""
    print("=" * 60)
    print("Testing READ Operations")
    print("=" * 60)
    
    # Test 1: Read metrics
    print("\n1. Reading metrics table...")
    try:
        result = supabase.table('metrics').select('*').execute()
        if result.data:
            print(f"   ✓ Found {len(result.data)} metrics:")
            for row in result.data:
                print(f"      - {row['key']}: {row['value']}")
        else:
            print("   ⚠️  Metrics table is empty")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    # Test 2: Count comments
    print("\n2. Counting comments...")
    try:
        result = supabase.table('comments').select('id', count='exact').limit(1).execute()
        count = result.count if hasattr(result, 'count') else len(result.data) if result.data else 0
        print(f"   ✓ Total comments in database: {count:,}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    # Test 3: Count videos
    print("\n3. Counting videos...")
    try:
        result = supabase.table('videos').select('video_id', count='exact').limit(1).execute()
        count = result.count if hasattr(result, 'count') else len(result.data) if result.data else 0
        print(f"   ✓ Total videos in database: {count:,}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    # Test 4: Read sample comments
    print("\n4. Reading sample comments...")
    try:
        result = supabase.table('comments').select('id, text, like_count').limit(5).execute()
        if result.data:
            print(f"   ✓ Retrieved {len(result.data)} sample comments:")
            for i, comment in enumerate(result.data[:3], 1):
                text_preview = comment['text'][:50] + "..." if len(comment.get('text', '')) > 50 else comment.get('text', '')
                print(f"      {i}. {text_preview} (likes: {comment.get('like_count', 0)})")
        else:
            print("   ⚠️  No comments found")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    # Test 5: Test search-like query
    print("\n5. Testing search query (filtering comments)...")
    try:
        # Get comments with text containing common words
        result = supabase.table('comments').select('id, text').ilike('text', '%support%').limit(5).execute()
        if result.data:
            print(f"   ✓ Found {len(result.data)} comments containing 'support'")
        else:
            print("   ⚠️  No comments found with 'support'")
    except Exception as e:
        print(f"   ⚠️  Search query test: {e} (this is okay if no matching data)")
    
    print("\n✅ All READ operations successful!")
    return True


def test_write_operations(supabase):
    """Test writing data to Supabase (using test data)."""
    print("\n" + "=" * 60)
    print("Testing WRITE Operations")
    print("=" * 60)
    
    # Test 1: Update metrics (safe test)
    print("\n1. Testing metrics update...")
    try:
        # Get current search count
        result = supabase.table('metrics').select('value').eq('key', 'searches').execute()
        current = result.data[0]['value'] if result.data else 0
        print(f"   Current search count: {current}")
        
        # Increment (this is safe, just a counter)
        new_value = current + 1
        supabase.table('metrics').update({'value': new_value}).eq('key', 'searches').execute()
        print(f"   ✓ Updated search count to: {new_value}")
        
        # Restore original value
        supabase.table('metrics').update({'value': current}).eq('key', 'searches').execute()
        print(f"   ✓ Restored original value: {current}")
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    print("\n✅ WRITE operations test passed!")
    print("   (Note: We only tested metrics update, not inserting comments/videos)")
    return True


def test_app_integration():
    """Test that the app can use Supabase."""
    print("\n" + "=" * 60)
    print("Testing App Integration")
    print("=" * 60)
    
    # Check if app.py can import and use Supabase
    try:
        import sys
        from pathlib import Path
        
        # Add project root to path
        project_root = Path(__file__).resolve().parent
        sys.path.insert(0, str(project_root))
        
        # Try importing the app module
        from src.dashboard import app
        
        print("✓ App module imported successfully")
        
        # Check if Supabase is configured
        if app.supabase:
            print("✓ Supabase client is initialized in app")
        else:
            print("⚠️  Supabase client is not initialized in app")
            print("   Make sure SUPABASE_URL and SUPABASE_KEY are set")
        
        # Check USE_SUPABASE setting
        if app.USE_SUPABASE:
            print("✓ USE_SUPABASE is enabled - app will use Supabase for queries")
        else:
            print("⚠️  USE_SUPABASE is disabled - app will use SQLite")
            print("   Set USE_SUPABASE=true in .env to use Supabase")
        
        return True
    except Exception as e:
        print(f"❌ ERROR importing app: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Supabase Local Connection Test")
    print("=" * 60)
    print("\nThis script tests your local connection to Supabase.")
    print("Make sure your .env file has SUPABASE_URL and SUPABASE_KEY set.\n")
    
    # Test connection
    supabase = test_connection()
    if not supabase:
        print("\n❌ Connection test failed. Check your .env file.")
        sys.exit(1)
    
    # Test read operations
    read_ok = test_read_operations(supabase)
    
    # Test write operations
    write_ok = test_write_operations(supabase)
    
    # Test app integration
    app_ok = test_app_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"Connection: {'✅ PASS' if supabase else '❌ FAIL'}")
    print(f"Read Operations: {'✅ PASS' if read_ok else '❌ FAIL'}")
    print(f"Write Operations: {'✅ PASS' if write_ok else '❌ FAIL'}")
    print(f"App Integration: {'✅ PASS' if app_ok else '❌ FAIL'}")
    
    if all([supabase, read_ok, write_ok, app_ok]):
        print("\n🎉 All tests passed! Your Supabase connection is working.")
        print("\nNext steps:")
        print("1. Set USE_SUPABASE=true in .env to enable Supabase in your app")
        print("2. Start your Flask app: python -m src.dashboard.app")
        print("3. Visit http://localhost:5000/dashboard to test")
        return True
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

