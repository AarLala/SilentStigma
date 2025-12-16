"""
Test script to verify Supabase integration is working correctly.
Run this from the project root: python test_supabase.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

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

def test_supabase_connection():
    """Test if Supabase connection works."""
    print("=" * 60)
    print("Testing Supabase Integration")
    print("=" * 60)
    
    # Check if credentials are set
    if not SUPABASE_URL:
        print("❌ ERROR: SUPABASE_URL not found in environment variables")
        print("   Make sure you have SUPABASE_URL in your .env file")
        return False
    
    if not SUPABASE_KEY:
        print("❌ ERROR: SUPABASE_KEY not found in environment variables")
        print("   Make sure you have SUPABASE_KEY in your .env file")
        return False
    
    print(f"✓ Found SUPABASE_URL: {SUPABASE_URL[:30]}...")
    print(f"✓ Found SUPABASE_KEY: {SUPABASE_KEY[:20]}...")
    
    # Try to create client
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✓ Supabase client created successfully")
    except Exception as e:
        print(f"❌ ERROR: Failed to create Supabase client: {e}")
        return False
    
    # Test 1: Check if metrics table exists and has data
    print("\n--- Test 1: Reading metrics table ---")
    try:
        result = supabase.table('metrics').select('*').execute()
        if result.data:
            print(f"✓ Metrics table exists and has {len(result.data)} rows:")
            for row in result.data:
                print(f"   - {row['key']}: {row['value']}")
        else:
            print("⚠ WARNING: Metrics table exists but is empty")
            print("   Run the SQL schema from supabase_schema.sql to initialize")
    except Exception as e:
        print(f"❌ ERROR: Could not read metrics table: {e}")
        print("   Make sure you've run the SQL schema from supabase_schema.sql")
        return False
    
    # Test 2: Test search tracking
    print("\n--- Test 2: Testing search tracking ---")
    try:
        # Get current search count
        result = supabase.table('metrics').select('value').eq('key', 'searches').execute()
        current_count = result.data[0]['value'] if result.data else 0
        print(f"   Current search count: {current_count}")
        
        # Increment search count
        if result.data:
            new_count = current_count + 1
            supabase.table('metrics').update({'value': new_count}).eq('key', 'searches').execute()
            print(f"✓ Search count incremented to: {new_count}")
        else:
            print("⚠ WARNING: Searches metric not found, creating it...")
            supabase.table('metrics').insert({'key': 'searches', 'value': 1}).execute()
            print("✓ Created searches metric with value 1")
    except Exception as e:
        print(f"❌ ERROR: Search tracking test failed: {e}")
        return False
    
    # Test 3: Test download tracking
    print("\n--- Test 3: Testing download tracking ---")
    try:
        test_client_id = "test_client_" + str(int(datetime.now().timestamp()))
        
        # Check if client already exists
        result = supabase.table('download_events').select('id').eq('client_id', test_client_id).execute()
        if result.data:
            print(f"   Test client already exists (unexpected)")
        else:
            # Insert test download
            supabase.table('download_events').insert({
                'client_id': test_client_id,
                'timestamp': datetime.utcnow().isoformat()
            }).execute()
            print(f"✓ Test download event created for client: {test_client_id}")
            
            # Try to insert again (should fail or be ignored)
            try:
                supabase.table('download_events').insert({
                    'client_id': test_client_id,
                    'timestamp': datetime.utcnow().isoformat()
                }).execute()
                print("⚠ WARNING: Duplicate download was allowed (UNIQUE constraint may not be working)")
            except:
                print("✓ Duplicate download correctly prevented (UNIQUE constraint working)")
            
            # Clean up test data
            supabase.table('download_events').delete().eq('client_id', test_client_id).execute()
            print("✓ Test download event cleaned up")
    except Exception as e:
        print(f"❌ ERROR: Download tracking test failed: {e}")
        return False
    
    # Test 4: Test session tracking
    print("\n--- Test 4: Testing session tracking ---")
    try:
        test_client_id = "test_session_" + str(int(datetime.now().timestamp()))
        
        # Insert test session
        supabase.table('session_events').insert({
            'client_id': test_client_id,
            'timestamp': datetime.utcnow().isoformat()
        }).execute()
        print(f"✓ Test session event created for client: {test_client_id}")
        
        # Check if we can query it
        result = supabase.table('session_events').select('*').eq('client_id', test_client_id).order('timestamp', desc=True).limit(1).execute()
        if result.data:
            print(f"✓ Session event retrieved: {result.data[0]['timestamp']}")
        
        # Clean up test data
        supabase.table('session_events').delete().eq('client_id', test_client_id).execute()
        print("✓ Test session event cleaned up")
    except Exception as e:
        print(f"❌ ERROR: Session tracking test failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Supabase integration is working correctly.")
    print("=" * 60)
    return True

if __name__ == "__main__":
    from datetime import datetime
    success = test_supabase_connection()
    sys.exit(0 if success else 1)

