"""
Quick script to verify Supabase schema is set up correctly.
Run this before migrating data.
"""

import os
import sys
from dotenv import load_dotenv
from supabase import create_client

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except:
        pass

load_dotenv()

SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')

def verify_schema():
    """Verify all required tables exist in Supabase."""
    print("=" * 60)
    print("Verifying Supabase Schema")
    print("=" * 60)
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("❌ ERROR: SUPABASE_URL and SUPABASE_KEY must be set in .env")
        return False
    
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✓ Connected to Supabase\n")
    except Exception as e:
        print(f"❌ ERROR: Could not connect to Supabase: {e}")
        return False
    
    # Required tables
    required_tables = [
        'comments',
        'videos', 
        'metrics',
        'download_events',
        'session_events'
    ]
    
    missing_tables = []
    
    for table in required_tables:
        try:
            # Try to query the table (even if empty)
            result = supabase.table(table).select('*').limit(1).execute()
            print(f"✓ Table '{table}' exists")
        except Exception as e:
            error_msg = str(e)
            if 'could not find the table' in error_msg.lower() or 'PGRST205' in error_msg:
                print(f"❌ Table '{table}' is MISSING")
                missing_tables.append(table)
            else:
                print(f"⚠️  Table '{table}' exists but error: {e}")
    
    print("\n" + "=" * 60)
    if missing_tables:
        print("❌ SCHEMA INCOMPLETE")
        print(f"\nMissing tables: {', '.join(missing_tables)}")
        print("\n📋 ACTION REQUIRED:")
        print("1. Go to your Supabase dashboard")
        print("2. Click 'SQL Editor' in the left sidebar")
        print("3. Open 'supabase_full_schema.sql' from this project")
        print("4. Copy the ENTIRE contents and paste into SQL Editor")
        print("5. Click 'Run' (or press Ctrl+Enter)")
        print("6. Wait for 'Success' message")
        print("7. Run this script again to verify")
        return False
    else:
        print("✅ ALL TABLES EXIST - Ready for migration!")
        print("\nYou can now run: python migrate_to_supabase.py")
        return True

if __name__ == "__main__":
    success = verify_schema()
    sys.exit(0 if success else 1)

