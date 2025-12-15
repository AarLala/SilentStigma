"""
Verification script to test SilenceVoice setup
"""

import sys
from pathlib import Path

def check_imports():
    """Check if all required modules can be imported"""
    print("Checking imports...")
    
    try:
        from src.data_collector import YouTubeDataCollector
        print("[OK] data_collector")
    except Exception as e:
        print(f"[FAIL] data_collector: {e}")
        return False
    
    try:
        from src.text_processor import TextProcessor
        print("[OK] text_processor")
    except Exception as e:
        print(f"[FAIL] text_processor: {e}")
        return False
    
    try:
        from src.semantic_encoder import SemanticEncoder
        print("[OK] semantic_encoder")
    except Exception as e:
        print(f"[FAIL] semantic_encoder: {e}")
        return False
    
    try:
        from src.clustering import DiscourseClusterer
        print("[OK] clustering")
    except Exception as e:
        print(f"[FAIL] clustering: {e}")
        return False
    
    try:
        from src.visualization import StigmaLandscapeVisualizer
        print("[OK] visualization")
    except Exception as e:
        print(f"[FAIL] visualization: {e}")
        return False
    
    try:
        from src.pattern_extraction import PatternExtractor
        print("[OK] pattern_extraction")
    except Exception as e:
        print(f"[FAIL] pattern_extraction: {e}")
        return False
    
    try:
        from src.pipeline import SilenceVoicePipeline
        print("[OK] pipeline")
    except Exception as e:
        print(f"[FAIL] pipeline: {e}")
        return False
    
    return True

def check_files():
    """Check if required files exist"""
    print("\nChecking files...")
    
    required_files = [
        "config.yaml",
        "requirements.txt",
        "README.md",
        "QUICKSTART.md",
        "src/__init__.py",
        "src/data_collector.py",
        "src/text_processor.py",
        "src/semantic_encoder.py",
        "src/clustering.py",
        "src/visualization.py",
        "src/pattern_extraction.py",
        "src/pipeline.py",
        "src/dashboard/__init__.py",
        "src/dashboard/app.py",
        "src/dashboard/templates/index.html",
        "src/utils/__init__.py",
        "src/utils/find_channels.py",
        "src/utils/find_high_engagement_channels.py",
    ]
    
    all_exist = True
    for file in required_files:
        if Path(file).exists():
            print(f"[OK] {file}")
        else:
            print(f"[MISSING] {file}")
            all_exist = False
    
    return all_exist

def check_directories():
    """Check if required directories exist"""
    print("\nChecking directories...")
    
    required_dirs = [
        "src",
        "src/dashboard",
        "src/dashboard/templates",
        "src/utils",
        "data",
        "outputs",
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"[OK] {dir_path}/")
        else:
            print(f"[MISSING] {dir_path}/")
            all_exist = False
    
    return all_exist

def check_config():
    """Check if config.yaml is valid"""
    print("\nChecking configuration...")
    
    try:
        import yaml
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f)
        
        required_sections = ['youtube', 'channels', 'processing', 'models', 
                           'clustering', 'umap', 'keybert', 'dashboard', 'database']
        
        for section in required_sections:
            if section in config:
                print(f"[OK] {section}")
            else:
                print(f"[MISSING] {section}")
                return False
        
        return True
    except Exception as e:
        print(f"✗ Config error: {e}")
        return False

def main():
    """Run all checks"""
    print("=" * 60)
    print("SilenceVoice Setup Verification")
    print("=" * 60)
    
    checks = [
        ("Files", check_files),
        ("Directories", check_directories),
        ("Configuration", check_config),
        ("Imports", check_imports),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\nError in {name} check: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n[SUCCESS] All checks passed! Setup is complete.")
        print("\nNext steps:")
        print("1. Create .env file with YOUTUBE_API_KEY")
        print("2. Run: python -m src.pipeline --collect")
        return 0
    else:
        print("\n[FAILED] Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

