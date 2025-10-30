#!/usr/bin/env python
"""
Quick test script to verify everything is working.
"""

import sys
import importlib

def test_imports():
    """Test that all required packages can be imported."""
    packages = [
        'pandas',
        'requests',
        'transformers',
        'streamlit',
        'plotly',
        'bertopic',
        'sentence_transformers',
        'mlflow',
        'nltk',
        'sklearn'
    ]
    
    print("Testing package imports...")
    print("-" * 60)
    
    all_ok = True
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package} - {e}")
            all_ok = False
    
    print("-" * 60)
    if all_ok:
        print("✅ All packages imported successfully!")
    else:
        print("❌ Some packages failed to import.")
    
    return all_ok

def test_data_structure():
    """Test that required directories exist."""
    import os
    
    print("\nTesting directory structure...")
    print("-" * 60)
    
    directories = [
        'data/raw',
        'data/processed',
        'models/topic_model',
        'mlflow_tracking',
        'src',
        'dashboard'
    ]
    
    all_ok = True
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ - missing")
            all_ok = False
    
    print("-" * 60)
    if all_ok:
        print("✅ All directories exist!")
    else:
        print("❌ Some directories are missing.")
    
    return all_ok

def test_scripts():
    """Test that all scripts can be imported."""
    import os
    import sys
    
    sys.path.insert(0, 'src')
    
    print("\nTesting script imports...")
    print("-" * 60)
    
    scripts = [
        'fetch_news',
        'preprocess',
        'train_sentiment_model',
        'train_topic_model',
        'summarize',
        'evaluate'
    ]
    
    all_ok = True
    for script in scripts:
        try:
            importlib.import_module(script)
            print(f"✅ src/{script}.py")
        except Exception as e:
            print(f"❌ src/{script}.py - {e}")
            all_ok = False
    
    print("-" * 60)
    if all_ok:
        print("✅ All scripts can be imported!")
    else:
        print("❌ Some scripts have issues.")
    
    return all_ok

def test_sample_data():
    """Test that sample data exists and can be processed."""
    import os
    import json
    
    print("\nTesting sample data...")
    print("-" * 60)
    
    sample_file = 'data/raw/sample_news.json'
    
    if not os.path.exists(sample_file):
        print(f"❌ {sample_file} not found")
        return False
    
    try:
        with open(sample_file, 'r') as f:
            data = json.load(f)
        
        print(f"✅ Sample data loaded: {len(data)} articles")
        
        # Check structure
        if data and isinstance(data, list):
            article = data[0]
            required_keys = ['title', 'description', 'content', 'source', 'publishedAt']
            has_keys = all(key in article for key in required_keys)
            
            if has_keys:
                print("✅ Sample data structure is correct")
                return True
            else:
                print("❌ Sample data missing required keys")
                return False
        else:
            print("❌ Sample data format incorrect")
            return False
            
    except Exception as e:
        print(f"❌ Error loading sample data: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("News Trend Analysis - System Check")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("Directory Structure", test_data_structure()))
    results.append(("Script Imports", test_scripts()))
    results.append(("Sample Data", test_sample_data()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("✅ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. python run_pipeline.py")
        print("  2. streamlit run dashboard/streamlit_app.py")
    else:
        print("❌ Some tests failed. Please check the output above.")
    
    print()
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
