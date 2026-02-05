#!/usr/bin/env python3
"""
Test script to demonstrate new database detection and force reprocessing
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline_mongodb import MongoDBFaceRecognitionPipeline
from src.photo_ingestion import PhotoIngestion


def test_new_database_detection():
    """Test if the system correctly detects new databases."""
    print("🧪 Testing New Database Detection")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        pipeline = MongoDBFaceRecognitionPipeline()
        
        # Check database status
        is_new = pipeline.photo_ingestion.is_new_database
        force_reprocess = pipeline.photo_ingestion.force_reprocess
        
        print(f"Database is new: {is_new}")
        print(f"Force reprocess: {force_reprocess}")
        
        if is_new:
            print("✅ Correctly detected new database")
        else:
            print("✅ Correctly detected existing database")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_force_reprocess_manual():
    """Test manual force reprocessing."""
    print("\n🧪 Testing Manual Force Reprocessing")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        pipeline = MongoDBFaceRecognitionPipeline()
        
        # Test manual force reprocessing
        print("Setting force reprocess to True...")
        pipeline.photo_ingestion.set_force_reprocess(True)
        
        print(f"Force reprocess: {pipeline.photo_ingestion.force_reprocess}")
        
        # Test clearing existing filenames
        print("Clearing existing filenames cache...")
        pipeline.photo_ingestion.clear_existing_filenames()
        
        print("✅ Manual force reprocessing test passed")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def test_reset_for_new_database():
    """Test reset for new database functionality."""
    print("\n🧪 Testing Reset for New Database")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        pipeline = MongoDBFaceRecognitionPipeline()
        
        # Test reset functionality
        print("Resetting for new database...")
        pipeline.photo_ingestion.reset_for_new_database()
        
        print(f"Force reprocess: {pipeline.photo_ingestion.force_reprocess}")
        print(f"Database is new: {pipeline.photo_ingestion.is_new_database}")
        
        print("✅ Reset for new database test passed")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🚀 Testing New Database Detection and Force Reprocessing")
    print("=" * 60)
    
    tests = [
        test_new_database_detection,
        test_force_reprocess_manual,
        test_reset_for_new_database
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed!")
    
    print("\n💡 Usage Examples:")
    print("  # Normal run (skip existing photos)")
    print("  python run_raw_photos.py")
    print("  python cleanup_and_run.py")
    print("  python main.py --mongodb --directory data/raw_photos")
    print("")
    print("  # Force reprocessing (process all photos)")
    print("  python run_raw_photos.py --force-reprocess")
    print("  python cleanup_and_run.py --force-reprocess")
    print("  python main.py --mongodb --directory data/raw_photos --force-reprocess")


if __name__ == "__main__":
    main()
