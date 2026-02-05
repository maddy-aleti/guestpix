#!/usr/bin/env python3
"""
Cleanup duplicates and run MongoDB pipeline with raw photos
"""

import os
import sys
import argparse
from pathlib import Path
import shutil

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline_mongodb import MongoDBFaceRecognitionPipeline


def cleanup_duplicates():
    """Clean up any duplicate photos in data/raw_photos."""
    print("🧹 Cleaning up duplicates...")
    
    raw_photos_dir = Path("data/raw_photos")
    if not raw_photos_dir.exists():
        print("❌ data/raw_photos directory not found!")
        return False
    
    # Find all image files
    supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]
    all_files = []
    
    for format_ext in supported_formats:
        all_files.extend(list(raw_photos_dir.glob(f"*{format_ext}")))
    
    # Group files by name (case-insensitive)
    file_groups = {}
    for file_path in all_files:
        name_lower = file_path.stem.lower()
        if name_lower not in file_groups:
            file_groups[name_lower] = []
        file_groups[name_lower].append(file_path)
    
    # Remove duplicates (keep the first one)
    removed_count = 0
    for name_lower, files in file_groups.items():
        if len(files) > 1:
            print(f"Found {len(files)} duplicates for '{name_lower}':")
            for i, file_path in enumerate(files):
                if i == 0:
                    print(f"  ✓ Keeping: {file_path.name}")
                else:
                    print(f"  🗑️  Removing: {file_path.name}")
                    try:
                        file_path.unlink()
                        removed_count += 1
                    except Exception as e:
                        print(f"    Error removing {file_path.name}: {e}")
    
    if removed_count > 0:
        print(f"✅ Cleaned up {removed_count} duplicate files")
    else:
        print("✅ No duplicates found")
    
    return True


def get_unique_photos():
    """Get unique photos from data/raw_photos."""
    raw_photos_dir = Path("data/raw_photos")
    if not raw_photos_dir.exists():
        print("❌ data/raw_photos directory not found!")
        return []
    
    # Get all image files (avoid duplicates)
    supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]
    raw_photos = []
    seen_files = set()
    
    for format_ext in supported_formats:
        for photo_file in raw_photos_dir.glob(f"*{format_ext}"):
            # Use absolute path to avoid duplicates
            abs_path = str(photo_file.absolute())
            if abs_path not in seen_files:
                raw_photos.append(photo_file)
                seen_files.add(abs_path)
    
    return raw_photos


def main():
    """Clean up duplicates and run the pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cleanup duplicates and run MongoDB pipeline")
    parser.add_argument(
        '--force-reprocess', 
        action='store_true',
        help='Force reprocessing of all photos (useful for new databases)'
    )
    args = parser.parse_args()
    
    print("🚀 Cleanup and Run MongoDB Pipeline")
    print("=" * 50)
    
    # Step 1: Clean up duplicates
    if not cleanup_duplicates():
        return
    
    # Step 2: Get unique photos
    raw_photos = get_unique_photos()
    
    if not raw_photos:
        print("❌ Error: No photos found in data/raw_photos directory!")
        print("Please add some photos and run again.")
        return
    
    print(f"📸 Found {len(raw_photos)} unique photos")
    
    # Step 3: Initialize pipeline
    try:
        pipeline = MongoDBFaceRecognitionPipeline()
        print("✅ MongoDB Pipeline initialized successfully")
        
        # Set force reprocessing if requested
        if args.force_reprocess:
            pipeline.photo_ingestion.set_force_reprocess(True)
            print("🔄 Force reprocessing enabled - will process all photos")
        
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {e}")
        return
    
    # Step 4: Run the complete pipeline
    try:
        print("\n🚀 Starting pipeline processing...")
        results = pipeline.run_complete_pipeline([str(photo) for photo in raw_photos])
        
        # Export results
        pipeline.export_pipeline_results()
        
        # Get MongoDB statistics
        mongodb_stats = pipeline.get_mongodb_stats()
        
        print("\n🎉 Pipeline completed successfully!")
        print("=" * 50)
        print(f"📊 Final Statistics:")
        print(f"   - Total photos processed: {len(raw_photos)}")
        print(f"   - Total faces detected: {results['pipeline_summary']['total_faces_detected']}")
        print(f"   - Total embeddings generated: {results['pipeline_summary']['total_embeddings_generated']}")
        print(f"   - MongoDB faces stored: {mongodb_stats.get('total_faces', 0)}")
        print(f"   - MongoDB photos stored: {mongodb_stats.get('total_photos', 0)}")
        print(f"   - Total execution time: {results['pipeline_summary']['total_execution_time']:.2f} seconds")
        
        print("\n💾 All face embeddings stored in MongoDB!")
        print("🔍 Ready for similarity searches and face matching queries!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        return


if __name__ == "__main__":
    main()
