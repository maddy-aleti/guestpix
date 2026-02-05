#!/usr/bin/env python3
"""
Simple script to run MongoDB pipeline with all photos from data/raw_photos
"""

import os
import sys
import argparse
import shutil
import uuid
import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline_mongodb import MongoDBFaceRecognitionPipeline


def get_user_input_and_organize_photos() -> List[Dict[str, Any]]:
    """
    Prompts the user for event and photographer details, organizes the photos
    into a dynamic directory structure, and returns a list of photo metadata.
    """
    print("--- User Input & Photo Organization ---")
    
    # Get event name and create unique event ID
    event_name = input("Enter the event name: ").strip()
    if not event_name:
        print("❌ Error: Event name cannot be empty.")
        return []
    
    event_id = f"{event_name.replace(' ', '_').lower()}_{uuid.uuid4().hex[:8]}"
    
    # Create the event directory
    event_dir = Path("data/raw_photos") / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created event directory: {event_dir}")
    
    # Get number of photographers with validation
    while True:
        try:
            num_photographers_input = input("Enter the number of photographers for this event: ").strip()
            if not num_photographers_input:
                print("❌ Error: Number of photographers cannot be empty.")
                continue
            num_photographers = int(num_photographers_input)
            if num_photographers <= 0:
                print("❌ Error: Number of photographers must be greater than 0.")
                continue
            break
        except ValueError:
            print("❌ Invalid input. Please enter a valid number.")
    
    all_photos_to_process: List[Dict[str, Any]] = []

    for i in range(num_photographers):
        print(f"\n--- Photographer {i+1}/{num_photographers} ---")
        
        # Get photographer name with validation
        while True:
            photographer_name = input(f"Enter the name of photographer {i+1}: ").strip()
            if photographer_name:
                break
            print("❌ Error: Photographer name cannot be empty.")
        
        # Create unique photographer ID
        photographer_id = f"{photographer_name.replace(' ', '_').lower()}_{uuid.uuid4().hex[:8]}"
        
        # Create the photographer directory
        photographer_dir = event_dir / photographer_id
        photographer_dir.mkdir(exist_ok=True)
        print(f"Created photographer directory: {photographer_dir}")

        # Get source path with validation
        while True:
            source_path_str = input(f"Enter the path to the folder where '{photographer_name}'s photos are saved: ").strip()
            if not source_path_str:
                print("❌ Error: Source path cannot be empty.")
                continue
            
            source_path = Path(source_path_str)
            if not source_path.exists():
                print(f"❌ Error: The provided path '{source_path}' does not exist.")
                continue
            if not source_path.is_dir():
                print(f"❌ Error: The provided path '{source_path}' is not a directory.")
                continue
            break
        
        # Copy photos with progress feedback
        supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]
        photos_copied = 0
        skipped_files = 0
        
        print(f"📁 Scanning photos in: {source_path}")
        
        try:
            for file_name in os.listdir(source_path):
                file_path = source_path / file_name
                if file_path.is_file() and file_path.suffix.lower() in [f.lower() for f in supported_formats]:
                    dest_path = photographer_dir / file_name
                    try:
                        shutil.copy2(file_path, dest_path)
                        photos_copied += 1
                        all_photos_to_process.append({
                            "path": str(dest_path),
                            "event_id": event_id,
                            "photographer_id": photographer_id
                        })
                    except Exception as e:
                        print(f"⚠️  Warning: Failed to copy {file_name}: {e}")
                        skipped_files += 1
                else:
                    skipped_files += 1
            
            print(f"Copied {photos_copied} photos for '{photographer_name}' to the new directory.")
            if skipped_files > 0:
                print(f"⚠️  Skipped {skipped_files} files (unsupported format or other issues)")
                
        except Exception as e:
            print(f"❌ Error reading directory '{source_path}': {e}")
            continue

    return all_photos_to_process


def main():
    """Run the MongoDB pipeline with dynamically organized raw photos."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run MongoDB pipeline with raw photos")
    parser.add_argument(
        '--force-reprocess', 
        action='store_true',
        help='Force reprocessing of all photos (useful for new databases)'
    )
    args = parser.parse_args()
    
    print("🚀 Starting MongoDB Pipeline with Raw Photos")
    print("=" * 50)
    
    # Get user input, organize photos, and get the list of photos to process
    photos_to_process = get_user_input_and_organize_photos()

    if not photos_to_process:
        print("❌ No photos were found or organized. Exiting.")
        return
    
    print(f"\n📸 Found {len(photos_to_process)} photos to process across various events and photographers.")
    print("📋 Photos to process:")
    for i, photo_data in enumerate(photos_to_process[:10], 1):  # Show first 10
        print(f"   {i}. Path: {photo_data['path']} | Event ID: {photo_data['event_id']} | Photographer ID: {photo_data['photographer_id']}")
    if len(photos_to_process) > 10:
        print(f"   ... and {len(photos_to_process) - 10} more photos")
    
    # Initialize pipeline
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
    
    # Run the complete pipeline
    try:
        print("\n🚀 Starting pipeline processing...")
        results = pipeline.run_complete_pipeline(photos_to_process)
        
        # Export results
        pipeline.export_pipeline_results()
        
        # Get MongoDB statistics
        mongodb_stats = pipeline.get_mongodb_stats()
        
        print("\n🎉 Pipeline completed successfully!")
        print("=" * 50)
        print(f"📊 Final Statistics:")
        print(f"   - Total photos processed: {len(photos_to_process)}")
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