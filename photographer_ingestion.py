#!/usr/bin/env python3
"""
Standalone Photographer Ingestion Script
Handles the ingestion of photos from multiple photographers for an event.
"""

import os
import sys
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Any

def ingest_photographer_photos() -> List[Dict[str, Any]]:
    """
    Interactive photographer photo ingestion with the exact output format requested.
    
    Returns:
        List of photo metadata dictionaries ready for pipeline processing
    """
    print("=== Photographer Photo Ingestion ===")
    
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
    """Main function for standalone photographer ingestion."""
    print("🚀 Photographer Photo Ingestion Tool")
    print("=" * 50)
    
    # Run the ingestion process
    photos_to_process = ingest_photographer_photos()
    
    if not photos_to_process:
        print("❌ No photos were ingested. Exiting.")
        return
    
    print(f"\n✅ Successfully ingested {len(photos_to_process)} photos!")
    print("📋 Summary:")
    
    # Group by photographer for summary
    photographer_summary = {}
    for photo_data in photos_to_process:
        photographer_id = photo_data['photographer_id']
        if photographer_id not in photographer_summary:
            photographer_summary[photographer_id] = 0
        photographer_summary[photographer_id] += 1
    
    for photographer_id, count in photographer_summary.items():
        photographer_name = photographer_id.split('_')[0].title()
        print(f"   - {photographer_name}: {count} photos")
    
    print(f"\n📁 Photos organized in: data/raw_photos/")
    print("🔍 Ready for face recognition pipeline processing!")
    
    # Optionally run the pipeline
    run_pipeline = input("\nWould you like to run the face recognition pipeline now? (y/n): ").strip().lower()
    if run_pipeline in ['y', 'yes']:
        try:
            # Add src directory to path
            sys.path.append(str(Path(__file__).parent / "src"))
            from src.pipeline_mongodb import MongoDBFaceRecognitionPipeline
            
            print("\n🚀 Starting face recognition pipeline...")
            pipeline = MongoDBFaceRecognitionPipeline()
            results = pipeline.run_complete_pipeline(photos_to_process)
            
            print("\n🎉 Pipeline completed successfully!")
            print(f"📊 Results: {results['pipeline_summary']}")
            
        except Exception as e:
            print(f"❌ Pipeline failed: {e}")
            print("💡 You can run the pipeline later using: python main.py --mongodb --directory data/raw_photos")


if __name__ == "__main__":
    main()
