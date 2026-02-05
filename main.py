#!/usr/bin/env python3
"""
Main Execution Script for Face Recognition Pipeline
Simple interface to run the complete face recognition workflow.
"""

import os
import sys
import argparse
from pathlib import Path
from src.pipeline_mongodb import MongoDBFaceRecognitionPipeline


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Face Recognition Pipeline - Identify and track individuals across multiple photos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run MongoDB pipeline (new streamlined workflow)
  python main.py --mongodb --photos photo1.jpg photo2.jpg photo3.jpg
  
  # Run MongoDB pipeline with photos from data/raw_photos directory
  python main.py --mongodb --directory data/raw_photos
  
  # Run pipeline with custom configuration
  python main.py --mongodb --photos photo1.jpg --config custom_config.yaml
        """
    )
    
    parser.add_argument(
        '--photos', 
        nargs='+', 
        help='List of photo file paths to process'
    )
    
    parser.add_argument(
        '--directory', 
        type=str, 
        help='Directory containing photos to process'
    )
    
    parser.add_argument(
        '--config', 
        type=str, 
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    

    parser.add_argument(
        '--mongodb', 
        action='store_true',
        help='Use MongoDB pipeline (new streamlined workflow)'
    )

    parser.add_argument(
        '--force-reprocess', 
        action='store_true',
        help='Force reprocessing of all photos (useful for new databases)'
    )

    # Note: user photo matching has been removed per request
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='results',
        help='Output directory for results (default: results)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.photos and not args.directory:
        print("❌ Error: Must specify either --photos or --directory")
        parser.print_help()
        sys.exit(1)
    
    if args.photos and args.directory:
        print("❌ Error: Cannot specify both --photos and --directory")
        parser.print_help()
        sys.exit(1)
    
    # Check configuration file
    if not os.path.exists(args.config):
        print(f"❌ Error: Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Get photo paths
    photo_paths = []
    
    if args.photos:
        # Use specified photo paths
        for photo_path in args.photos:
            if os.path.exists(photo_path):
                photo_paths.append(photo_path)
            else:
                print(f"⚠️  Warning: Photo not found: {photo_path}")
    
    elif args.directory:
        # Scan directory for photos
        directory = Path(args.directory)
        if not directory.exists():
            print(f"❌ Error: Directory not found: {args.directory}")
            sys.exit(1)
        
        # Supported image formats
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.JPG', '.JPEG', '.PNG', '.BMP']
        seen_files = set()
        
        for photo_file in directory.iterdir():
            if photo_file.suffix.lower() in [f.lower() for f in supported_formats]:
                # Use absolute path to avoid duplicates
                abs_path = str(photo_file.absolute())
                if abs_path not in seen_files:
                    photo_paths.append(str(photo_file))
                    seen_files.add(abs_path)
        
        if not photo_paths:
            print(f"❌ Error: No supported photos found in directory: {args.directory}")
            sys.exit(1)
    
    if not photo_paths:
        print("❌ Error: No valid photos found to process")
        sys.exit(1)
    
    print(f"📸 Found {len(photo_paths)} photos to process:")
    for photo_path in photo_paths:
        print(f"   - {photo_path}")
    
    # Initialize MongoDB pipeline
    try:
        pipeline = MongoDBFaceRecognitionPipeline(args.config)
        print(f"✅ MongoDB Pipeline initialized with config: {args.config}")
        print("📊 Using streamlined workflow: Photo Ingestion → Face Detection → Face Embedding & MongoDB Storage")
        
        # Set force reprocessing if requested
        if args.force_reprocess:
            pipeline.photo_ingestion.set_force_reprocess(True)
            print("🔄 Force reprocessing enabled - will process all photos")
    except Exception as e:
        print(f"❌ Failed to initialize pipeline: {str(e)}")
        sys.exit(1)
    
    # Run MongoDB pipeline
    try:
        print(f"\n🚀 Starting MongoDB pipeline (streamlined workflow)...")
        results = pipeline.run_complete_pipeline(photo_paths)
        
        # Export results
        output_file = os.path.join(args.output_dir, "mongodb_pipeline_results.json")
        pipeline.export_pipeline_results(output_file)
        
        print(f"\n🎉 MongoDB Pipeline completed successfully!")
        print(f"📄 Results saved to: {output_file}")
        print(f"💾 All face embeddings stored in MongoDB!")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 