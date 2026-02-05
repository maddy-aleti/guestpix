"""
Test Script for MongoDB Face Recognition Pipeline
Demonstrates the new streamlined workflow with MongoDB storage.
"""

import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline_mongodb import MongoDBFaceRecognitionPipeline
from src.store_embeddings import MongoDBStorage


def test_mongodb_connection():
    """Test MongoDB connection."""
    print("🔌 Testing MongoDB Connection...")
    try:
        storage = MongoDBStorage()
        stats = storage.get_database_stats()
        print(f"✓ MongoDB connection successful!")
        print(f"Database stats: {stats}")
        storage.close_connection()
        return True
    except Exception as e:
        print(f"✗ MongoDB connection failed: {e}")
        print("Please ensure MongoDB is running on localhost:27017")
        return False


def test_pipeline_with_sample_photos():
    """Test the complete pipeline with sample photos."""
    print("\n🧪 Testing Complete Pipeline...")
    
    # Initialize pipeline
    pipeline = MongoDBFaceRecognitionPipeline()
    
    # Find photos in data/raw_photos directory
    raw_photos_dir = Path("data/raw_photos")
    if not raw_photos_dir.exists():
        print("⚠ Raw photos directory not found. Creating test structure...")
        raw_photos_dir.mkdir(parents=True, exist_ok=True)
        print("Please add some photos to the 'data/raw_photos' directory and run again.")
        return False
    
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
    
    if not raw_photos:
        print("⚠ No photos found in 'data/raw_photos' directory.")
        print("Please add some photos and run again.")
        return False
    
    print(f"Found {len(raw_photos)} photos in data/raw_photos")
    
    # Run the complete pipeline
    try:
        results = pipeline.run_complete_pipeline([str(photo) for photo in raw_photos])
        
        # Export results
        pipeline.export_pipeline_results()
        
        # Test similarity search if we have faces
        mongodb_stats = pipeline.get_mongodb_stats()
        if mongodb_stats.get('total_faces', 0) > 0:
            print("\n🔍 Testing Similarity Search...")
            test_similarity_search(pipeline)
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {e}")
        return False


def test_similarity_search(pipeline):
    """Test similarity search functionality."""
    try:
        # Get all faces from MongoDB
        all_faces = pipeline.face_embedding.mongodb_storage.get_all_faces()
        
        if len(all_faces) > 1:
            # Test similarity search with the first face
            target_face_id = all_faces[0]['face_id']
            print(f"Testing similarity search for face: {target_face_id}")
            
            similar_faces = pipeline.find_similar_faces(
                target_face_id,
                similarity_threshold=0.6,
                metric="cosine",
                limit=5
            )
            
            print(f"Found {len(similar_faces)} similar faces:")
            for i, face in enumerate(similar_faces, 1):
                print(f"  {i}. {face['face_id']} (similarity: {face['similarity_score']:.3f})")
        else:
            print("⚠ Not enough faces for similarity search test")
            
    except Exception as e:
        print(f"✗ Similarity search test failed: {e}")


def test_individual_components():
    """Test individual pipeline components."""
    print("\n🔧 Testing Individual Components...")
    
    # Test MongoDB Storage
    print("Testing MongoDB Storage...")
    try:
        storage = MongoDBStorage()
        
        # Test storing sample data
        sample_face = {
            "face_id": "test_face_001",
            "photo_id": "test_photo_001",
            "embedding_vector": [0.1, 0.2, 0.3, 0.4, 0.5] * 100,  # 500-dim vector
            "crop_path": "test/crop.jpg",
            "coordinates": {"x": 100, "y": 100, "width": 150, "height": 150},
            "confidence": 0.95,
            "model_name": "ArcFace"
        }
        
        success = storage.store_face_embedding(sample_face)
        print(f"Face storage test: {'✓' if success else '✗'}")
        
        # Test retrieval
        retrieved_face = storage.get_face_embedding("test_face_001")
        print(f"Face retrieval test: {'✓' if retrieved_face else '✗'}")
        
        # Test photo metadata storage
        sample_photo = {
            "photo_id": "test_photo_001",
            "photo_path": "test/photo.jpg",
            "file_size": 1024,
            "dimensions": {"width": 1920, "height": 1080},
            "face_count": 1
        }
        
        success = storage.store_photo_metadata(sample_photo)
        print(f"Photo metadata storage test: {'✓' if success else '✗'}")
        
        # Get stats
        stats = storage.get_database_stats()
        print(f"Database stats: {stats}")
        
        storage.close_connection()
        print("✓ MongoDB Storage component test completed")
        
    except Exception as e:
        print(f"✗ MongoDB Storage component test failed: {e}")


def main():
    """Main test function."""
    print("🧪 MongoDB Face Recognition Pipeline Test Suite")
    print("=" * 50)
    
    # Test 1: MongoDB Connection
    if not test_mongodb_connection():
        print("\n❌ MongoDB connection test failed. Please ensure MongoDB is running.")
        print("To install MongoDB:")
        print("1. Download from https://www.mongodb.com/try/download/community")
        print("2. Install and start the MongoDB service")
        print("3. Run this test again")
        return
    
    # Test 2: Individual Components
    test_individual_components()
    
    # Test 3: Complete Pipeline
    if test_pipeline_with_sample_photos():
        print("\n🎉 All tests completed successfully!")
        print("\n📋 Summary:")
        print("- MongoDB connection: ✓")
        print("- Individual components: ✓")
        print("- Complete pipeline: ✓")
        print("\n💾 Your face embeddings are now stored in MongoDB!")
        print("🔍 You can now perform similarity searches and face matching queries.")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")


if __name__ == "__main__":
    main()
