"""
Streamlined MongoDB Pipeline Orchestrator
Coordinates the new three-step workflow with MongoDB storage.
"""

import os
import time
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import yaml
from tqdm import tqdm

from .photo_ingestion import PhotoIngestion
from .face_detection import FaceDetection
from .face_embedding import FaceEmbedding


class MongoDBFaceRecognitionPipeline:
    """
    Streamlined orchestrator for the new face recognition pipeline with MongoDB.
    
    This class coordinates the new three-step workflow:
    1. Photo Ingestion & Storage
    2. Face Detection
    3. Face Embedding & MongoDB Storage
    
    Expected output:
    - All face embeddings stored in MongoDB
    - Photo metadata stored in MongoDB
    - Pipeline statistics and reports
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the streamlined face recognition pipeline."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # Initialize MongoDB storage first to get database connection
        try:
            from .store_embeddings import MongoDBStorage
            self.mongodb_storage = MongoDBStorage(config_path)
            print("✓ MongoDB storage initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize MongoDB storage: {e}")
            self.mongodb_storage = None
        
        # Initialize pipeline components
        # Pass MongoDB db handle to PhotoIngestion for duplicate checking
        # Check if this is a new database and set force_reprocess accordingly
        force_reprocess = False
        if self.mongodb_storage is not None and self.mongodb_storage.db is not None:
            try:
                photos_collection = self.config['mongodb']['collections']['photos']
                faces_collection = self.config['mongodb']['collections']['faces']
                photos_count = self.mongodb_storage.db[photos_collection].count_documents({})
                faces_count = self.mongodb_storage.db[faces_collection].count_documents({})
                force_reprocess = (photos_count == 0 and faces_count == 0)
                if force_reprocess:
                    print("🆕 New database detected - will process all photos")
            except Exception:
                pass
        
        self.photo_ingestion = PhotoIngestion(
            config_path, 
            db=(self.mongodb_storage.db if self.mongodb_storage is not None else None),
            force_reprocess=force_reprocess
        )
        self.face_detection = FaceDetection(config_path)
        # Pass the same MongoDB storage instance to face embedding
        self.face_embedding = FaceEmbedding(config_path)
        if self.mongodb_storage is not None:
            self.face_embedding.mongodb_storage = self.mongodb_storage
        
        # Pipeline state
        self.pipeline_results = {}
        self.execution_times = {}
        self.current_step = None
        
        print("🚀 MongoDB Face Recognition Pipeline Initialized")
        print(f"Configuration: {config_path}")
        print("📋 New Workflow: Photo Ingestion → Face Detection → Face Embedding & MongoDB Storage")
    
    def run_complete_pipeline(self, photo_paths: List[str]) -> Dict:
        """
        Run the complete streamlined face recognition pipeline.
        
        Args:
            photo_paths: List of paths to input photos
            
        Returns:
            Complete pipeline results
        """
        start_time = time.time()
        
        print("\n" + "="*60)
        print("🎯 STARTING STREAMLINED FACE RECOGNITION PIPELINE")
        print("📊 New Workflow: 3 Steps with MongoDB Storage")
        print("="*60)
        
        try:
            # Step 1: Photo Ingestion & Storage
            print(f"\n📸 STEP 1: Photo Ingestion & Storage")
            print("-" * 40)
            step1_start = time.time()
            
            photo_ids = self.photo_ingestion.ingest_photos(photo_paths)
            if len(photo_ids) == 0:
                print("\n✅ No new photos to process. All files in directory were already ingested.")
                print("Checking if existing photos need face detection and embedding...")
                
                # Get existing photos that might need processing
                photos_needing_processing = self.photo_ingestion.get_photos_needing_processing()
                if photos_needing_processing:
                    print(f"Found {len(photos_needing_processing)} photos that need face detection and embedding")
                    # Use photos that need processing
                    photo_ids = photos_needing_processing
                else:
                    # All photos are fully processed - return early
                    self.pipeline_results['photo_ingestion'] = {
                        'photo_ids': [],
                        'metadata': {},
                        'stats': self.photo_ingestion.get_storage_stats()
                    }
                    total_time = time.time() - start_time
                    self.execution_times['total'] = total_time
                    self.pipeline_results['pipeline_summary'] = {
                        'total_execution_time': total_time,
                        'step_execution_times': self.execution_times,
                        'total_photos_processed': 0,
                        'total_faces_detected': 0,
                        'total_embeddings_generated': 0,
                        'mongodb_faces_stored': 0,
                        'mongodb_photos_stored': self.mongodb_storage.get_database_stats().get('total_photos', 0) if self.mongodb_storage is not None else 0
                    }
                    print("\n" + "="*60)
                    print("🎉 STREAMLINED PIPELINE COMPLETED SUCCESSFULLY!")
                    print("="*60)
                    print(f"Total execution time: {total_time:.2f} seconds")
                    print("All photos are fully processed - no work needed")
                    print("💾 All face embeddings stored in MongoDB!")
                    print("🔍 Ready for similarity searches and face matching queries!")
                    return self.pipeline_results
            self.pipeline_results['photo_ingestion'] = {
                'photo_ids': photo_ids,
                'metadata': self.photo_ingestion.photos_metadata,
                'stats': self.photo_ingestion.get_storage_stats()
            }
            
            step1_time = time.time() - step1_start
            self.execution_times['photo_ingestion'] = step1_time
            print(f"✅ Step 1 completed in {step1_time:.2f} seconds")
            
            # Step 2: Face Detection
            print(f"\n🧐 STEP 2: Face Detection")
            print("-" * 40)
            step2_start = time.time()
            
            # Get metadata for photos that need processing
            if len(photo_ids) > 0:
                photos_metadata = self.photo_ingestion.get_metadata_for_photo_ids(photo_ids)
            else:
                photos_metadata = {}
            
            detection_results = self.face_detection.process_photos(photos_metadata)
            self.pipeline_results['face_detection'] = {
                'results': detection_results,
                'stats': self.face_detection.get_face_statistics()
            }
            
            step2_time = time.time() - step2_start
            self.execution_times['face_detection'] = step2_time
            print(f"✅ Step 2 completed in {step2_time:.2f} seconds")
            
            # Step 3: Face Embedding & MongoDB Storage
            print(f"\n🔢 STEP 3: Face Embedding & MongoDB Storage")
            print("-" * 40)
            step3_start = time.time()
            
            embedding_results = self.face_embedding.process_faces(detection_results)
            self.pipeline_results['face_embedding'] = {
                'results': embedding_results,
                'stats': self.face_embedding.get_embedding_statistics()
            }
            
            step3_time = time.time() - step3_start
            self.execution_times['face_embedding'] = step3_time
            print(f"✅ Step 3 completed in {step3_time:.2f} seconds")
            
            # Get MongoDB statistics
            mongodb_stats = {}
            if self.face_embedding.mongodb_storage:
                mongodb_stats = self.face_embedding.mongodb_storage.get_database_stats()
                self.pipeline_results['mongodb_stats'] = mongodb_stats
            
            # Compile final results
            total_time = time.time() - start_time
            self.execution_times['total'] = total_time
            
            self.pipeline_results['pipeline_summary'] = {
                'total_execution_time': total_time,
                'step_execution_times': self.execution_times,
                'total_photos_processed': len(photo_ids),
                'total_faces_detected': sum(len(result['faces']) for result in detection_results.values()),
                'total_embeddings_generated': sum(len(result['embeddings']) for result in embedding_results.values()),
                'mongodb_faces_stored': mongodb_stats.get('total_faces', 0),
                'mongodb_photos_stored': mongodb_stats.get('total_photos', 0)
            }
            
            print("\n" + "="*60)
            print("🎉 STREAMLINED PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Total execution time: {total_time:.2f} seconds")
            print(f"Photos processed: {len(photo_ids)}")
            print(f"Faces detected: {self.pipeline_results['pipeline_summary']['total_faces_detected']}")
            print(f"Embeddings generated: {self.pipeline_results['pipeline_summary']['total_embeddings_generated']}")
            print(f"MongoDB faces stored: {mongodb_stats.get('total_faces', 0)}")
            print(f"MongoDB photos stored: {mongodb_stats.get('total_photos', 0)}")
            print("\n💾 All face embeddings and metadata stored in MongoDB!")
            print("🔍 Ready for similarity searches and face matching queries!")
            
            return self.pipeline_results
            
        except Exception as e:
            print(f"\n❌ Pipeline failed at step: {self.current_step}")
            print(f"Error: {str(e)}")
            raise
    
    def run_step(self, step_name: str, **kwargs) -> Dict:
        """
        Run a specific pipeline step.
        
        Args:
            step_name: Name of the step to run
            **kwargs: Arguments for the step
            
        Returns:
            Results from the specified step
        """
        step_methods = {
            'photo_ingestion': self._run_photo_ingestion,
            'face_detection': self._run_face_detection,
            'face_embedding': self._run_face_embedding
        }
        
        if step_name not in step_methods:
            raise ValueError(f"Unknown step: {step_name}")
        
        self.current_step = step_name
        return step_methods[step_name](**kwargs)
    
    def _run_photo_ingestion(self, photo_paths: List[str]) -> Dict:
        """Run photo ingestion step."""
        photo_ids = self.photo_ingestion.ingest_photos(photo_paths)
        return {
            'photo_ids': photo_ids,
            'metadata': self.photo_ingestion.photos_metadata,
            'stats': self.photo_ingestion.get_storage_stats()
        }
    
    def _run_face_detection(self, photo_metadata: Dict) -> Dict:
        """Run face detection step."""
        detection_results = self.face_detection.process_photos(photo_metadata)
        return {
            'results': detection_results,
            'stats': self.face_detection.get_face_statistics()
        }
    
    def _run_face_embedding(self, detection_results: Dict) -> Dict:
        """Run face embedding step."""
        embedding_results = self.face_embedding.process_faces(detection_results)
        return {
            'results': embedding_results,
            'stats': self.face_embedding.get_embedding_statistics()
        }
    
    def get_mongodb_stats(self) -> Dict:
        """
        Get MongoDB database statistics.
        
        Returns:
            Dictionary with MongoDB statistics
        """
        if self.mongodb_storage is not None:
            return self.mongodb_storage.get_database_stats()
        return {}
    
    def find_similar_faces(self, target_face_id: str, 
                          similarity_threshold: float = 0.6,
                          metric: str = "cosine",
                          limit: int = 10) -> List[Dict]:
        """
        Find faces similar to a target face using MongoDB.
        
        Args:
            target_face_id: ID of the target face
            similarity_threshold: Threshold for similarity
            metric: Similarity metric ('cosine', 'euclidean')
            limit: Maximum number of results to return
            
        Returns:
            List of similar faces with similarity scores
        """
        if self.mongodb_storage is None:
            print("✗ MongoDB storage not available")
            return []
        
        # Get target face embedding
        target_face = self.face_embedding.mongodb_storage.get_face_embedding(target_face_id)
        if not target_face:
            print(f"✗ Target face {target_face_id} not found in database")
            return []
        
        # Find similar faces
        similar_faces = self.face_embedding.mongodb_storage.find_similar_faces(
            target_face['embedding_vector'],
            similarity_threshold,
            metric,
            limit
        )
        
        return similar_faces
    
    def search_faces_by_photo(self, photo_id: str) -> List[Dict]:
        """
        Get all faces from a specific photo.
        
        Args:
            photo_id: ID of the photo
            
        Returns:
            List of face documents
        """
        if self.mongodb_storage is None:
            print("✗ MongoDB storage not available")
            return []
        
        return self.mongodb_storage.get_faces_by_photo(photo_id)
    
    def export_pipeline_results(self, output_path: str = "results/mongodb_pipeline_results.json"):
        """
        Export pipeline results to JSON file.
        
        Args:
            output_path: Path to save the results
        """
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_to_save = self.pipeline_results.copy()
        
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        print(f"Pipeline results saved to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Test the streamlined pipeline
    pipeline = MongoDBFaceRecognitionPipeline()
    
    # Sample photo paths (replace with actual paths)
    sample_photos = [
        "sample_photos/photo1.jpg",
        "sample_photos/photo2.jpg"
    ]
    
    # Check if sample photos exist
    existing_photos = [photo for photo in sample_photos if os.path.exists(photo)]
    
    if existing_photos:
        # Run the complete pipeline
        results = pipeline.run_complete_pipeline(existing_photos)
        
        # Get MongoDB statistics
        mongodb_stats = pipeline.get_mongodb_stats()
        print(f"\nMongoDB Statistics: {mongodb_stats}")
        
        # Export results
        pipeline.export_pipeline_results()
    else:
        print("No sample photos found. Please provide valid photo paths.")
