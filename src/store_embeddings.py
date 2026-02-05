"""
MongoDB Storage Module for Face Recognition Pipeline
Handles storage and retrieval of face embeddings in MongoDB.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import yaml
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import numpy as np
from bson import ObjectId
import json


class MongoDBStorage:
    """
    MongoDB storage handler for face recognition data.
    
    What it does:
    - Connects to MongoDB database
    - Stores face embeddings with metadata
    - Provides retrieval and query methods
    - Handles data serialization/deserialization
    
    Collection:
    - photos: Stores face embeddings and metadata, with each document representing a detected face.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize MongoDB connection and collections."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        # MongoDB configuration
        self.connection_string = self.config['mongodb']['connection_string']
        self.database_name = self.config['mongodb']['database_name']
        self.collections_config = self.config['mongodb']['collections']
        
        # Initialize connection
        self.client = None
        self.db = None
        self.photos_collection = None
        # Removed: self.photos_collection
        # Removed: self.persons_collection
        
        # Connect to MongoDB
        self._connect()
    
    def _connect(self):
        """Establish connection to MongoDB."""
        try:
            self.client = MongoClient(self.connection_string)
            # Test connection
            self.client.admin.command('ping')
            print(f"✓ Connected to MongoDB at {self.connection_string}")
            
            # Get database
            self.db = self.client[self.database_name]
            
            # Get collections
            self.photos_collection = self.db[self.collections_config['photos']]
            # Removed: self.photos_collection = self.db[self.collections_config['photos']]
            # Removed: self.persons_collection = self.db[self.collections_config['persons']]
            
            # Create indexes for better performance
            self._create_indexes()
            
        except ConnectionFailure as e:
            print(f"✗ Failed to connect to MongoDB: {e}")
            raise
    
    def _create_indexes(self):
        """Create database indexes for better query performance."""
        try:
            # Index on face_id for quick lookups
            self.photos_collection.create_index("face_id", unique=True)
            
            # Index on photo_id for filtering by photo
            self.photos_collection.create_index("photo_id")
            # These two indexes are now essential for the new filtering logic
            self.photos_collection.create_index("event_id")
            self.photos_collection.create_index("photographer_id")
            
            # Index on embedding vector for similarity searches (if using vector search)
            # Note: This requires MongoDB Atlas or specific vector search extensions
            
            # Removed: Indexes for photos collection
            
            print("✓ Database indexes created successfully")
            
        except Exception as e:
            print(f"Warning: Could not create indexes: {e}")
    
    def store_face_embedding(self, face_data: Dict[str, Any]) -> bool:
        """
        Store a single face embedding in MongoDB.
        
        Args:
            face_data: Dictionary containing face information including:
                - face_id: Unique identifier for the face
                - photo_id: ID of the source photo
                - embedding_vector: Numerical embedding vector
                - crop_path: Path to the cropped face image
                - coordinates: Face coordinates in original photo
                - confidence: Detection confidence score
                - model_name: Name of the embedding model used
                
        Returns:
            bool: True if successfully stored, False otherwise
        """
        try:
            # Prepare document for storage
            face_document = {
                "face_id": face_data['face_id'],
                "photo_id": face_data['photo_id'],
                "embedding_vector": face_data['embedding_vector'].tolist() if isinstance(face_data['embedding_vector'], np.ndarray) else face_data['embedding_vector'],
                "embedding_dimension": len(face_data['embedding_vector']),
                "crop_path": face_data['crop_path'],
                "coordinates": face_data['coordinates'],
                "confidence": face_data.get('confidence', 0.0),
                "model_name": face_data.get('model_name', 'Unknown'),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
                # Added fields for filtering from GuestSearchMongoDB
                "event_id": face_data.get('event_id'),
                "photographer_id": face_data.get('photographer_id'),
                "photo_path": face_data.get('photo_path'), # Merged from photo metadata
                "public_url": face_data.get('public_url'), # Merged from photo metadata
            }
            
            # Insert into database
            result = self.photos_collection.insert_one(face_document)
            
            if result.inserted_id:
                print(f"✓ Stored face embedding: {face_data['face_id']}")
                return True
            else:
                print(f"✗ Failed to store face embedding: {face_data['face_id']}")
                return False
                
        except DuplicateKeyError:
            print(f"⚠ Face embedding already exists: {face_data['face_id']}")
            return True
        except Exception as e:
            print(f"✗ Error storing face embedding {face_data['face_id']}: {e}")
            return False
    
    # Removed: store_photo_metadata method
    # Removed: upsert_photo_from_path method

    def insert_detected_face(self, *, face_id: str, photo_id: str, event_id: str, photographer_id: str, bounding_box: Dict, crop_path: str, confidence: float, detector_backend: Optional[str] = None, photo_path: Optional[str] = None, file_size: Optional[int] = None, dimensions: Optional[Dict] = None, public_url: Optional[str] = None, face_count: Optional[int] = None) -> bool:
        """Insert a detected face document before embedding is available.

        Note: We now merge necessary photo metadata into the photos document so that
        a single collection (`photos`) contains all required information. Fields
        coming from the old `photos` collection are: `photo_path`, `file_size`,
        `dimensions`, `public_url`, and `face_count`.
        """
        try:
            doc = {
                "face_id": face_id,
                "photo_id": photo_id,
                "event_id": event_id,
                "photographer_id": photographer_id,
                "bounding_box": bounding_box,
                "crop_path": crop_path,
                "confidence": confidence,
                "detector_backend": detector_backend,
                # merged photo metadata
                "photo_path": photo_path,
                "file_size": file_size,
                "dimensions": dimensions or {},
                "public_url": public_url,
                "face_count": face_count,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            self.photos_collection.update_one({"face_id": face_id}, {"$setOnInsert": doc}, upsert=True)
            return True
        except Exception as e:
            print(f"✗ Error inserting detected face {face_id}: {e}")
            return False

    def update_face_embedding(self, *, face_id: str, embedding_vector: np.ndarray, model_name: str) -> bool:
        """Update a face document with embedding fields."""
        try:
            vector_list = embedding_vector.tolist() if isinstance(embedding_vector, np.ndarray) else embedding_vector
            update = {
                "$set": {
                    "embedding_vector": vector_list,
                    "embedding_dimension": len(vector_list),
                    "model_name": model_name,
                    "updated_at": datetime.utcnow()
                }
            }
            self.photos_collection.update_one({"face_id": face_id}, update, upsert=False)
            return True
        except Exception as e:
            print(f"✗ Error updating embedding for {face_id}: {e}")
            return False
    
    def get_face_embedding(self, face_id: str) -> Optional[Dict]:
        """
        Retrieve a face embedding by face_id.
        
        Args:
            face_id: Unique identifier for the face
            
        Returns:
            Dictionary containing face data or None if not found
        """
        try:
            face_doc = self.photos_collection.find_one({"face_id": face_id})
            if face_doc:
                # Convert embedding back to numpy array
                face_doc['embedding_vector'] = np.array(face_doc['embedding_vector'])
                return face_doc
            return None
        except Exception as e:
            print(f"✗ Error retrieving face embedding {face_id}: {e}")
            return None
    
    def get_faces_by_photo(self, photo_id: str) -> List[Dict]:
        """
        Retrieve all faces from a specific photo.
        
        Args:
            photo_id: ID of the photo
            
        Returns:
            List of face documents
        """
        try:
            faces = list(self.photos_collection.find({"photo_id": photo_id}))
            # Convert embeddings back to numpy arrays
            for face in faces:
                if 'embedding_vector' in face:
                    face['embedding_vector'] = np.array(face['embedding_vector'])
            return faces
        except Exception as e:
            print(f"✗ Error retrieving faces for photo {photo_id}: {e}")
            return []

    # Removed: get_photo_by_id method

    def get_all_faces(self) -> List[Dict]:
        """
        Retrieve all face embeddings from the database.
        
        Returns:
            List of all face documents
        """
        try:
            faces = list(self.photos_collection.find())
            # Convert embeddings back to numpy arrays
            for face in faces:
                if 'embedding_vector' in face:
                    face['embedding_vector'] = np.array(face['embedding_vector'])
            return faces
        except Exception as e:
            print(f"✗ Error retrieving all faces: {e}")
            return []
    
    # 🆕 New method to filter faces by event and photographer 
    def get_faces_by_event_and_photographer(self, event_id: str, photographer_id: str) -> List[Dict]:
        """
        Retrieves all face documents that are associated with a specific event and photographer.
        """
        try:
            query = {
                "event_id": event_id,
                "photographer_id": photographer_id
            }
            faces = list(self.photos_collection.find(query))
            # Convert embeddings back to numpy arrays
            for face in faces:
                if 'embedding_vector' in face:
                    face['embedding_vector'] = np.array(face['embedding_vector'])
            print(f"✓ Found {len(faces)} faces for event '{event_id}' and photographer '{photographer_id}'")
            return faces
        except Exception as e:
            print(f"✗ Error retrieving faces by event/photographer: {e}")
            return []

    # 🆕 New method: filter faces by event only
    def get_faces_by_event(self, event_id: str) -> List[Dict]:
        try:
            query = {"event_id": event_id}
            faces = list(self.photos_collection.find(query))
            for face in faces:
                if 'embedding_vector' in face:
                    face['embedding_vector'] = np.array(face['embedding_vector'])
            print(f"✓ Found {len(faces)} faces for event '{event_id}'")
            return faces
        except Exception as e:
            print(f"✗ Error retrieving faces by event: {e}")
            return []

    def find_similar_faces(self, target_embedding: np.ndarray, 
                            similarity_threshold: float = 0.6,
                            metric: str = "cosine",
                            limit: int = 10) -> List[Dict]:
        """
        Find faces similar to a target embedding.
        
        Args:
            target_embedding: Target embedding vector
            similarity_threshold: Threshold for similarity
            metric: Similarity metric ('cosine', 'euclidean')
            limit: Maximum number of results to return
            
        Returns:
            List of similar faces with similarity scores
        """
        try:
            all_faces = self.get_all_faces()
            similar_faces = []
            
            for face in all_faces:
                if 'embedding_vector' in face and face['embedding_vector'] is not None:
                    face_embedding = face['embedding_vector']
                    similarity = self._compute_similarity(target_embedding, face_embedding, metric)
                    
                    # Apply threshold
                    if metric == "cosine" and similarity >= similarity_threshold:
                        similar_faces.append({
                            'face_id': face['face_id'],
                            'photo_id': face['photo_id'],
                            'similarity_score': similarity,
                            'crop_path': face['crop_path'],
                            # Include fields for output matching guest_search.py
                            'event_id': face.get('event_id'),
                            'photographer_id': face.get('photographer_id'),
                            'photo_path': face.get('photo_path'),
                            'public_url': face.get('public_url'),
                        })
                    elif metric != "cosine" and similarity <= similarity_threshold:
                        similar_faces.append({
                            'face_id': face['face_id'],
                            'photo_id': face['photo_id'],
                            'similarity_score': similarity,
                            'crop_path': face['crop_path'],
                            # Include fields for output matching guest_search.py
                            'event_id': face.get('event_id'),
                            'photographer_id': face.get('photographer_id'),
                            'photo_path': face.get('photo_path'),
                            'public_url': face.get('public_url'),
                        })
            
            # Sort by similarity
            if metric == "cosine":
                similar_faces.sort(key=lambda x: x['similarity_score'], reverse=True)
            else:
                similar_faces.sort(key=lambda x: x['similarity_score'])
            
            return similar_faces[:limit]
            
        except Exception as e:
            print(f"✗ Error finding similar faces: {e}")
            return []
    
    def _compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                            metric: str = "cosine") -> float:
        """
        Compute similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric
            
        Returns:
            Similarity score
        """
        if metric == "cosine":
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            if norm1 == 0 or norm2 == 0: # Handle zero vectors
                return 0.0
            return dot_product / (norm1 * norm2)
        elif metric == "euclidean":
            return np.linalg.norm(embedding1 - embedding2)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def get_database_stats(self) -> Dict:
        """
        Get statistics about the database.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            total_faces = self.photos_collection.count_documents({})
            # Removed: total_photos = self.photos_collection.count_documents({})
            
            # Get unique photo IDs from faces collection
            unique_photos = len(self.photos_collection.distinct("photo_id"))
            
            # Get average embedding dimension
            pipeline = [
                {"$group": {"_id": None, "avg_dim": {"$avg": "$embedding_dimension"}}}
            ]
            avg_dim_result = list(self.photos_collection.aggregate(pipeline))
            avg_dimension = avg_dim_result[0]['avg_dim'] if avg_dim_result else 0
            
            return {
                "total_faces": total_faces,
                # Removed: "total_photos": total_photos,
                "unique_photos_with_faces": unique_photos,
                "average_embedding_dimension": avg_dimension,
                "database_name": self.database_name
            }
        except Exception as e:
            print(f"✗ Error getting database stats: {e}")
            return {}
    
    def close_connection(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            print("✓ MongoDB connection closed")


# Example usage
if __name__ == "__main__":
    # Test MongoDB storage
    storage = MongoDBStorage()
    
    # Test storing a sample face embedding
    sample_face = {
        "face_id": "test_face_001",
        "photo_id": "test_photo_001",
        "event_id": "event_A",  # Added for testing
        "photographer_id": "photo_B", # Added for testing
        "embedding_vector": np.random.rand(512),
        "crop_path": "data/cropped_faces/test_face_001.jpg",
        "coordinates": {"x": 100, "y": 100, "width": 150, "height": 150},
        "confidence": 0.95,
        "model_name": "ArcFace",
        # Adding photo metadata fields that are now merged into the faces collection
        "photo_path": "data/raw_photos/test_photo_001.jpg",
        "public_url": "http://example.com/photos/test_photo_001.jpg"
    }
    
    # Store the face embedding
    success = storage.store_face_embedding(sample_face)
    print(f"Storage success: {success}")

    # Test the new filter method
    print("\nTesting new filtering method:")
    filtered_faces = storage.get_faces_by_event_and_photographer(
        event_id="event_A", 
        photographer_id="photo_B"
    )
    print(f"Retrieved {len(filtered_faces)} faces matching the filter.")
    
    # Get database stats
    stats = storage.get_database_stats()
    print(f"\nDatabase stats: {stats}")
    
    # Close connection
    storage.close_connection()