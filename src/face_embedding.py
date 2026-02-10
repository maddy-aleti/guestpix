"""
Step 3: Face Embedding & MongoDB Storage
Converts cropped face images into numerical vectors and stores them in MongoDB.
"""

import os
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from deepface import DeepFace
import yaml
from tqdm import tqdm
import json
import pickle
import requests
import tempfile
from .store_embeddings import MongoDBStorage
from .cloudinary_service import CloudinaryService


class FaceEmbedding:
    """
    Generates numerical embeddings (vectors) for detected faces.
    
    What it does:
    - Converts each cropped face image into a numerical vector
    - Uses pre-trained models (ArcFace, FaceNet, etc.) to extract features
    - Generates unique "face fingerprints" that represent facial characteristics
    - Faces of the same person will have very similar embeddings
    
    Expected output:
    - Collection of embeddings (numerical vectors) for each face
    - Each embedding linked to a specific face and source photo
    - Data stored in JSON and pickle formats for further processing
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the face embedding system with MongoDB integration."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.model_name = self.config['face_embedding']['model_name']
        self.enforce_detection = self.config['face_embedding']['enforce_detection']
        self.align = self.config['face_embedding']['align']
        self.output_path = Path(self.config['face_embedding']['output_path'])
        
        # Create output directory (for legacy support)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Store embedding results (for legacy support)
        self.embedding_results = {}
        
        # Available models for comparison
        self.available_models = ["ArcFace", "FaceNet", "VGG-Face", "OpenFace", "DeepID", "Dlib"]
        
        # Initialize MongoDB storage
        try:
            self.mongodb_storage = MongoDBStorage(config_path)
            print("✓ MongoDB storage initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize MongoDB storage: {e}")
            self.mongodb_storage = None
    
    def _download_image_from_url(self, url: str) -> Optional[str]:
        """
        Download image from URL and save to temporary file.
        
        Args:
            url: Cloudinary or other image URL
            
        Returns:
            Path to temporary file or None if download failed
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(response.content)
                return tmp.name
        except Exception as e:
            print(f"✗ Failed to download image from {url}: {e}")
            return None
    
    def _get_image_path(self, img_path: str) -> Optional[str]:
        """
        Handle both local file paths and URLs.
        If URL, download first and return temp file path.
        
        Args:
            img_path: Local file path or URL
            
        Returns:
            Path to file (local or temp)
        """
        if img_path.startswith('http://') or img_path.startswith('https://'):
            return self._download_image_from_url(img_path)
        return img_path
    
    def _preprocess_arcface(self, img_path: str) -> Optional[np.ndarray]:
        """
        Load image, ensure RGB, resize to 112x112, and normalize per ArcFace.
        Returns float32 numpy array ready for model input.
        Handles both local paths and Cloudinary URLs.
        """
        try:
            import cv2
            # Handle URL if needed
            actual_path = self._get_image_path(img_path)
            if actual_path is None:
                return None
            
            img_bgr = cv2.imread(actual_path)
            if img_bgr is None:
                return None
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (112, 112))
            img = img_rgb.astype('float32')
            img = (img - 127.5) / 128.0
            return img
        except Exception:
            return None

    def _preprocess_arcface_from_array(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Take an RGB uint8 array and apply ArcFace preprocessing.
        """
        import cv2
        resized = cv2.resize(img_rgb, (112, 112))
        img = resized.astype('float32')
        img = (img - 127.5) / 128.0
        return img

    def _generate_embedding_from_array(self, img_array: np.ndarray) -> Optional[np.ndarray]:
        """Call DeepFace.represent on a preprocessed numpy array and return vector."""
        embedding = DeepFace.represent(
            img_path=img_array,
            model_name=self.model_name,
            detector_backend="skip",
            enforce_detection=False,
            align=False
        )
        if isinstance(embedding, list):
            return np.array(embedding[0]['embedding'])
        if isinstance(embedding, dict):
            return np.array(embedding['embedding'])
        return np.array(embedding)

    def _augment_rgb_images(self, img_bgr: np.ndarray) -> List[np.ndarray]:
        """
        Create simple robustness augmentations (flip and slight rotations) and return RGB arrays.
        """
        import cv2
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        aug_list = [rgb]
        # Horizontal flip
        aug_list.append(cv2.flip(rgb, 1))
        # Small rotations around center
        h, w = rgb.shape[:2]
        center = (w // 2, h // 2)
        for angle in (-7, +7):
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            aug_list.append(rotated)
        return aug_list

    def generate_embedding(self, face_image_path: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a single face image.
        Handles both local paths and Cloudinary URLs.
        
        Args:
            face_image_path: Path to the cropped face image or URL
            
        Returns:
            Numerical embedding vector (numpy array) or None if failed
        """
        try:
            # Handle URL if needed
            actual_path = self._get_image_path(face_image_path)
            if actual_path is None:
                return None
            
            # Standardize preprocessing for ArcFace
            if self.model_name.lower() == "arcface":
                # Optional robustness via augmentation
                augmentation_cfg = self.config['face_embedding'].get('augmentation', {})
                use_augmentation = bool(augmentation_cfg.get('enabled', False))
                try:
                    import cv2
                    img_bgr = cv2.imread(actual_path)
                except Exception:
                    img_bgr = None

                if use_augmentation and img_bgr is not None:
                    rgb_variants = self._augment_rgb_images(img_bgr)
                    vectors: List[np.ndarray] = []
                    for rgb_img in rgb_variants:
                        pre = self._preprocess_arcface_from_array(rgb_img)
                        vec = self._generate_embedding_from_array(pre)
                        if vec is not None:
                            vectors.append(vec)
                    if not vectors:
                        return None
                    embedding = np.mean(np.stack(vectors, axis=0), axis=0)
                else:
                    preprocessed = self._preprocess_arcface(actual_path)
                    if preprocessed is None:
                        return None
                    embedding = self._generate_embedding_from_array(preprocessed)
            else:
                # Fallback to DeepFace internal pipeline for other models
                embedding = DeepFace.represent(
                    img_path=actual_path,
                    model_name=self.model_name,
                    enforce_detection=self.enforce_detection,
                    align=self.align
                )
            
            # Convert to numpy array if it's not already
            if isinstance(embedding, list):
                embedding = np.array(embedding[0]['embedding'])
            elif isinstance(embedding, dict):
                embedding = np.array(embedding['embedding'])
            else:
                embedding = np.array(embedding)

            # L2-normalize embedding for consistent cosine similarity
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
            
            return embedding
            
        except Exception as e:
            print(f"Error generating embedding for {face_image_path}: {str(e)}")
            return None
    
    def process_faces(self, detection_results: Dict) -> Dict:
        """
        Process all detected faces to generate embeddings and store in MongoDB.
        
        Args:
            detection_results: Results from face detection step
            
        Returns:
            Dictionary with embeddings for all faces
        """
        all_embeddings = []
        total_faces = sum(len(result['faces']) for result in detection_results.values())
        
        print(f"Starting face embedding generation for {total_faces} faces...")
        print(f"Using model: {self.model_name}")
        
        # Track MongoDB storage success
        mongodb_success_count = 0
        mongodb_failure_count = 0
        
        for photo_id, photo_result in tqdm(detection_results.items(), desc="Generating embeddings"):
            photo_embeddings = []
            
            # No longer storing/updating separate photo metadata in MongoDB; merged into faces
            
            for face in photo_result['faces']:
                face_id = face['face_id']
                crop_path = face['crop_path']
                
                # Generate embedding for this face
                embedding = self.generate_embedding(crop_path)
                
                if embedding is not None:
                    # Store embedding information
                    embedding_info = {
                        'face_id': face_id,
                        'photo_id': photo_id,
                        'embedding_vector': embedding.tolist(),  # Convert to list for JSON serialization
                        'embedding_dimension': len(embedding),
                        'model_name': self.model_name,
                        'crop_path': crop_path,
                        'coordinates': face['coordinates'],
                        'confidence': face['confidence']
                    }
                    
                    photo_embeddings.append(embedding_info)
                    all_embeddings.append(embedding_info)
                    
                    # Store/update embedding in MongoDB on the existing face doc
                    if self.mongodb_storage is not None:
                        if self.mongodb_storage.update_face_embedding(face_id=face_id, embedding_vector=embedding, model_name=self.model_name):
                            mongodb_success_count += 1
                        else:
                            mongodb_failure_count += 1
                    
                    print(f"✓ Generated embedding for {face_id} (dim: {len(embedding)})")
                else:
                    print(f"✗ Failed to generate embedding for {face_id}")
            
            # Store results for this photo (legacy support)
            self.embedding_results[photo_id] = {
                'photo_path': photo_result['photo_path'],
                'faces_processed': len(photo_embeddings),
                'embeddings': photo_embeddings
            }
        
        # Save embedding results (legacy support)
        self.save_embedding_results()
        
        print(f"\nFace embedding completed!")
        print(f"Total faces processed: {len(all_embeddings)}")
        print(f"Successful embeddings: {len(all_embeddings)}")
        print(f"Model used: {self.model_name}")
        
        if self.mongodb_storage is not None:
            print(f"MongoDB storage: {mongodb_success_count} successful, {mongodb_failure_count} failed")
            
            # Get database statistics
            stats = self.mongodb_storage.get_database_stats()
            print(f"Database stats: {stats}")
        
        return self.embedding_results
    
    def save_embedding_results(self):
        """Save embedding results to JSON and pickle files."""
        # Save as JSON (human-readable, but vectors as lists)
        json_path = self.output_path / "face_embeddings.json"
        with open(json_path, 'w') as f:
            json.dump(self.embedding_results, f, indent=2)
        
        # Save as pickle (preserves numpy arrays exactly)
        pickle_path = self.output_path / "face_embeddings.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(self.embedding_results, f)
        
        print(f"Embedding results saved to:")
        print(f"  JSON: {json_path}")
        print(f"  Pickle: {pickle_path}")
    
    def get_embedding_statistics(self) -> Dict:
        """Get statistics about generated embeddings."""
        total_embeddings = sum(len(result['embeddings']) for result in self.embedding_results.values())
        
        if total_embeddings > 0:
            dimensions = []
            confidences = []
            
            for result in self.embedding_results.values():
                for embedding in result['embeddings']:
                    dimensions.append(embedding['embedding_dimension'])
                    confidences.append(embedding['confidence'])
            
            avg_dimension = np.mean(dimensions)
            avg_confidence = np.mean(confidences)
        else:
            avg_dimension = 0
            avg_confidence = 0.0
        
        return {
            'total_embeddings': total_embeddings,
            'model_name': self.model_name,
            'average_embedding_dimension': avg_dimension,
            'average_confidence': avg_confidence,
            'photos_processed': len(self.embedding_results),
            'available_models': self.available_models
        }
    
    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray, 
                          metric: str = "cosine") -> float:
        """
        Compare two embeddings using specified similarity metric.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ('cosine', 'euclidean', 'manhattan')
            
        Returns:
            Similarity score (higher = more similar for cosine, lower = more similar for distance metrics)
        """
        if metric == "cosine":
            # Cosine similarity (higher = more similar)
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            return dot_product / (norm1 * norm2)
        
        elif metric == "euclidean":
            # Euclidean distance (lower = more similar)
            return np.linalg.norm(embedding1 - embedding2)
        
        elif metric == "manhattan":
            # Manhattan distance (lower = more similar)
            return np.sum(np.abs(embedding1 - embedding2))
        
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def find_similar_faces(self, target_face_id: str, similarity_threshold: float = 0.6,
                          metric: str = "cosine") -> List[Dict]:
        """
        Find faces similar to a target face.
        
        Args:
            target_face_id: ID of the target face
            similarity_threshold: Threshold for considering faces similar
            metric: Similarity metric to use
            
        Returns:
            List of similar faces with similarity scores
        """
        # Find target embedding
        target_embedding = None
        for result in self.embedding_results.values():
            for embedding in result['embeddings']:
                if embedding['face_id'] == target_face_id:
                    target_embedding = np.array(embedding['embedding_vector'])
                    break
            if target_embedding is not None:
                break
        
        if target_embedding is None:
            print(f"Target face {target_face_id} not found")
            return []
        
        similar_faces = []
        
        # Compare with all other faces
        for result in self.embedding_results.values():
            for embedding in result['embeddings']:
                if embedding['face_id'] == target_face_id:
                    continue  # Skip self-comparison
                
                current_embedding = np.array(embedding['embedding_vector'])
                similarity = self.compare_embeddings(target_embedding, current_embedding, metric)
                
                # Apply threshold based on metric
                if metric == "cosine":
                    if similarity >= similarity_threshold:
                        similar_faces.append({
                            'face_id': embedding['face_id'],
                            'photo_id': embedding['photo_id'],
                            'similarity_score': similarity,
                            'crop_path': embedding['crop_path']
                        })
                else:  # Distance metrics
                    if similarity <= similarity_threshold:
                        similar_faces.append({
                            'face_id': embedding['face_id'],
                            'photo_id': embedding['photo_id'],
                            'similarity_score': similarity,
                            'crop_path': embedding['crop_path']
                        })
        
        # Sort by similarity (descending for cosine, ascending for distance metrics)
        if metric == "cosine":
            similar_faces.sort(key=lambda x: x['similarity_score'], reverse=True)
        else:
            similar_faces.sort(key=lambda x: x['similarity_score'])
        
        return similar_faces
    
    def export_embeddings_for_matching(self) -> List[Dict]:
        """
        Export embeddings in format suitable for face matching step.
        Now retrieves from MongoDB if available, falls back to local storage.
        
        Returns:
            List of embeddings with all necessary information for matching
        """
        # Try to get embeddings from MongoDB first
        if self.mongodb_storage is not None:
            try:
                all_faces = self.mongodb_storage.get_all_faces()
                if all_faces:
                    print(f"✓ Retrieved {len(all_faces)} embeddings from MongoDB")
                    return all_faces
                else:
                    print("⚠ No embeddings found in MongoDB, falling back to local storage")
            except Exception as e:
                print(f"⚠ Error retrieving from MongoDB: {e}, falling back to local storage")
        
        # Fallback to local storage (legacy)
        all_embeddings = []
        
        for result in self.embedding_results.values():
            for embedding in result['embeddings']:
                all_embeddings.append({
                    'face_id': embedding['face_id'],
                    'photo_id': embedding['photo_id'],
                    'embedding_vector': np.array(embedding['embedding_vector']),
                    'crop_path': embedding['crop_path'],
                    'coordinates': embedding['coordinates']
                })
        
        return all_embeddings
    
    def get_embeddings_from_mongodb(self) -> List[Dict]:
        """
        Retrieve all embeddings directly from MongoDB.
        
        Returns:
            List of embeddings from MongoDB
        """
        if self.mongodb_storage is None:
            print("✗ MongoDB storage not available")
            return []
        
        try:
            all_faces = self.mongodb_storage.get_all_faces()
            print(f"✓ Retrieved {len(all_faces)} embeddings from MongoDB")
            return all_faces
        except Exception as e:
            print(f"✗ Error retrieving embeddings from MongoDB: {e}")
            return []


# Example usage and testing
if __name__ == "__main__":
    # Test the face embedding system
    embedder = FaceEmbedding()
    
    # Load sample detection results (this would come from FaceDetection)
    sample_detection_results = {
        "sample_photo_1": {
            "photo_path": "data/raw_photos/sample1.jpg",
            "faces": [
                {
                    "face_id": "sample_photo_1_face_000",
                    "crop_path": "data/cropped_faces/sample_photo_1_face_000.jpg",
                    "coordinates": {"x": 100, "y": 100, "width": 150, "height": 150},
                    "confidence": 0.95
                }
            ]
        }
    }
    
    # Check if sample face exists
    if os.path.exists("data/cropped_faces/sample_photo_1_face_000.jpg"):
        results = embedder.process_faces(sample_detection_results)
        stats = embedder.get_embedding_statistics()
        print(f"\nEmbedding Statistics: {stats}")
    else:
        print("Sample face not found. Please run face detection first.") 