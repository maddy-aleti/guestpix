"""
Step 2: Face Detection
Locates and crops faces from photos using DeepFace library.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from deepface import DeepFace
import yaml
from tqdm import tqdm
import json


class FaceDetection:
    """
    Detects faces in photos and crops them for further processing.
    
    What it does:
    - Scans each stored photo to locate faces
    - Draws bounding boxes around detected faces
    - Crops each face into individual images
    - Links cropped faces to original photo IDs and coordinates
    
    Expected output:
    - Collection of cropped face images
    - Metadata linking faces to source photos and coordinates
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the face detection system."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.detector_backend = self.config['face_detection']['detector_backend']
        self.confidence_threshold = self.config['face_detection']['confidence_threshold']
        self.min_face_size = self.config['face_detection']['min_face_size']
        self.output_path = Path(self.config['face_detection']['output_path'])
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        # Store face detection results
        self.detection_results = {}
    
    def detect_faces_in_photo(self, photo_path: str, photo_id: str) -> List[Dict]:
        """
        Detect faces in a single photo.
        Tries multiple DeepFace backends if needed.
        """
        backends_to_try = [self.detector_backend, "mtcnn", "retinaface", "ssd", "opencv"]
        tried = set()
        for backend in backends_to_try:
            if backend in tried:
                continue
            tried.add(backend)
            try:
                faces = DeepFace.extract_faces(
                    img_path=photo_path,
                    detector_backend=backend,
                    enforce_detection=False,
                    align=True
                )
                detected_faces = []
                for i, face_data in enumerate(faces):
                    facial_area = face_data['facial_area']
                    confidence = face_data.get('confidence')
                    if confidence is None:
                        confidence = 1.0
                    if confidence < self.confidence_threshold:
                        continue
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    if w < self.min_face_size or h < self.min_face_size:
                        continue
                    # Prefer the aligned face returned by DeepFace when available
                    aligned_face = face_data.get('face')
                    if aligned_face is not None:
                        # DeepFace returns RGB float image in range [0,1] or [0,255] depending on version
                        face_img = aligned_face
                        if face_img.dtype != np.uint8:
                            face_img = (np.clip(face_img, 0, 1) * 255).astype(np.uint8)
                        # Ensure size 112x112 as expected by ArcFace
                        face_img = cv2.resize(face_img, (112, 112))
                        # Convert RGB -> BGR for cv2.imwrite
                        face_crop = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                    else:
                        img = cv2.imread(photo_path)
                        face_crop = img[y:y+h, x:x+w]
                        face_crop = cv2.resize(face_crop, (112, 112))
                    face_id = f"{photo_id}_face_{i:03d}"
                    face_filename = f"{face_id}.jpg"
                    face_path = self.output_path / face_filename
                    cv2.imwrite(str(face_path), face_crop)
                    face_info = {
                        'face_id': face_id,
                        'photo_id': photo_id,
                        'coordinates': {
                            'x': x, 'y': y, 'width': w, 'height': h
                        },
                        'confidence': confidence,
                        'face_size': (w, h),
                        'crop_path': str(face_path),
                        'detector_backend': backend
                    }
                    detected_faces.append(face_info)
                if detected_faces:
                    return detected_faces
            except Exception as e:
                continue
        print(f"Error detecting faces in {photo_path}: No faces found with any backend.")
        return []

    def process_photos(self, photos_metadata: Dict) -> Dict:
        """
        Process all photos to detect and crop faces.
        
        Args:
            photos_metadata: Dictionary containing photo information
            
        Returns:
            Dictionary with detection results for all photos
        """
        all_faces = []
        total_photos = len(photos_metadata)
        
        print(f"Starting face detection for {total_photos} photos...")
        
        # Optional Mongo integration
        mongodb_storage = None
        try:
            from .store_embeddings import MongoDBStorage
            mongodb_storage = MongoDBStorage()
        except Exception:
            mongodb_storage = None

        for photo_id, photo_info in tqdm(photos_metadata.items(), desc="Detecting faces"):
            photo_path = photo_info['stored_path']
            
            # Detect faces in this photo
            faces = self.detect_faces_in_photo(photo_path, photo_id)
            
            if faces:
                all_faces.extend(faces)
                print(f"✓ Photo {photo_id}: Found {len(faces)} faces")
            else:
                print(f"✗ Photo {photo_id}: No faces detected")
            
            # Store results
            self.detection_results[photo_id] = {
                'photo_path': photo_path,
                'faces_detected': len(faces),
                'faces': faces
            }

            # If Mongo is available, insert Face docs with merged photo metadata
            if mongodb_storage is not None and faces:
                event_id = photo_info.get('event_id') or self._infer_event_from_path(photo_path)
                photographer_id = photo_info.get('photographer_id') or self._infer_photographer_from_path(photo_path)
                public_url = photo_info.get('public_url')
                try:
                    file_size = os.path.getsize(photo_path) if os.path.exists(photo_path) else 0
                except Exception:
                    file_size = 0
                dimensions = {}
                for f in faces:
                    bbox = {
                        'x': f['coordinates']['x'],
                        'y': f['coordinates']['y'],
                        'width': f['coordinates']['width'],
                        'height': f['coordinates']['height']
                    }
                    try:
                        mongodb_storage.insert_detected_face(
                            face_id=f['face_id'],
                            photo_id=photo_id,
                            event_id=event_id or '',
                            photographer_id=photographer_id or '',
                            bounding_box=bbox,
                            crop_path=f['crop_path'],
                            confidence=float(f.get('confidence', 0.0)),
                            detector_backend=f.get('detector_backend'),
                            photo_path=photo_path,
                            file_size=file_size,
                            dimensions=dimensions,
                            public_url=public_url,
                            face_count=len(faces)
                        )
                    except Exception:
                        pass
        
        # Save detection results
        self.save_detection_results()
        
        print(f"\nFace detection completed!")
        print(f"Total photos processed: {total_photos}")
        print(f"Total faces detected: {len(all_faces)}")
        print(f"Average faces per photo: {len(all_faces) / total_photos:.2f}")
        
        return self.detection_results

    def _infer_event_from_path(self, photo_path: str) -> Optional[str]:
        try:
            p = Path(photo_path)
            return p.parent.parent.name
        except Exception:
            return None

    def _infer_photographer_from_path(self, photo_path: str) -> Optional[str]:
        try:
            p = Path(photo_path)
            return p.parent.name
        except Exception:
            return None
    
    def save_detection_results(self):
        """Save face detection results to JSON file."""
        results_path = self.output_path / "face_detection_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.detection_results, f, indent=2)
        
        print(f"Detection results saved to: {results_path}")
    
    def get_face_statistics(self) -> Dict:
        """Get statistics about detected faces."""
        total_faces = sum(len(result['faces']) for result in self.detection_results.values())
        photos_with_faces = sum(1 for result in self.detection_results.values() if result['faces'])
        
        if total_faces > 0:
            face_sizes = []
            confidences = []
            
            for result in self.detection_results.values():
                for face in result['faces']:
                    face_sizes.append(face['face_size'])
                    confidences.append(face['confidence'])
            
            avg_face_size = np.mean(face_sizes, axis=0)
            avg_confidence = np.mean(confidences)
        else:
            avg_face_size = (0, 0)
            avg_confidence = 0.0
        
        return {
            'total_faces_detected': total_faces,
            'photos_with_faces': photos_with_faces,
            'photos_without_faces': len(self.detection_results) - photos_with_faces,
            'average_faces_per_photo': total_faces / len(self.detection_results) if self.detection_results else 0,
            'average_face_size': avg_face_size.tolist() if hasattr(avg_face_size, 'tolist') else list(avg_face_size),
            'average_confidence': avg_confidence,
            'detector_backend': self.detector_backend
        }
    
    def visualize_detections(self, photo_id: str, save_path: Optional[str] = None):
        """
        Visualize face detections on a photo.
        
        Args:
            photo_id: ID of the photo to visualize
            save_path: Optional path to save the visualization
        """
        if photo_id not in self.detection_results:
            print(f"Photo ID {photo_id} not found in detection results")
            return
        
        photo_info = self.detection_results[photo_id]
        photo_path = photo_info['photo_path']
        
        # Load image
        img = cv2.imread(photo_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw bounding boxes
        for face in photo_info['faces']:
            coords = face['coordinates']
            x, y, w, h = coords['x'], coords['y'], coords['width'], coords['height']
            
            # Draw rectangle
            cv2.rectangle(img_rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add face ID label
            cv2.putText(img_rgb, face['face_id'], (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Save or display
        if save_path:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            plt.imshow(img_rgb)
            plt.title(f"Face Detections in Photo {photo_id}")
            plt.axis('off')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Visualization saved to: {save_path}")
        else:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 8))
            plt.imshow(img_rgb)
            plt.title(f"Face Detections in Photo {photo_id}")
            plt.axis('off')
            plt.show()


# Example usage and testing
if __name__ == "__main__":
    # Test the face detection system
    detector = FaceDetection()
    
    # Load sample photos metadata (this would come from PhotoIngestion)
    sample_metadata = {
        "sample_photo_1": {
            "stored_path": "data/raw_photos/sample1.jpg",
            "filename": "sample1.jpg"
        }
    }
    
    # Check if sample photo exists
    if os.path.exists("data/raw_photos/sample1.jpg"):
        results = detector.process_photos(sample_metadata)
        stats = detector.get_face_statistics()
        print(f"\nDetection Statistics: {stats}")
    else:
        print("Sample photo not found. Please run photo ingestion first.")