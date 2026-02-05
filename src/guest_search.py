import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

# Local imports
from .face_detection import FaceDetection
from .face_embedding import FaceEmbedding
from .store_embeddings import MongoDBStorage


def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Computes the cosine similarity between two vectors.
    """
    if vec1.ndim > 1:
        vec1 = vec1.flatten()
    if vec2.ndim > 1:
        vec2 = vec2.flatten()
    denom = float(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if denom == 0.0:
        return 0.0
    return float(np.dot(vec1, vec2) / denom)


class GuestSearchMongoDB:
    """
    Guest search against MongoDB faces collection for a single input photo.
    """

    def __init__(self, config_path: str = "config/config.yaml") -> None:
        self.face_detection = FaceDetection(config_path)
        self.face_embedding = FaceEmbedding(config_path)

        # Mongo storage (read-only usage for search)
        self.storage = None
        try:
            self.storage = MongoDBStorage(config_path)
        except Exception as e:
            print(f"✗ Failed to connect to MongoDB: {e}")
            raise

        # Output directories
        self.output_dir = Path("output_user_photos")
        self.output_dir.mkdir(exist_ok=True)
        self.user_output_dir = self.output_dir / "user_input"
        self.user_output_dir.mkdir(exist_ok=True)

    def process_photo(self, user_photo_path: str,
                      event_id: str,
                      photographer_id: str,
                      similarity_threshold: float = 0.6,
                      metric: str = "cosine",
                      limit: Optional[int] = None) -> bool:
        print(f"\n📷 Processing photo: {user_photo_path}")
        print(f"Filtering by Event ID: {event_id} and Photographer ID: {photographer_id}")

        # Detect faces
        user_faces = self.face_detection.detect_faces_in_photo(user_photo_path, photo_id="user_guest")
        if not user_faces:
            print("❌ No face detected in the provided photo.")
            return False

        print(f"✓ Detected {len(user_faces)} face(s)")

        # For each detected face, generate embedding and search
        for idx, user_face in enumerate(user_faces, start=1):
            user_crop_path = user_face["crop_path"]

            # Save cropped face copy for reference
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            user_crop_name = f"user_crop_{timestamp}_{idx:03d}.jpg"
            user_saved_crop_path = self.user_output_dir / user_crop_name
            try:
                shutil.copy(user_crop_path, user_saved_crop_path)
            except Exception:
                # Best-effort; continue if copy fails
                user_saved_crop_path = Path(user_crop_path)
            print(f"✓ Saved cropped face to: {user_saved_crop_path}")

            # Prepare output directory to collect all matched images for this face
            matches_root = self.output_dir / "matched_results"
            matches_root.mkdir(exist_ok=True)
            face_matches_dir = matches_root / f"{timestamp}_{idx:03d}"
            face_matches_dir.mkdir(parents=True, exist_ok=True)
            # Save a copy of the user crop inside the folder for reference
            try:
                shutil.copy(str(user_saved_crop_path), face_matches_dir / f"query_crop_{timestamp}_{idx:03d}.jpg")
            except Exception:
                pass

            # Generate embedding for the cropped face
            embedding_vector = self.face_embedding.generate_embedding(user_crop_path)
            if embedding_vector is None:
                print("✗ Failed to generate embedding for detected face; skipping")
                continue

            print("✓ Generated query embedding")

            # Perform vector search in MongoDB with initial filtering
            matches = self._find_similar_faces_client_side(
                target_embedding=embedding_vector,
                event_id=event_id,
                photographer_id=photographer_id,
                similarity_threshold=similarity_threshold,
                metric=metric,
                limit=limit
            )

            # Print results and save matched photos
            if not matches:
                print("🔍 No matches above threshold found.")
            else:
                print(f"🔍 Top {len(matches)} match(es):")
                for rank, m in enumerate(matches, start=1):
                    score_label = "similarity" if metric == "cosine" else "distance"
                    
                    # Get photo path
                    photo_path = m.get('photo_path')
                    
                    print(f"  {rank}. similarity={m['similarity_score']:.4f}, event_id={m.get('event_id')}, photographer_id={m.get('photographer_id')}, face_id={m.get('face_id')}")
                    print(f"     photo_path: {photo_path}")
                    print(f"     public_url: {m.get('public_url') or 'N/A'}")

                    # Save matched image into the face-specific output folder
                    if photo_path:
                        try:
                            src_path = Path(str(photo_path))
                            ext = src_path.suffix if src_path.suffix else ".jpg"
                            dest_name = f"rank_{rank:03d}_photo_{m['photo_id']}{ext}"
                            shutil.copy(str(src_path), face_matches_dir / dest_name)
                        except Exception as e:
                            print(f"✗ Failed to copy matched image: {e}")

                # Summary of saved outputs for this face
                print(f"📁 Saved {len(matches)} matched image(s) to: {face_matches_dir}")

        return True

    def search_photo(self, user_photo_path: str,
                      event_id: str,
                      photographer_id: str,
                      similarity_threshold: float = 0.6,
                      metric: str = "cosine",
                      limit: Optional[int] = 10) -> Dict:
        """
        API-friendly search that returns structured data instead of printing.

        Returns a dict containing:
        - query_crops: list of saved crop paths
        - matches: list of match dicts with similarity score and photo details
        - output_dir: directory where matches were saved
        """
        result: Dict = {
            "query_crops": [],
            "matches": [],
            "output_dir": None,
        }

        # Detect faces
        user_faces = self.face_detection.detect_faces_in_photo(user_photo_path, photo_id="user_guest")
        if not user_faces:
            return result

        # Prepare directories
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        matches_root = self.output_dir / "matched_results"
        matches_root.mkdir(exist_ok=True)
        face_matches_dir = matches_root / f"{timestamp}_001"
        face_matches_dir.mkdir(parents=True, exist_ok=True)
        result["output_dir"] = str(face_matches_dir)

        aggregated_matches: List[Dict] = []

        for idx, user_face in enumerate(user_faces, start=1):
            user_crop_path = user_face["crop_path"]

            # Save cropped face copy for reference
            user_crop_name = f"user_crop_{timestamp}_{idx:03d}.jpg"
            user_saved_crop_path = self.user_output_dir / user_crop_name
            try:
                shutil.copy(user_crop_path, user_saved_crop_path)
            except Exception:
                user_saved_crop_path = Path(user_crop_path)
            result["query_crops"].append(str(user_saved_crop_path))

            try:
                shutil.copy(str(user_saved_crop_path), face_matches_dir / f"query_crop_{timestamp}_{idx:03d}.jpg")
            except Exception:
                pass

            # Generate embedding for the cropped face
            embedding_vector = self.face_embedding.generate_embedding(user_crop_path)
            if embedding_vector is None:
                continue

            # Perform vector search in MongoDB with initial filtering
            matches = self._find_similar_faces_client_side(
                target_embedding=embedding_vector,
                event_id=event_id,
                photographer_id=photographer_id,
                similarity_threshold=similarity_threshold,
                metric=metric,
                limit=limit,
            )

            # Save matched photos and collect metadata
            for rank, m in enumerate(matches, start=1):
                photo_path = m.get("photo_path")
                saved_path = None
                if photo_path:
                    try:
                        src_path = Path(str(photo_path))
                        ext = src_path.suffix if src_path.suffix else ".jpg"
                        dest_name = f"rank_{rank:03d}_photo_{m.get('photo_id','unknown')}{ext}"
                        saved_path = face_matches_dir / dest_name
                        shutil.copy(str(src_path), saved_path)
                    except Exception:
                        saved_path = None

                aggregated_matches.append({
                    "rank": rank,
                    "similarity": float(m.get("similarity_score", 0.0)),
                    "event_id": m.get("event_id"),
                    "photographer_id": m.get("photographer_id"),
                    "face_id": m.get("face_id"),
                    "photo_id": m.get("photo_id"),
                    "photo_path": photo_path,
                    "saved_copy_path": str(saved_path) if saved_path else None,
                    "public_url": m.get("public_url"),
                })

        # Sort final aggregated matches by similarity desc
        aggregated_matches.sort(key=lambda x: x["similarity"], reverse=True)
        if limit is not None:
            aggregated_matches = aggregated_matches[:limit]
        result["matches"] = aggregated_matches
        return result

    def _find_similar_faces_client_side(self, target_embedding: np.ndarray,
                                        event_id: str,
                                        photographer_id: Optional[str],
                                        similarity_threshold: float,
                                        metric: str,
                                        limit: Optional[int]) -> List[Dict]:
        """
        Client-side similarity search with pre-filtering on event_id and photographer_id.
        Returns the entire face document for each match.
        """
        if self.storage is None:
            return []

        try:
            # First, get faces filtered by event and optionally by photographer
            if photographer_id:
                all_faces = self.storage.get_faces_by_event_and_photographer(
                    event_id=event_id,
                    photographer_id=photographer_id
                )
            else:
                all_faces = self.storage.get_faces_by_event(event_id=event_id)
        except Exception as e:
            print(f"✗ Failed to retrieve faces from MongoDB: {e}")
            return []

        target_dim = int(target_embedding.shape[0])
        results: List[Dict] = []

        for face in all_faces:
            face_vec = face.get("embedding_vector")
            if face_vec is None:
                continue
            if isinstance(face_vec, list):
                face_vec = np.array(face_vec, dtype=float)

            # Skip dimension mismatch
            if face_vec.shape[0] != target_dim:
                continue

            if metric == "cosine":
                score = _cosine_similarity(target_embedding, face_vec)
                is_match = score >= similarity_threshold
            elif metric == "euclidean":
                score = float(np.linalg.norm(target_embedding - face_vec))
                is_match = score <= similarity_threshold
            else:
                print(f"✗ Unsupported metric: {metric}")
                return []

            if is_match:
                # Store the entire face document with the similarity score
                face['similarity_score'] = score
                results.append(face)

        # Sort results
        if metric == "cosine":
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
        else:
            results.sort(key=lambda x: x["similarity_score"])  # lower distance is better

        if limit is not None:
            return results[:limit]
        return results


if __name__ == "__main__":
    import argparse
    import sys

    print("Welcome to the Photo Search System!")

    # --- Interactive input for required fields ---
    event_id_input = input("Please enter the Event ID: ").strip()
    photographer_id_input = input("Please enter the Photographer ID: ").strip()
    photo_path_input = input("Please enter the path to the photo: ").strip()

    # --- Argument parser for optional settings ---
    parser = argparse.ArgumentParser(
        description="Search for matching faces in MongoDB using a single input photo."
    )
    # The default value for --limit is now None, which means no limit is applied
    parser.add_argument("--threshold", type=float, default=0.5, help="Similarity threshold (cosine) or max distance (euclidean)")
    parser.add_argument("--metric", type=str, default="cosine", choices=["cosine", "euclidean"], help="Similarity metric")
    parser.add_argument("--limit", type=int, default=None, help="Max number of matches to show (omit to show all)")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")

    # Parse only the arguments that are meant to be command-line options
    # We intentionally ignore any positional arguments here since they are handled by input()
    args = parser.parse_args()

    # Validate inputs
    if not event_id_input or not photographer_id_input or not photo_path_input:
        print("Error: Event ID, Photographer ID, and photo path are required.")
        sys.exit(1)

    engine = GuestSearchMongoDB(config_path=args.config)
    ok = engine.process_photo(
        user_photo_path=photo_path_input,
        event_id=event_id_input,
        photographer_id=photographer_id_input,
        similarity_threshold=args.threshold,
        metric=args.metric,
        limit=args.limit,
    )

    print("\n✅ Process complete." if ok else "\n❌ Process failed.")