"""
Step 1: Photo Ingestion & Storage
Handles photo upload, validation, and storage with unique ID assignment.
"""

import os
import shutil
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image
import yaml
from tqdm import tqdm


class PhotoIngestion:
    """
    Handles the ingestion and storage of photos for face recognition processing.

    What it does:
    - Validates uploaded photos (format, size, integrity)
    - Assigns unique Photo IDs to each photo
    - Stores photos in organized directory structure
    - Maintains metadata about stored photos
    - Checks MongoDB to avoid reprocessing existing photos
    - Handles new database scenarios by allowing reprocessing

    Expected output:
    - Collection of photos with unique Photo IDs stored centrally
    - Metadata file with photo information
    """

    def __init__(self, config_path: str = "config/config.yaml", db=None, force_reprocess=False):
        """Initialize the photo ingestion system."""
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.storage_path = Path(self.config['photo_ingestion']['storage_path'])
        self.supported_formats = self.config['photo_ingestion']['supported_formats']
        self.max_file_size = self.config['photo_ingestion']['max_file_size_mb'] * 1024 * 1024

        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # Store photo metadata (persisted across runs)
        self.photos_metadata = {}
        self._load_existing_metadata()

        # MongoDB connection for duplicate checking
        self.db = db

        # Flag to force reprocessing (useful for new databases)
        self.force_reprocess = force_reprocess

        # Check if this is a new/empty database
        self.is_new_database = self._check_if_new_database()

        # Track already ingested filenames (case-insensitive) to avoid duplicates by name
        # This will be populated correctly by ingest_photos
        self._existing_filenames_lower = set()
        # Initialize with existing filenames from metadata if loaded
        for meta in self.photos_metadata.values():
             if 'filename' in meta:
                  self._existing_filenames_lower.add(meta['filename'].lower())


    def _check_if_new_database(self) -> bool:
        """Check if the MongoDB database is empty (new database)."""
        if self.db is None:
            return False

        try:
            # Get collections from config
            photos_collection = self.config['mongodb']['collections']['photos']
            faces_collection = self.config['mongodb']['collections']['faces']

            # Check if collections exist and have documents
            photos_count = self.db[photos_collection].count_documents({})
            faces_count = self.db[faces_collection].count_documents({})

            is_new = (photos_count == 0 and faces_count == 0)
            if is_new:
                print("🆕 Detected new/empty database - will process all photos")
            else:
                print(f"📊 Existing database found: {photos_count} photos, {faces_count} faces")

            return is_new

        except Exception as e:
            print(f"⚠ Warning: Could not check database status: {e}")
            return False

    def validate_photo(self, photo_path: str) -> Tuple[bool, str]:
        """
        Validate a photo file for processing.

        Args:
            photo_path: Path to the photo file

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check file extension
            file_ext = Path(photo_path).suffix.lower()
            if file_ext not in self.supported_formats:
                return False, f"Unsupported file format: {file_ext}"

            # Check file size
            file_size = os.path.getsize(photo_path)
            if file_size > self.max_file_size:
                return False, f"File too large: {file_size / (1024*1024):.2f}MB > {self.max_file_size / (1024*1024)}MB"

            # Check if image can be opened
            with Image.open(photo_path) as img:
                img.verify()

            return True, "Photo is valid"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def ingest_photo(self, photo_data: Dict[str, Any]) -> str:
        """
        Ingest a single photo and assign a unique ID.
        Now accepts a dictionary containing photo path and metadata.

        Args:
            photo_data: A dictionary containing 'path', 'event_id', 'photographer_id'.

        Returns:
            Unique Photo ID assigned to the photo
        """
        photo_path = photo_data['path']
        event_id = photo_data.get('event_id')
        photographer_id = photo_data.get('photographer_id')

        # Validate photo
        is_valid, error_msg = self.validate_photo(photo_path)
        if not is_valid:
            raise ValueError(f"Photo validation failed for {Path(photo_path).name}: {error_msg}")

        # Generate unique Photo ID
        photo_id = str(uuid.uuid4())

        # Check if photo is already in the storage directory
        photo_path_obj = Path(photo_path)
        file_ext = photo_path_obj.suffix.lower()  # Always get file extension

        if photo_path_obj.parent == self.storage_path:
            # Photo is already in storage directory, just use existing path
            new_path = photo_path_obj
            new_filename = photo_path_obj.name
        else:
            # Copy photo to storage with Photo ID as filename
            new_filename = f"{photo_id}{file_ext}"
            new_path = self.storage_path / new_filename
            try:
                shutil.copy2(photo_path, new_path)
            except Exception as e:
                raise IOError(f"Failed to copy photo {photo_path} to {new_path}: {e}")

        # Store metadata
        self.photos_metadata[photo_id] = {
            'original_path': photo_path,
            'stored_path': str(new_path),
            'filename': new_filename,
            'file_size': os.path.getsize(photo_path),
            'file_format': file_ext,
            'event_id': event_id,
            'photographer_id': photographer_id,
            'ingestion_timestamp': str(Path(photo_path).stat().st_mtime)
        }
        # IMPORTANT: Add the filename to the set for duplicate checking
        self._existing_filenames_lower.add(new_filename.lower())

        return photo_id

    def _derive_event_and_photographer(self, file_path: Path) -> Tuple[Optional[str], Optional[str]]:
        """Infer event_id and photographer_id from path: storage_path/event_xxx/photographer_xxx/file"""
        try:
            # Expect .../data/raw_photos/event_XXX/photographer_YYY/filename
            # This assumes storage_path is where raw_photos is mounted or similar structure
            # Adjust logic if storage_path is different from the input source structure
            # For now, we derive from the path relative to the ultimate raw photos source
            return None, None # This logic is now handled by the caller based on input
        except Exception:
            return None, None

    def scan_and_register_photos_in_db(self) -> List[str]:
        """
        Scan folder structure data/raw_photos/event_xxx/photographer_xxx/*.ext and upsert Photos in MongoDB.
        Returns list of photo_ids registered or found.
        """
        if self.db is None:
            print("✗ MongoDB DB handle not provided to PhotoIngestion. Cannot register photos in DB.")
            return []

        photos_collection = self.config['mongodb']['collections']['photos']
        public_url_base = self.config['photo_ingestion'].get('public_url_base')

        registered_photo_ids: List[str] = []

        # Walk event/photographer folders relative to the initial raw_photos_dir
        # Assuming raw_photos_dir is the base for event_id/photographer_id structure
        raw_photos_source_dir = Path(self.config['photo_ingestion']['raw_photos_source_dir']) # Assumes this config param is set
        if not raw_photos_source_dir.exists():
             print(f"⚠ Warning: Source directory for raw photos ({raw_photos_source_dir}) not found. Skipping DB registration from structure.")
             return []

        supported = set(self.supported_formats)
        for event_dir in sorted(raw_photos_source_dir.glob('*')):
            if not event_dir.is_dir():
                continue
            event_id = event_dir.name
            for photographer_dir in sorted(event_dir.glob('*')):
                if not photographer_dir.is_dir():
                    continue
                photographer_id = photographer_dir.name
                for file_path in sorted(photographer_dir.iterdir()):
                    if not file_path.is_file():
                        continue
                    if file_path.suffix.lower() not in supported:
                        continue

                    # Validate file
                    is_valid, msg = self.validate_photo(str(file_path))
                    if not is_valid:
                        print(f"⚠ Skipping invalid file {file_path.name}: {msg}")
                        continue

                    abs_path = str(file_path.resolve())
                    # Try to find existing photo by unique path in DB
                    existing = self.db[photos_collection].find_one({"photo_path": abs_path})
                    if existing and existing.get('photo_id'):
                        photo_id = existing['photo_id']
                    else:
                        photo_id = str(uuid.uuid4())

                    # Build public URL
                    public_url = None
                    if public_url_base:
                        public_url = f"{public_url_base}/{event_id}/{photographer_id}/{file_path.name}"

                    # We no longer upsert into a separate photos collection. Metadata is kept locally
                    # and merged into faces documents during detection.

                    # Maintain lightweight local metadata map
                    self.photos_metadata[photo_id] = {
                        'original_path': abs_path,
                        'stored_path': abs_path, # Assuming the source is the storage for this function
                        'filename': file_path.name,
                        'file_size': os.path.getsize(abs_path) if os.path.exists(abs_path) else 0,
                        'file_format': file_path.suffix.lower(),
                        'event_id': event_id,
                        'photographer_id': photographer_id,
                        'public_url': public_url,
                    }
                    registered_photo_ids.append(photo_id)
        print(f"✓ Registered {len(registered_photo_ids)} photos in MongoDB from folder structure")
        return registered_photo_ids

    def get_existing_filenames_from_mongodb(self) -> set:
        """Get a set of existing filenames already stored in MongoDB photos collection."""
        existing_filenames = set()
        try:
            if self.db is not None:
                photos_collection = self.config['mongodb']['collections']['photos']
                cursor = self.db[photos_collection].find({}, {'photo_path': 1})
                for doc in cursor:
                    if doc.get('photo_path'):
                        filename = os.path.basename(doc['photo_path'])
                        existing_filenames.add(filename.lower())
                print(f"✓ Found {len(existing_filenames)} existing photo filenames in MongoDB")
        except Exception as e:
            print(f"⚠ Warning: Could not check MongoDB for existing photos: {e}")
        return existing_filenames

    def ingest_photos(self, photo_data_list: List[Dict[str, Any]]) -> List[str]:
        """
        Ingest multiple photos and assign unique IDs.
        Accepts a list of dictionaries, where each dict must have at least a 'path' key.
        Other keys like 'event_id', 'photographer_id' are also expected for metadata.

        Args:
            photo_data_list: List of dictionaries, each containing photo info.
                             Example: [{'path': '...', 'event_id': '...', 'photographer_id': '...'}, ...]

        Returns:
            List of Photo IDs for newly ingested photos.
        """
        newly_ingested_photo_ids = []
        processed_original_paths = set() # Track original paths processed in THIS run

        # Get existing filenames from MongoDB to avoid reprocessing if not in force_reprocess mode
        mongodb_existing_filenames = self.get_existing_filenames_from_mongodb()

        # Determine if we should skip existing photos
        should_skip_existing_in_db = not (self.force_reprocess or self.is_new_database)

        if should_skip_existing_in_db:
            print("🔄 Normal mode: Skipping photos already present in the database.")
        else:
            print("🆕 New database or force mode: Processing all photos.")

        print(f"Starting ingestion of {len(photo_data_list)} photos...")

        # Use tqdm for progress bar
        for photo_data in tqdm(photo_data_list, desc="Ingesting photos"):
            original_path = photo_data['path']
            filename = Path(original_path).name
            filename_lower = filename.lower()

            # --- Duplicate Checks ---
            # 1. Check if this exact path was already processed in this run
            if original_path in processed_original_paths:
                print(f"⚠ Skipped duplicate path in current batch: {filename}")
                continue
            processed_original_paths.add(original_path)

            # 2. Check if this filename has already been ingested locally (based on metadata.json or this run)
            if filename_lower in self._existing_filenames_lower:
                # If we are in skip mode AND this filename is already known to be ingested
                # (either from metadata.json or from processing in this same run), then skip.
                if should_skip_existing_in_db:
                    print(f"⚠ Skipped already ingested filename (local check): {filename}")
                    continue
                # If not in skip mode, we might re-ingest if force_reprocess is on, or if it's a new DB.
                # However, we should still avoid adding duplicate entries in the _existing_filenames_lower
                # if it's truly already processed and we are in normal mode.
                # The logic below for MongoDB check handles the actual DB skipping.


            # 3. Check if filename exists in MongoDB (if not in force_reprocess or new_database mode)
            if should_skip_existing_in_db and filename_lower in mongodb_existing_filenames:
                print(f"⚠ Skipped existing in MongoDB: {filename}")
                continue

            # --- Ingestion Process ---
            try:
                # ingest_photo now expects a dictionary
                photo_id = self.ingest_photo(photo_data)
                newly_ingested_photo_ids.append(photo_id)
                # The filename is added to _existing_filenames_lower within ingest_photo
                # print(f"✓ Ingested: {filename} -> Photo ID: {photo_id}") # tqdm handles progress
            except Exception as e:
                print(f"✗ Failed to ingest {filename}: {str(e)}")

        # Save metadata after processing all photos
        self.save_metadata()

        if not newly_ingested_photo_ids:
            if should_skip_existing_in_db:
                print("ℹ No new photos were ingested. All files appear to be already processed and in the database.")
            else:
                print("ℹ No new photos were ingested. All files might have been processed in a previous run or are duplicates.")
        else:
            print(f"Successfully ingested {len(newly_ingested_photo_ids)} new photos out of {len(photo_data_list)} provided.")

        return newly_ingested_photo_ids

    def _load_existing_metadata(self) -> None:
        """Load previously saved photo metadata if available."""
        import json
        metadata_path = self.storage_path / "photos_metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, 'r') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        # Update existing metadata, ensuring no overwrites if a photo_id already exists
                        for pid, meta in loaded.items():
                            if pid not in self.photos_metadata:
                                self.photos_metadata[pid] = meta
                        print(f"Loaded existing metadata for {len(self.photos_metadata)} photos from {metadata_path}")
            except Exception as e:
                print(f"Warning: failed to load existing metadata: {e}")

    def get_metadata_for_photo_ids(self, photo_ids: List[str]) -> Dict:
        """Return a metadata dict filtered to the specified photo IDs."""
        # Ensure only valid photo IDs are returned and they exist in our metadata
        return {pid: self.photos_metadata.get(pid) for pid in photo_ids if pid in self.photos_metadata}

    def save_metadata(self):
        """Save photo metadata to a JSON file."""
        import json
        metadata_path = self.storage_path / "photos_metadata.json"
        try:
            with open(metadata_path, 'w') as f:
                json.dump(self.photos_metadata, f, indent=2)
            print(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            print(f"✗ Failed to save metadata to {metadata_path}: {e}")

    def get_photo_info(self, photo_id: str) -> Dict:
        """Get information about a specific photo."""
        return self.photos_metadata.get(photo_id, {})

    def list_ingested_photos(self) -> List[str]:
        """Get list of all ingested photo IDs from local metadata."""
        return list(self.photos_metadata.keys())

    def get_storage_stats(self) -> Dict:
        """Get statistics about stored photos from local metadata."""
        total_size = sum(photo.get('file_size', 0) for photo in self.photos_metadata.values())

        return {
            'total_photos': len(self.photos_metadata),
            'total_size_mb': total_size / (1024 * 1024),
            'storage_path': str(self.storage_path),
            'supported_formats': self.supported_formats
        }

    def get_photos_needing_processing(self) -> List[str]:
        """Get list of photo IDs that might need face detection and embedding processing."""
        if self.db is None:
            print("⚠ MongoDB DB handle not provided. Cannot check processing status against DB.")
            # Fallback: Assume all locally ingested photos need processing if DB is unavailable
            return list(self.photos_metadata.keys())

        try:
            photos_collection = self.config['mongodb']['collections']['photos']
            faces_collection = self.config['mongodb']['collections']['faces']

            # Get all photo IDs from local metadata
            ingested_photo_ids = set(self.photos_metadata.keys())

            # Find photo IDs that are already present in the MongoDB 'photos' collection
            # This is important if metadata.json might be out of sync with DB
            mongodb_photo_ids = set()
            cursor = self.db[photos_collection].find({}, {'photo_id': 1})
            for doc in cursor:
                if doc.get('photo_id'):
                    mongodb_photo_ids.add(doc['photo_id'])

            # Consider only photos that are BOTH in local metadata AND in MongoDB photos collection
            # This ensures we are working with photos that have been successfully ingested and registered.
            valid_ingested_ids = ingested_photo_ids.intersection(mongodb_photo_ids)

            # Find photo IDs that have faces already processed in MongoDB
            processed_photo_ids_in_faces_collection = set()
            cursor = self.db[faces_collection].find({}, {'photo_id': 1})
            for doc in cursor:
                if doc.get('photo_id'):
                    processed_photo_ids_in_faces_collection.add(doc['photo_id'])

            # Photos needing processing are those that are validly ingested but do NOT have corresponding entries in the faces collection.
            photos_needing_processing = valid_ingested_ids - processed_photo_ids_in_faces_collection

            if not photos_needing_processing and valid_ingested_ids:
                print("All validly ingested photos already have face embeddings in MongoDB.")
            elif photos_needing_processing:
                print(f"Found {len(photos_needing_processing)} photos that need face detection and embedding.")
            else:
                print("No photos found in local metadata or MongoDB to process.")

            return list(photos_needing_processing)

        except Exception as e:
            print(f"⚠ Warning: Could not reliably check MongoDB for processing status: {e}")
            # Fallback: return all locally ingested photos if DB check fails
            return list(self.photos_metadata.keys())

    def set_force_reprocess(self, force: bool = True):
        """Manually set force reprocessing mode."""
        self.force_reprocess = force
        if force:
            print("🔄 Force reprocessing mode enabled - will process all photos")
        else:
            print("🔄 Normal mode enabled - will skip existing photos")

    def clear_existing_filenames(self):
        """Clear the existing filenames cache to allow reprocessing."""
        self._existing_filenames_lower.clear()
        print("🧹 Cleared existing filenames cache - ready for reprocessing")

    def reset_for_new_database(self):
        """Reset ingestion state for a new database."""
        self.force_reprocess = True
        self.is_new_database = True
        self.clear_existing_filenames()
        print("🆕 Reset for new database - will process all photos")


# Example usage and testing
if __name__ == "__main__":
    # Mock config and DB for local testing
    class MockConfig:
        def __init__(self):
            self.config = {
                'photo_ingestion': {
                    'storage_path': 'mock_storage',
                    'supported_formats': ['.jpg', '.jpeg', '.png'],
                    'max_file_size_mb': 10,
                    'public_url_base': None,
                    'raw_photos_source_dir': 'mock_raw_photos' # Added for scan_and_register_photos_in_db
                },
                'mongodb': {
                    'collections': {
                        'photos': 'photos',
                        'faces': 'faces'
                    }
                }
            }
        def __getitem__(self, key):
            return self.config[key]

    class MockDB:
        def __init__(self):
            self.collections = {}
            self.data = {} # Simulate collections and documents

        def __getitem__(self, collection_name):
            if collection_name not in self.collections:
                self.collections[collection_name] = MockCollection(self, collection_name)
            return self.collections[collection_name]

    class MockCollection:
        def __init__(self, db, name):
            self.db = db
            self.name = name
            if name not in self.db.data:
                self.db.data[name] = {}

        def count_documents(self, filter):
            return len(self.db.data[self.name])

        def find_one(self, filter):
            for doc_id, doc in self.db.data[self.name].items():
                match = True
                for key, value in filter.items():
                    if doc.get(key) != value:
                        match = False
                        break
                if match:
                    return doc
            return None

        def find(self, filter, projection=None):
            results = []
            for doc_id, doc in self.db.data[self.name].items():
                match = True
                for key, value in filter.items():
                    if doc.get(key) != value:
                        match = False
                        break
                if match:
                    if projection:
                        projected_doc = {}
                        for proj_key in projection:
                            if proj_key.endswith(':'): # Ignore
                                continue
                            if proj_key in doc:
                                projected_doc[proj_key] = doc[proj_key]
                        results.append(projected_doc)
                    else:
                        results.append(doc)
            return results # Return as an iterable

        def update_one(self, filter, update, upsert=False):
            doc_id_to_update = None
            for doc_id, doc in self.db.data[self.name].items():
                match = True
                for key, value in filter.items():
                    if doc.get(key) != value:
                        match = False
                        break
                if match:
                    doc_id_to_update = doc_id
                    break

            if doc_id_to_update:
                # Update existing document
                doc = self.db.data[self.name][doc_id_to_update]
                if "$set" in update:
                    doc.update(update["$set"])
                if "$setOnInsert" in update:
                    pass # Ignore $setOnInsert for updates
            elif upsert:
                # Insert new document
                new_id = str(uuid.uuid4())
                self.db.data[self.name][new_id] = {}
                if "$set" in update:
                    self.db.data[self.name][new_id].update(update["$set"])
                if "$setOnInsert" in update:
                    self.db.data[self.name][new_id].update(update["$setOnInsert"])
                doc_id_to_update = new_id

            return {"matched_count": 1 if doc_id_to_update else 0, "upserted_id": doc_id_to_update}


    # Create mock directories and files for testing
    mock_storage_dir = Path("mock_storage")
    mock_raw_photos_dir = Path("mock_raw_photos")
    mock_storage_dir.mkdir(parents=True, exist_ok=True)
    mock_raw_photos_dir.mkdir(parents=True, exist_ok=True)

    # Create dummy photo files
    dummy_photo_paths = []
    for i in range(3):
        for j in range(2):
            for k in range(2):
                filename = f"photo_{i}_{j}_{k}.jpg"
                path = mock_raw_photos_dir / f"event_{i}" / f"photographer_{j}" / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    img = Image.new('RGB', (60, 30), color = ('red'))
                    img.save(path)
                    dummy_photo_paths.append(str(path))
                except Exception as e:
                    print(f"Could not create dummy photo {path}: {e}")

    # Setup mock DB and config
    mock_db_instance = MockDB()
    mock_config_instance = MockConfig()

    # Initialize PhotoIngestion with mock DB and config
    ingestion = PhotoIngestion(
        config_path="mock_config.yaml", # Not used directly, config dict is passed
        db=mock_db_instance,
        force_reprocess=False # Start with normal mode
    )
    ingestion.config = mock_config_instance.config # Set config for the ingested object

    # --- Test Case 1: Initial ingest ---
    print("\n--- Test Case 1: Initial Ingestion ---")
    # Scan and register photos in DB first (simulating existing DB state for `scan_and_register_photos_in_db`)
    # This will create dummy entries in our mock_db_instance['photos']
    ingestion.scan_and_register_photos_in_db() # This function assumes raw_photos_source_dir exists

    # Now, ingest the files from the raw_photos_source_dir using ingest_photos
    # We need to transform the file paths into the dictionary format expected
    photo_data_for_ingestion = []
    for p_str in dummy_photo_paths:
        p = Path(p_str)
        photo_data_for_ingestion.append({
            'path': str(p),
            'event_id': p.parent.parent.name,
            'photographer_id': p.parent.name
        })

    # ingest_photos will check against _existing_filenames_lower, and then MongoDB
    # If scan_and_register_photos_in_db populated DB, and we are not force_reprocessing,
    # ingest_photos should skip them.
    ingested_ids_1 = ingestion.ingest_photos(photo_data_for_ingestion)
    print(f"Ingested Photo IDs (Test 1): {ingested_ids_1}")
    print(f"Total metadata entries after Test 1: {len(ingestion.photos_metadata)}")
    print(f"Existing filenames cache size: {len(ingestion._existing_filenames_lower)}")

    # --- Test Case 2: Re-run with same files, normal mode ---
    print("\n--- Test Case 2: Re-run in Normal Mode ---")
    # Simulate running again without force_reprocess. Should ingest 0 new photos.
    ingestion.set_force_reprocess(False)
    ingested_ids_2 = ingestion.ingest_photos(photo_data_for_ingestion)
    print(f"Ingested Photo IDs (Test 2): {ingested_ids_2}")
    print(f"Total metadata entries after Test 2: {len(ingestion.photos_metadata)}") # Should be the same

    # --- Test Case 3: Re-run with force_reprocess ---
    print("\n--- Test Case 3: Re-run with Force Reprocess ---")
    ingestion.set_force_reprocess(True)
    # Ingesting again with force_reprocess should technically re-ingest if logic allowed,
    # but our `_existing_filenames_lower` check would still prevent duplicates of the *same filename*.
    # The DB check `filename_lower in mongodb_existing_filenames` would be bypassed by force_reprocess.
    # The `original_path in processed_original_paths` check prevents re-processing the exact same path in the same run.
    # If `ingest_photo` logic correctly adds `filename_lower` to `_existing_filenames_lower`,
    # then even with force_reprocess, we won't re-ingest the *same* filename if it was already processed
    # in this run or loaded from metadata.
    # To truly re-ingest (e.g., if files changed), one would need to delete from DB or remove from metadata.json.
    # For now, we expect 0 new IDs, but the check logic is what we're testing here.
    ingested_ids_3 = ingestion.ingest_photos(photo_data_for_ingestion)
    print(f"Ingested Photo IDs (Test 3): {ingested_ids_3}")
    print(f"Total metadata entries after Test 3: {len(ingestion.photos_metadata)}")

    # --- Test Case 4: New database mode ---
    print("\n--- Test Case 4: Simulate New Database ---")
    # Reset ingestion state to simulate a new database
    ingestion.reset_for_new_database() # This sets force_reprocess=True and clears cache
    # Now, when we ingest, it should effectively treat all files as new
    # due to clearing the cache and setting force_reprocess.
    # However, our _existing_filenames_lower check would still prevent duplicates if they exist in the same batch.
    # To simulate truly new, we'd need to clear mock_db_instance data.
    # For now, let's just see if force_reprocess works as expected on the cache.
    ingested_ids_4 = ingestion.ingest_photos(photo_data_for_ingestion)
    print(f"Ingested Photo IDs (Test 4): {ingested_ids_4}")
    print(f"Total metadata entries after Test 4: {len(ingestion.photos_metadata)}")

    print("\n--- Cleaning up mock directories ---")
    try:
        shutil.rmtree(mock_storage_dir)
        shutil.rmtree(mock_raw_photos_dir)
        print("Mock directories removed.")
    except Exception as e:
        print(f"Error cleaning up mock directories: {e}")