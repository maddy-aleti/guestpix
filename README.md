# Face Recognition with FastAPI + MongoDB

End-to-end face recognition pipeline to ingest event photos, detect faces, generate embeddings, and search matches, exposed via a simple FastAPI UI and backed by MongoDB.

### What you can do
- Ingest photos for an event and one or more photographers via the UI or CLI.
- Detect faces and compute embeddings with DeepFace (ArcFace by default).
- Store one document per detected face in MongoDB (with merged photo metadata).
- Search for a guest by uploading a photo and get ranked matches filtered by event/photographer.

---

## Architecture and Flow

Pipeline is orchestrated in three steps and exposed via an API:

1) Photo Ingestion (assign IDs, copy to storage, save local metadata)
2) Face Detection (DeepFace to crop faces; insert documents in MongoDB)
3) Face Embedding (DeepFace/ArcFace to compute vectors; update MongoDB)

Files and responsibilities:
- `api_server.py`: FastAPI app and minimal HTML UI for ingest + guest search
- `src/pipeline_mongodb.py`: Orchestrator coordinating the 3 steps and MongoDB
- `src/photo_ingestion.py`: Validates/copies photos, assigns `photo_id`, saves `photos_metadata.json`
- `src/face_detection.py`: DeepFace detection, saves 112×112 crops, inserts detected face docs
- `src/face_embedding.py`: DeepFace embeddings (ArcFace), L2-normalized, updates face docs with vectors
- `src/store_embeddings.py`: Mongo connector, indexes, CRUD, similarity utilities
- `src/guest_search.py`: Detects + embeds guest photo, pulls candidates by `event_id` / `photographer_id`, ranks by similarity, saves outputs

Tech stack:
- FastAPI (API), DeepFace (detection + embeddings), OpenCV/Numpy, MongoDB (pymongo)

---

## Data model (MongoDB)

Single collection: `photos` (one document per detected face, plus merged photo fields)

Indexes: `face_id` (unique), `photo_id`, `event_id`, `photographer_id`

Common fields in documents:
- Face: `face_id`, `photo_id`, `coordinates`/`bounding_box`, `confidence`, `detector_backend`, `crop_path`, `face_count`
- Embedding: `embedding_vector` (list[float]), `embedding_dimension`, `model_name`
- Photo metadata (merged): `photo_path`, `file_size`, `dimensions`, `public_url`, `event_id`, `photographer_id`
- Meta: `created_at`, `updated_at`

Configuration for MongoDB lives in `config/config.yaml` (see below).

---

## Repository Structure

```
facee/
├── api_server.py                 # FastAPI app (UI + endpoints)
├── main.py                       # CLI to run the pipeline on photos/directories
├── cleanup_and_run.py            # Remove duplicates then run pipeline
├── run_raw_photos.py             # Interactive organizer + pipeline runner
├── photographer_ingestion.py     # Standalone ingestion tool (optional pipeline)
├── src/
│   ├── __init__.py
│   ├── pipeline_mongodb.py       # Orchestrator (ingestion → detection → embedding)
│   ├── photo_ingestion.py        # Step 1: copy + assign IDs + metadata
│   ├── face_detection.py         # Step 2: DeepFace detection + crops + insert docs
│   ├── face_embedding.py         # Step 3: embeddings + update docs
│   ├── guest_search.py           # Guest search flow (client-side similarity)
│   └── store_embeddings.py       # MongoDB DAO (pymongo)
├── config/config.yaml            # MongoDB + pipeline settings
├── data/
│   ├── raw_photos/               # Organized as event_id/photographer_id/*.jpg
│   ├── cropped_faces/            # 112×112 crops from detection step
│   └── embeddings/               # Legacy embedding outputs (JSON/pickle)
├── output_user_photos/
│   └── matched_results/          # Guest query + matched outputs per search
├── results/
│   └── mongodb_pipeline_results.json
└── requirements.txt
```

---

## Configuration (`config/config.yaml`)

Example keys used by the codebase:

```yaml
mongodb:
  connection_string: "mongodb://localhost:27017"
  database_name: "face_recognition"
  collections:
    photos: "photos"   # single collection containing face docs + merged photo metadata

photo_ingestion:
  storage_path: "data/raw_photos"
  supported_formats: [".jpg", ".jpeg", ".png", ".bmp"]
  max_file_size_mb: 50
  # Optional public URL base for images copied to storage
  public_url_base: null
  # Optional source directory if scanning existing event/photographer folders
  raw_photos_source_dir: "data/raw_photos"

face_detection:
  detector_backend: "retinaface"   # fallback order also tries: mtcnn, ssd, opencv
  confidence_threshold: 0.5
  min_face_size: 20
  output_path: "data/cropped_faces"

face_embedding:
  model_name: "ArcFace"
  enforce_detection: false
  align: true
  output_path: "data/embeddings"
  augmentation:
    enabled: false
```

---

## Running the API (recommended)

Start the FastAPI server with built-in UI:

```bash
python api_server.py
```

Open `http://127.0.0.1:8000` for a minimal UI, or `http://127.0.0.1:8000/docs` for Swagger.

### Endpoints

1) POST `/photographer/ingest`
- Form fields:
  - `event_name`: string (e.g., "prasanna sneha")
  - `num_photographers`: number (used by UI)
  - `photographer_data`: JSON string `[{"name": "rahul", "folder": "C:\\path\\to\\photos"}, ...]`
  - `config_path`: optional (defaults to `config/config.yaml`)
- What it does:
  - Creates `event_id` and copies files into `data/raw_photos/<event_id>/<photographer_id>/...`
  - Builds records `[{'path','event_id','photographer_id'}]`
  - Runs the 3-step pipeline (ingestion → detection → embedding)
- Response includes: `event_id`, copied counts, pipeline summary, and per-photographer info.

2) POST `/guest/search`
- Form fields:
  - `event_id`: required
  - `photographer_id`: optional (to narrow search)
  - `photo_path`: path to guest image on disk
  - `threshold`: float (default 0.5), `metric`: `cosine` or `euclidean`, `limit`: int
  - `config_path`: optional
- What it does:
  - Detects face(s) in the guest image, computes embedding(s)
  - Retrieves candidate faces from MongoDB filtered by `event_id` (+ optional `photographer_id`)
  - Computes similarity on the client side and returns ranked matches
- Outputs:
  - `query_crops`: saved crop paths for the guest image
  - `matches`: ranked matches with `similarity`, `photo_path`, `face_id`, `photo_id`, etc.
  - `output_dir`: folder in `output_user_photos/matched_results/...` with copied ranked images

---

## Running from CLI

Process photos via command line (no API):

1) Run the pipeline for files or a folder
```bash
# Files
python main.py --mongodb --photos path/to/a.jpg path/to/b.jpg

# Directory (scans files in the folder)
python main.py --mongodb --directory data/raw_photos

# Optional flags
python main.py --mongodb --directory data/raw_photos --config config/config.yaml --force-reprocess
```

2) Clean duplicates, then run pipeline
```bash
python cleanup_and_run.py [--force-reprocess]
```

3) Interactive: organize event/photographer folders, then run
```bash
python run_raw_photos.py [--force-reprocess]
```

4) Standalone photographer ingestion (optional)
```bash
python photographer_ingestion.py
```

Outputs are written to `results/mongodb_pipeline_results.json` and MongoDB.

---
 
 run command :
 python -m venv venv   
 venv\Scripts\activate   
 pip install -r requirements.txt     
 python api_server.py   

 
## How it works (internals)

- `MongoDBStorage` (`src/store_embeddings.py`) connects using `pymongo.MongoClient` and prepares the `photos` collection and indexes.
- `PhotoIngestion` assigns UUID `photo_id`, copies files to `photo_ingestion.storage_path`, and records a persistent `photos_metadata.json` locally.
- `FaceDetection` uses DeepFace backends (configured + fallback) to detect/crop faces, writes 112×112 crops to `data/cropped_faces`, and inserts an initial face document in MongoDB (includes `event_id`, `photographer_id`, `photo_path`, `file_size`, etc.).
- `FaceEmbedding` generates a L2-normalized embedding (ArcFace by default) for each crop and updates the same MongoDB document with `embedding_vector`, `embedding_dimension`, `model_name`.
- `GuestSearchMongoDB` detects/embeds the query face, fetches candidates by `event_id`/`photographer_id`, computes cosine similarity client-side, and copies matched images into `output_user_photos/matched_results/<timestamp>`.

Similarity metrics supported: `cosine` (default) and `euclidean`.

---

## Prerequisites and Setup

1) Python
- Python 3.10+ recommended
- Install dependencies: `pip install -r requirements.txt`

2) MongoDB
- Local MongoDB or Atlas; ensure `config/config.yaml` points to the correct `connection_string` and `database_name`.
- The app creates needed indexes automatically.

3) DeepFace models
- DeepFace will download models (e.g., ArcFace) on first run.

---

## Tips and Troubleshooting

- Detection backends: if detection fails, try changing `face_detection.detector_backend` to `retinaface`/`mtcnn`/`ssd`/`opencv`.
- Thresholds: for stricter or looser search, adjust `threshold` in the guest search form or switch `metric`.
- Force reprocess: pass `--force-reprocess` or use API ingest again to re-run embeddings.
- Paths on Windows: ensure you pass absolute or correctly escaped paths in the UI/CLI.

---

## What has been implemented so far

- Full 3-step pipeline with MongoDB-backed storage (single `photos` collection with merged metadata).
- FastAPI app with a minimal in-browser UI for ingest and guest search.
- Scripts for organizing photos, deduping, and running the pipeline.
- Client-side similarity search filtered by event/photographer with output folders for results.

---

## License

See `LICENSE` (if provided by the repository owner). Otherwise, all rights reserved by the author.
