#!/usr/bin/env python3
"""
FastAPI server exposing two endpoints:
- POST /photographer/ingest: Ingest photos for an event + photographer and run pipeline
- POST /guest/search: Search matches for a guest's photo filtered by event + photographer

Both endpoints use the existing MongoDB-backed pipeline and search utilities.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse, RedirectResponse

# Ensure src is importable
sys.path.append(str(Path(__file__).parent / "src"))

from src.pipeline_mongodb import MongoDBFaceRecognitionPipeline
from src.guest_search import GuestSearchMongoDB


app = FastAPI(title="Face Recognition API", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
async def root_ui():
    """Simple UI for photographer ingest and guest search."""
    return HTMLResponse(
        """
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Face Recognition UI</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 20px; }
    h2 { margin-top: 28px; }
    form { border: 1px solid #ddd; padding: 16px; border-radius: 8px; }
    .row { margin-bottom: 8px; }
    label { display: inline-block; width: 180px; }
    input[type=text], input[type=number], select { width: 280px; padding: 6px; }
    input[type=file] { width: 280px; }
    button { padding: 8px 14px; }
    pre { background: #f7f7f7; padding: 12px; border-radius: 6px; overflow-x: auto; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
    textarea { width: 280px; padding: 6px; font-family: monospace; }
    .card { box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  </style>
  <script>
    function setText(id, text) { document.getElementById(id).textContent = text; }
    function setValue(id, val) { document.getElementById(id).value = val; }

    function buildPhotographerRows() {
      const num = parseInt(document.getElementById('numPhotographers').value || '0', 10);
      const rows = document.getElementById('photographerRows');
      rows.innerHTML = '';
      for (let i = 0; i < num; i++) {
        const idx = i + 1;
        const wrap = document.createElement('div');
        wrap.className = 'row';
        wrap.innerHTML = `
          <div class="row"><label>Photographer ${idx} name</label><input type="text" id="photographer_name_${idx}" placeholder="e.g. rahul" required /></div>
          <div class="row"><label>Source folder (${idx})</label><input type="text" id="source_folder_${idx}" placeholder="C:\\path\\to\\photos" required /></div>
        `;
        rows.appendChild(wrap);
      }
    }

    async function postForm(url, form, outputId, evt) {
      if (evt && evt.preventDefault) evt.preventDefault();
      const out = document.getElementById(outputId);
      out.textContent = 'Submitting...';
      // Special handling for photographer ingest to assemble dynamic rows
      if (url === '/photographer/ingest') {
        const num = parseInt(document.getElementById('numPhotographers').value || '0', 10);
        const list = [];
        for (let i = 0; i < num; i++) {
          const idx = i + 1;
          const name = document.getElementById(`photographer_name_${idx}`)?.value || '';
          const folder = document.getElementById(`source_folder_${idx}`)?.value || '';
          if (name && folder) list.push({ name, folder });
        }
        const fd = new FormData(form);
        fd.set('photographer_data', JSON.stringify(list));
        try {
          const resp = await fetch(url, { method: 'POST', body: fd });
          const text = await resp.text();
          try {
            const data = JSON.parse(text);
            out.textContent = JSON.stringify(data, null, 2);
            if (data.event_id) setValue('searchEventId', data.event_id);
            // Steps rendering (created directories and copied photos)
            const steps = document.getElementById('ingestSteps');
            steps.innerHTML = '';
            if (Array.isArray(data.steps)) {
              data.steps.forEach(s => { const li = document.createElement('li'); li.textContent = s; steps.appendChild(li); });
            }
            // Photographers list + prefill first photographer id for search
            const list = document.getElementById('ingestPhotographers');
            list.innerHTML = '';
            if (Array.isArray(data.photographer_results) && data.photographer_results.length > 0) {
              data.photographer_results.forEach((r, i) => {
                const li = document.createElement('li');
                li.textContent = `${r.photographer_name} → ID: ${r.photographer_id} | copied: ${r.photos_copied}`;
                list.appendChild(li);
                if (i === 0 && r.photographer_id) setValue('searchPhotographerId', r.photographer_id);
              });
            }
          } catch(e) { out.textContent = text; }
        } catch (e) { out.textContent = 'Request failed: ' + e; }
        return false;
      }
      try {
        const resp = await fetch(url, { method: 'POST', body: new FormData(form) });
        const text = await resp.text();
        try {
          const data = JSON.parse(text);
          out.textContent = JSON.stringify(data, null, 2);
          // If search, render human-friendly matches
          // If search, render human-friendly matches
          if (url === '/guest/search') {
            try { renderMatches(data); } catch (e) { /* ignore */ }
          }
        }
        catch(e) { out.textContent = text; }
      } catch (e) {
        out.textContent = 'Request failed: ' + e;
      }
      return false;
    }

    function renderMatches(data) {
      const list = document.getElementById('matchList');
      list.innerHTML = '';
      if (!data || !Array.isArray(data.matches) || data.matches.length === 0) {
        list.innerHTML = '<li>No matches found above threshold.</li>';
        return;
      }
      data.matches.forEach(m => {
        const li = document.createElement('li');
        const sim = (m.similarity !== undefined) ? m.similarity.toFixed(4) : 'n/a';
        const photo = m.saved_copy_path || m.photo_path || '';
        li.textContent = `rank ${m.rank}: similarity ${sim} | face ${m.face_id} | photo ${photo}`;
        list.appendChild(li);
      });
      if (data.output_dir) {
        document.getElementById('matchesOutputDir').textContent = data.output_dir;
      }
    }
  </script>
  </head>
<body>
  <h1>Face Recognition</h1>
  <p>Use the forms below. Open API docs at <a href="/docs" target="_blank">/docs</a>.</p>

  <div class="grid">
    <div class="card">
      <h2>Photographer Ingest</h2>
      <form id="ingestForm" method="post" onsubmit="return postForm('/photographer/ingest', this, 'ingestOut', event);">
        <div class="row"><label>Event name</label><input type="text" name="event_name" required placeholder="e.g. prasana sneha" /></div>
        <div class="row"><label>Number of photographers</label><input type="number" id="numPhotographers" name="num_photographers" value="1" min="1" onchange="buildPhotographerRows()" onkeyup="buildPhotographerRows()" /></div>
        <div id="photographerRows"></div>
        <input type="hidden" name="photographer_data" />
        <div class="row"><button type="submit">Upload & Run</button></div>
      </form>
      <h3>Response</h3>
      <div style="margin-top:8px;"><strong>Steps</strong>
        <ul id="ingestSteps"></ul>
      </div>
      <div style="margin-top:8px;"><strong>Photographers</strong>
        <ul id="ingestPhotographers"></ul>
      </div>
      <pre id="ingestOut"></pre>
    </div>

    <div class="card">
      <h2>Guest Search</h2>
      <form id="searchForm" method="post" onsubmit="return postForm('/guest/search', this, 'searchOut', event);">
        <div class="row"><label>Event ID</label><input id="searchEventId" type="text" name="event_id" required placeholder="from ingest response" /></div>
        <div class="row"><label>Photographer ID (optional)</label><input id="searchPhotographerId" type="text" name="photographer_id" placeholder="leave blank to search whole event" /></div>
        <div class="row"><label>Guest photo path</label><input type="text" name="photo_path" required placeholder="C:\\TECHJOSE\\photography\\sample_photos\\prasanna.jpg" /></div>
        <div class="row"><label>Threshold</label><input type="number" step="0.01" name="threshold" value="0.5" /></div>
        <div class="row"><label>Metric</label><select name="metric"><option value="cosine" selected>cosine</option><option value="euclidean">euclidean</option></select></div>
        <div class="row"><label>Limit</label><input type="number" name="limit" value="10" /></div>
        <div class="row"><button type="submit">Search</button></div>
      </form>
      <h3>Response</h3>
      <pre id="searchOut"></pre>
      <h3>Matches</h3>
      <ul id="matchList"></ul>
      <div><strong>Saved output folder:</strong> <span id="matchesOutputDir">-</span></div>
    </div>
  </div>

  <p style="margin-top:24px;">Note: Query crops and matched images are saved under <code>output_user_photos/</code>.</p>
</body>
</html>
        """
    )


@app.post("/photographer/ingest")
async def ingest_photographer_folder(
    event_name: str = Form(..., description="Event name, e.g. 'prasana sneha'"),
    photographer_data: str = Form(..., description="JSON string: [{name, folder}, ...]"),
    config_path: str = Form("config/config.yaml"),
):
    """
    Ingest photos for an event from one or more photographers and run the pipeline.
    photographer_data: JSON list like [{"name":"rahul","folder":"C:\\path\\to\\photos"}, ...]
    """
    try:
        import uuid
        import json

        # Parse photographers list
        try:
            photographers = json.loads(photographer_data)
            if not isinstance(photographers, list):
                raise ValueError("photographer_data must be a list")
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON in photographer_data: {e}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        if not photographers:
            raise HTTPException(status_code=400, detail="No photographers provided")

        # Create event directory
        event_id = f"{event_name.replace(' ', '_').lower()}_{uuid.uuid4().hex[:8]}"
        event_dir = Path("data/raw_photos") / event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        all_photo_data: List[dict] = []
        photographer_results = []
        steps = [f"Created event directory: {str(event_dir)}"]
        supported_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        total_photos_copied = 0

        for i, info in enumerate(photographers):
            if not isinstance(info, dict) or "name" not in info or "folder" not in info:
                raise HTTPException(status_code=400, detail=f"Invalid photographer data at index {i}. Must have 'name' and 'folder'.")

            photographer_name = info["name"]
            src_dir = Path(info["folder"])
            if not src_dir.exists() or not src_dir.is_dir():
                steps.append(f"Folder not found for '{photographer_name}': {src_dir}")
                continue

            photographer_id = f"{photographer_name.replace(' ', '_').lower()}_{uuid.uuid4().hex[:8]}"
            dest_dir = event_dir / photographer_id
            dest_dir.mkdir(parents=True, exist_ok=True)
            steps.append(f"Created photographer directory: {str(dest_dir)}")

            copied = 0
            for entry in src_dir.iterdir():
                if entry.is_file() and entry.suffix.lower() in supported_exts:
                    target = dest_dir / entry.name
                    try:
                        target.write_bytes(entry.read_bytes())
                        copied += 1
                        all_photo_data.append({
                            "path": str(target),
                            "event_id": event_id,
                            "photographer_id": photographer_id,
                        })
                    except Exception:
                        pass
            steps.append(f"Copied {copied} photos for '{photographer_name}' to the new directory.")
            photographer_results.append({
                "photographer_name": photographer_name,
                "photographer_id": photographer_id,
                "dest_dir": str(dest_dir),
                "photos_copied": copied
            })
            total_photos_copied += copied

        if total_photos_copied == 0:
            raise HTTPException(status_code=400, detail="No photos copied from provided folders")

        pipeline = MongoDBFaceRecognitionPipeline(config_path)
        results = pipeline.run_complete_pipeline(all_photo_data)

        return JSONResponse({
            "event_id": event_id,
            "event_dir": str(event_dir),
            "total_photos_copied": total_photos_copied,
            "photographer_results": photographer_results,
            "steps": steps,
            "pipeline_summary": results.get("pipeline_summary", {}),
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/guest/search")
async def guest_search(
    event_id: str = Form(...),
    photographer_id: Optional[str] = Form(None),
    photo_path: str = Form(..., description="Full path to guest photo on disk"),
    threshold: float = Form(0.5),
    metric: str = Form("cosine"),
    limit: Optional[int] = Form(10),
    config_path: str = Form("config/config.yaml"),
):
    """
    Given a path to a guest's photo, detect face(s), compute embedding, and find matches
    in MongoDB filtered by event_id and photographer_id.
    """
    try:
        p = Path(photo_path)
        if not p.exists() or not p.is_file():
            raise HTTPException(status_code=400, detail=f"Photo not found: {photo_path}")

        engine = GuestSearchMongoDB(config_path)
        result = engine.search_photo(
            user_photo_path=str(p),
            event_id=event_id,
            photographer_id=photographer_id,
            similarity_threshold=threshold,
            metric=metric,
            limit=limit,
        )

        return JSONResponse(result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Bind to 127.0.0.1 for local browser access
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)


