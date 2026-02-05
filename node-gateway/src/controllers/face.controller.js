// face.controller.js
// Controller functions for face routes
import { ingestPhotographers, searchGuest } from "../services/fastapi.service.js";

export async function ingestPhotographersController(req, res) {
  try {
    const result = await ingestPhotographers(req.body);
    res.json(result);
  } catch (err) {
    res.status(500).json({
      error: "FastAPI ingest failed",
      details: err.message
    });
  }
}

export async function searchGuestController(req, res) {
  try {
    const result = await searchGuest(req.body);
    res.json(result);
  } catch (err) {
    res.status(500).json({
      error: "FastAPI search failed",
      details: err.message
    });
  }
}
