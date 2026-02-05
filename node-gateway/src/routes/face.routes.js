import express from "express";
import {
  ingestPhotographersController,
  searchGuestController
} from "../controllers/face.controller.js";

const router = express.Router();

/**
 * POST /api/face/ingest
 */
router.post("/ingest", ingestPhotographersController);

/**
 * POST /api/face/search
 */
router.post("/search", searchGuestController);

export default router;
