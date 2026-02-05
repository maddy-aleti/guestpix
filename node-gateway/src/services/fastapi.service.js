import axios from "axios";
import FormData from "form-data";
import fs from "fs";

const FASTAPI_BASE_URL = "http://127.0.0.1:8000";

/**
 * Call POST /photographer/ingest
 */
export const ingestPhotographers = async (payload) => {
  const form = new FormData();

  form.append("event_name", payload.event_name);
  form.append("photographer_data", JSON.stringify(payload.photographers));
  form.append("config_path", "config/config.yaml");

  const response = await axios.post(
    `${FASTAPI_BASE_URL}/photographer/ingest`,
    form,
    { headers: form.getHeaders() }
  );

  return response.data;
};

/**
 * Call POST /guest/search
 */
export const searchGuest = async (payload) => {
  const form = new FormData();

  form.append("event_id", payload.event_id);
  if (payload.photographer_id) {
    form.append("photographer_id", payload.photographer_id);
  }
  form.append("photo_path", payload.photo_path);
  form.append("threshold", payload.threshold ?? 0.5);
  form.append("metric", payload.metric ?? "cosine");
  form.append("limit", payload.limit ?? 10);
  form.append("config_path", "config/config.yaml");

  const response = await axios.post(
    `${FASTAPI_BASE_URL}/guest/search`,
    form,
    { headers: form.getHeaders() }
  );

  return response.data;
};
