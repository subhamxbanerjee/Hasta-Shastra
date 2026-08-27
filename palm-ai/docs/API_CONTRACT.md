# Hasta-Shastra Backend API Contract

## Overview
This document specifies the contract for the Hasta-Shastra Backend API (Phase 6.1). 
The API is built with FastAPI and wraps the existing computer vision pipeline, accepting raw palm images and returning structured feature analysis.

**Important Operational Notes:**
- The API processes requests entirely in memory and isolates disk outputs per request.
- No database is implemented in this phase.
- `request_id`s are ephemeral and mapped only to temporary server-side directories for debugging and evaluation.

## Endpoints

### 1. Health Check
Checks if the API server is operational. This endpoint does **not** invoke the CV pipeline.

**Request:**
`GET /api/health`

**Success Response (200 OK):**
```json
{
  "status": "ok",
  "service": "hasta-shastra-api"
}
```

---

### 2. Palm Analysis
Uploads an image, orchestrates the multi-stage CV pipeline, and returns the structured `palm_features.json` analysis.

**Request:**
`POST /api/analyze`

**Headers:**
`Content-Type: multipart/form-data`

**Body Parameters:**
- `file` (File, Required): The raw palm image. 
  - Supported formats: `.jpg`, `.jpeg`, `.png`.
  - Max size: 10 MB.

**Request Isolation Behavior:**
Every request is assigned a unique `request_id`. The server saves the image and all intermediate CV artifacts into a fully isolated directory (`data/processed/api_requests/<request_id>/`). This ensures no state contamination between concurrent requests and prevents overwriting batch validation data.

**Success Response (200 OK):**
```json
{
  "request_id": "api-123456789abc",
  "status": "success",
  "result": {
    "image_id": "uploaded_file.jpg",
    "image_metadata": {
      "normalized_width": 512,
      "normalized_height": 512
    },
    "lines": {
      "LifeLine": {
        "detected": true,
        "num_fragments": 2,
        "max_confidence": 0.85,
        "total_length": 420.5,
        "normalized_length": 0.82,
        "weighted_orientation": 75.2,
        "weighted_curvature": 0.45
      },
      "HeadLine": { ... },
      "HeartLine": { ... },
      "FateLine": { ... }
    },
    "summary": {
      "detected_major_lines": 3,
      "total_major_line_groups": 6,
      "average_confidence": 0.65,
      "minimum_confidence": 0.45
    }
  }
}
```
*Note: The `result` field matches the exact `palm_features.json` schema defined in `docs/PALM_FEATURE_SCHEMA.md`.*

**Validation Error Response (400 Bad Request):**
Returned when the uploaded file fails pre-pipeline validations (e.g., missing file, unsupported format, too large, unreadable image data).
```json
{
  "request_id": "unknown",
  "status": "failed",
  "error": {
    "code": "INVALID_FORMAT",
    "message": "Supported formats are .jpg, .jpeg, .png"
  }
}
```

**Server Error Response (500 Internal Server Error):**
Returned when the CV pipeline encounters an unexpected error during execution.
```json
{
  "request_id": "api-123456789abc",
  "status": "failed",
  "error": {
    "code": "PIPELINE_ERROR",
    "message": "Detailed error string from the internal CV scripts."
  }
}
```
*Note: Python tracebacks are safely caught and stripped from the API response payload.*
