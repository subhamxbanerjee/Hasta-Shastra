# Palm Feature Extraction Schema (`palm_features.json`)

## Purpose
The Feature Extraction layer (Phase 6.0) bridges the gap between raw computer vision outputs (lines, contours, heuristic thresholds) and the future interpretation engine (semantic logic, palmistry rules, API frontend).

This layer guarantees that:
- The backend/frontend APIs do not need to process raw image matrices or OpenCV structures.
- Measurement normalization is standardized (e.g. mapping pixel lengths to relative palm sizes).
- A unified, deterministic JSON contract is available per image.

## Extracted Schema Definition

```json
{
  "image_id": "string",
  "image_metadata": {
    "normalized_width": "integer",
    "normalized_height": "integer"
  },
  "lines": {
    "LifeLine": {
      "detected": "boolean",
      "num_fragments": "integer",
      "max_confidence": "float | null",
      "total_length": "float",
      "normalized_length": "float",
      "weighted_orientation": "float | null",
      "weighted_curvature": "float | null"
    },
    "HeadLine": { ... },
    "HeartLine": { ... },
    "FateLine": { ... }
  },
  "summary": {
    "detected_major_lines": "integer",
    "total_major_line_groups": "integer",
    "average_confidence": "float",
    "minimum_confidence": "float"
  }
}
```

## Field Documentation

### Root Object
- `image_id`: Unique identifier derived from the source image filename.
- `image_metadata`: Metadata about the standardized space the CV engine operated within. Currently fixed to 512x512 but abstracted for forward compatibility.
- `lines`: The core feature container. Always contains exactly the four major lines (`LifeLine`, `HeadLine`, `HeartLine`, `FateLine`) regardless of detection status.
- `summary`: High-level aggregated statistics used for quick diagnostic reporting or batch evaluation.

### Line Object
- `detected` (boolean): `true` if at least one CV candidate group was assigned to this semantic class.
- `num_fragments` (integer): The number of independent connected components forming this line. A highly fragmented line suggests faintness or noise.
- `max_confidence` (float | null): The highest confidence score (`0.0` to `1.0`) across all fragments of this line. `null` if undetected.
- `total_length` (float): Sum of arc-lengths (in pixels) of all fragments.
- `normalized_length` (float): `total_length` divided by `image_metadata.normalized_height`. Provides a scale-invariant measure of line extent.
- `weighted_orientation` (float | null): The average orientation (in degrees `[0, 180)`) of all fragments, weighted by each fragment's length to prevent tiny noise dots from skewing the dominant angle.
- `weighted_curvature` (float | null): The average curvature metric (arc_length / area) of all fragments, weighted by length. `null` if undetected.

## Interpretation vs Features
**CRITICAL RULE:** This file contains *Features*, not *Interpretations*.

- **Feature**: `normalized_length: 0.85`
- **Interpretation (Future)**: `"You have a remarkably long LifeLine indicating..."`

The CV engine must never generate text predictions or map traits. It strictly outputs measurements (Features) that downstream modules consume.
