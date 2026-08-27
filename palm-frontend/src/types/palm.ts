/**
 * palm.ts — Shared TypeScript types for PalmVerse
 *
 * WHY A SEPARATE TYPES FILE:
 *   TypeScript interfaces define the "shape" of data flowing through the app.
 *   Keeping them in one place means:
 *   - Any component can import exactly the type it needs
 *   - If the API response shape changes (e.g. when we replace the mock
 *     with a real FastAPI response), we only update this file
 *   - TypeScript will immediately flag any component using the old shape
 */

// ── User details collected in Step 1 ─────────────────────────────────────────
export interface UserDetails {
  name: string;
  dateOfBirth: string;      // ISO date string: "YYYY-MM-DD"
  placeOfBirth: string;
  gender?: "male" | "female" | "non-binary" | "prefer-not-to-say" | "";
}

// ── CV Major Line Features ──────────────────────────────────────────────────
export interface LineFeatures {
  detected: boolean;
  num_fragments: number;
  max_confidence: number | null;
  total_length: number;
  normalized_length: number;
  weighted_orientation: number | null;
  weighted_curvature: number | null;
}

// ── The full palm reading result (CV Pipeline Response) ──────────────────────
export interface PalmReadingResult {
  image_id: string;
  image_metadata: {
    normalized_width: number;
    normalized_height: number;
  };
  lines: {
    LifeLine: LineFeatures;
    HeadLine: LineFeatures;
    HeartLine: LineFeatures;
    FateLine: LineFeatures;
  };
  summary: {
    detected_major_lines: number;
    total_major_line_groups: number;
    average_confidence: number;
    minimum_confidence: number;
  };
}

// ── Global application state ─────────────────────────────────────────────────
export interface PalmReadingState {
  // Step 1 — User details
  userDetails: UserDetails | null;

  // Step 2 — Captured palm image (stored as a data URL or object URL)
  palmImage: string | null;

  // Step 3 — Analysis result
  result: PalmReadingResult | null;

  // Whether the user has completed the full flow
  isComplete: boolean;
}
