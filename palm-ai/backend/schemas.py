from typing import Dict, Optional
from pydantic import BaseModel, Field

class ImageMetadata(BaseModel):
    normalized_width: int
    normalized_height: int

class LineFeatures(BaseModel):
    detected: bool
    num_fragments: int
    max_confidence: Optional[float]
    total_length: float
    normalized_length: float
    weighted_orientation: Optional[float]
    weighted_curvature: Optional[float]

class FeaturesSummary(BaseModel):
    detected_major_lines: int
    total_major_line_groups: int
    average_confidence: float
    minimum_confidence: float

class PalmFeatures(BaseModel):
    image_id: str
    image_metadata: ImageMetadata
    lines: Dict[str, LineFeatures]
    summary: FeaturesSummary

class AnalyzeResponseResult(BaseModel):
    features: PalmFeatures

class AnalyzeSuccessResponse(BaseModel):
    request_id: str
    status: str = "success"
    result: PalmFeatures

class ErrorDetail(BaseModel):
    code: str
    message: str

class ErrorResponse(BaseModel):
    request_id: str
    status: str = "failed"
    error: ErrorDetail
