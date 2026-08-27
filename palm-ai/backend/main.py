from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import cv2
import numpy as np

from backend.schemas import AnalyzeSuccessResponse, ErrorResponse, ErrorDetail
from backend.services.analysis_service import process_palm_image

app = FastAPI(
    title="Hasta-Shastra API",
    description="Backend API for structured palm feature extraction.",
    version="1.0.0"
)

# CORS configuration for local development
# Allow origins can be made configurable later via env variables
app.add_middleware(
    CORSMiddleware,
    allow_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

@app.get("/api/health")
def health_check():
    """Health check endpoint to verify server is running."""
    return {"status": "ok", "service": "hasta-shastra-api"}

@app.post("/api/analyze", response_model=AnalyzeSuccessResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def analyze_palm(file: UploadFile = File(...)):
    """
    Accepts a palm image upload, processes it through the CV pipeline, and returns the structured features.
    """
    if not file:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                request_id="unknown",
                status="failed",
                error=ErrorDetail(code="VALIDATION_ERROR", message="No file uploaded.")
            ).model_dump()
        )
        
    filename = file.filename.lower()
    if not any(filename.endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                request_id="unknown",
                status="failed",
                error=ErrorDetail(code="INVALID_FORMAT", message="Supported formats are .jpg, .jpeg, .png")
            ).model_dump()
        )
        
    file_bytes = await file.read()
    
    if len(file_bytes) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                request_id="unknown",
                status="failed",
                error=ErrorDetail(code="EMPTY_FILE", message="Uploaded file is empty.")
            ).model_dump()
        )
        
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                request_id="unknown",
                status="failed",
                error=ErrorDetail(code="FILE_TOO_LARGE", message=f"File exceeds maximum size of {MAX_FILE_SIZE_MB}MB.")
            ).model_dump()
        )

    # Decode check with OpenCV (ensures it's an image OpenCV can process)
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                request_id="unknown",
                status="failed",
                error=ErrorDetail(code="CORRUPT_IMAGE", message="Could not decode image data.")
            ).model_dump()
        )
        
    # Execute the pipeline orchestrator
    req_id, proc_status, result_payload = process_palm_image(file_bytes, file.filename)
    
    if proc_status == "success":
        return AnalyzeSuccessResponse(
            request_id=req_id,
            result=result_payload
        )
    else:
        # Pipeline execution failed
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                request_id=req_id,
                status="failed",
                error=ErrorDetail(**result_payload)
            ).model_dump()
        )
