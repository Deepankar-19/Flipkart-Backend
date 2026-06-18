"""
TrafficVision AI — FastAPI Backend
====================================
Entry-point for the traffic violation detection API.

Endpoints
---------
GET  /           → Health-check
POST /analyze    → Upload an image and receive a violation report

Run with:
    uvicorn main:app --reload
"""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from services.violation_engine import analyze_image

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="TrafficVision AI",
    description=(
        "Automated Photo Identification and Classification "
        "for Traffic Violations Using Computer Vision — "
        "Flipkart Grid 7.0 Round 2"
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# Temporary upload directory
# ---------------------------------------------------------------------------
_UPLOAD_DIR: Path = Path(__file__).resolve().parent / "uploads"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
async def health_check() -> dict[str, str]:
    """Simple health-check endpoint."""
    return {"status": "running"}


@app.post("/analyze", tags=["Analysis"])
async def analyze(
    file: UploadFile = File(..., description="Traffic image to analyse"),
) -> dict[str, Any]:
    """Accept an uploaded image, run the full violation pipeline, and
    return a structured JSON report.

    Response shape::

        {
            "helmet_violation": true,
            "triple_riding": false,
            "rider_count": 2,
            "number_plate": "MH48BR4639",
            "ocr_source": "alpr",
            "ocr_confidence": 0.97,
            "violations": ["Helmet Non Compliance"],
            "detections": [...],
            "annotated_image_path": "outputs/annotated_xxx.jpg"
        }
    """
    # --- Validate content type ------------------------------------------------
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"Expected an image file, got '{file.content_type}'.",
        )

    # --- Save uploaded file to a temporary location ---------------------------
    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    temp_filename = f"{uuid.uuid4().hex}{suffix}"
    temp_path = _UPLOAD_DIR / temp_filename

    try:
        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info("Saved upload to %s", temp_path)
    except Exception as exc:
        logger.exception("Failed to save uploaded file")
        raise HTTPException(
            status_code=500,
            detail="Could not save uploaded file.",
        ) from exc

    # --- Run analysis ---------------------------------------------------------
    try:
        report = analyze_image(temp_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Analysis failed")
        raise HTTPException(
            status_code=500,
            detail="An error occurred during image analysis.",
        ) from exc
    finally:
        # Clean up the temporary upload
        if temp_path.exists():
            temp_path.unlink()
            logger.info("Cleaned up temp file %s", temp_path)

    return JSONResponse(content=report)
