"""
TrafficVision AI — Plate OCR Service
======================================
Extracts vehicle registration numbers from cropped plate images.

Multi-stage Pipeline:
    1. Primary: ALPR API (Provider agnostic)
    2. Fallback: EasyOCR with local image enhancement

Image Enhancement Pipeline (for Fallback):
    Plate Crop
        → Upscale (2×)
        → Bilateral Filtering (noise reduction)
        → CLAHE (contrast limited adaptive histogram equalization)
        → Sharpening (edge enhancement)
        → Grayscale conversion
        → Optional Deblur Module (Future enhancement placeholder)
        → EasyOCR
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np
import requests

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# EasyOCR reader singleton
# ---------------------------------------------------------------------------
_ALLOWLIST: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
reader: easyocr.Reader = easyocr.Reader(["en"], gpu=True)
logger.info("EasyOCR reader initialised (GPU enabled if available)")

# ---------------------------------------------------------------------------
# ALPR API Integration (Primary OCR)
# ---------------------------------------------------------------------------
def _call_alpr_api(image: np.ndarray) -> dict[str, Any] | None:
    """
    Attempt to call an external ALPR API.
    Must be provider agnostic.

    Expected env vars:
        ALPR_API_URL
        ALPR_API_KEY
    """
    api_url = os.environ.get("ALPR_API_URL")
    api_key = os.environ.get("ALPR_API_KEY")

    if not api_url or not api_key:
        logger.info("ALPR API credentials not found. Skipping primary OCR.")
        return None

    try:
        # Encode image to JPEG for the API request
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            logger.error("Failed to encode image for ALPR API.")
            return None

        # This is a generic implementation. Provider specific payload adjustments
        # (e.g. Plate Recognizer format) would go here.
        headers = {
            "Authorization": f"Token {api_key}"
        }
        files = {
            "upload": ("plate.jpg", buffer.tobytes(), "image/jpeg")
        }

        response = requests.post(
            api_url,
            headers=headers,
            files=files,
            timeout=5.0
        )
        response.raise_for_status()
        data = response.json()

        # Assuming a generic response structure parsing.
        # If using Plate Recognizer: data["results"][0]["plate"], etc.
        # We try to extract the plate and confidence defensively.
        results = data.get("results", [])
        if not results:
            return None

        best_result = results[0]
        plate = str(best_result.get("plate", "")).upper()
        # Plate recognizer returns confidence as a float out of 1 or 100 depending on version, normalize to [0,1]
        conf = float(best_result.get("score", best_result.get("confidence", 0.0)))
        if conf > 1.0:
            conf = conf / 100.0

        plate_number = _clean_plate_text(plate)
        if plate_number and conf > 0.5: # ALPR confidence threshold
            logger.info("ALPR API success: %s (conf: %.2f)", plate_number, conf)
            return {
                "plate_number": plate_number,
                "source": "alpr",
                "confidence": round(conf, 4)
            }
        return None

    except requests.RequestException as exc:
        logger.warning("ALPR API request failed: %s", exc)
        return None
    except Exception as exc:
        logger.warning("ALPR API processing failed: %s", exc)
        return None

# ---------------------------------------------------------------------------
# Image Enhancement Pipeline (Fallback)
# ---------------------------------------------------------------------------

def _upscale_image(image: np.ndarray, scale_factor: int = 2) -> np.ndarray:
    """Increase plate resolution preserving edges."""
    width = image.shape[1] * scale_factor
    height = image.shape[0] * scale_factor
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)

def _apply_clahe(image: np.ndarray) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization."""
    # Convert to LAB color space to apply CLAHE to the L channel only
    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    else:
        # If already grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

def _sharpen_image(image: np.ndarray) -> np.ndarray:
    """Enhance character edges using an unsharp mask technique or kernel."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def _apply_deblur_model(image: np.ndarray) -> np.ndarray:
    """
    Placeholder for future advanced restoration models.
    (e.g., FFTFormer, SRMNet, ConvIR, EVSSM).
    Currently returns the image unmodified.
    """
    # TODO: Implement advanced deblurring model inference here
    return image

def _preprocess_plate(image: np.ndarray) -> np.ndarray:
    """
    Full image enhancement pipeline for local OCR fallback.
    Upscale -> Bilateral Filter -> CLAHE -> Sharpen -> Grayscale
    """
    # 1. Intelligent Upscaling
    upscaled = _upscale_image(image, scale_factor=2)

    # 2. Bilateral Filtering (reduce noise, preserve edges)
    filtered = cv2.bilateralFilter(upscaled, d=11, sigmaColor=17, sigmaSpace=17)

    # 3. CLAHE (improve local contrast)
    clahe_applied = _apply_clahe(filtered)

    # 4. Sharpening (enhance character edges)
    sharpened = _sharpen_image(clahe_applied)

    # 5. Grayscale Conversion
    if len(sharpened.shape) == 3:
        gray = cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY)
    else:
        gray = sharpened

    # Future Extension Point
    final_processed = _apply_deblur_model(gray)

    return final_processed

# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def _clean_plate_text(raw_text: str) -> str:
    """Remove all characters outside the allowlist and collapse spaces."""
    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    return cleaned

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_plate_text(
    image_input: str | Path | np.ndarray,
) -> dict[str, Any]:
    """Extract the vehicle registration number using a multi-stage OCR pipeline.

    Returns
    -------
    dict[str, Any]
        {
            "plate_number": str (e.g. "MH48BR4639" or "UNKNOWN"),
            "source": str ("alpr", "easyocr", or "none"),
            "confidence": float
        }
    """
    # Load image if a path was supplied
    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)
        if not image_path.exists():
            logger.warning("Plate image not found: %s", image_path)
            return {"plate_number": "UNKNOWN", "source": "none", "confidence": 0.0}
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Failed to read plate image: %s", image_path)
            return {"plate_number": "UNKNOWN", "source": "none", "confidence": 0.0}
    else:
        image = image_input

    # -----------------------------------------------------------------------
    # Stage 1: ALPR API (Primary)
    # -----------------------------------------------------------------------
    alpr_result = _call_alpr_api(image)
    if alpr_result is not None:
        return alpr_result

    logger.info("Falling back to local EasyOCR pipeline")

    # -----------------------------------------------------------------------
    # Stage 2: Local Image Enhancement + EasyOCR (Fallback)
    # -----------------------------------------------------------------------
    processed = _preprocess_plate(image)

    try:
        # EasyOCR returns a list of tuples: (bbox, text, confidence)
        ocr_results = reader.readtext(
            processed,
            allowlist=_ALLOWLIST,
            paragraph=False,
        )
    except Exception:
        logger.exception("EasyOCR failed")
        return {"plate_number": "UNKNOWN", "source": "none", "confidence": 0.0}

    if not ocr_results:
        logger.info("EasyOCR returned no results")
        return {"plate_number": "UNKNOWN", "source": "none", "confidence": 0.0}

    # Aggregate text and calculate an average confidence
    raw_texts = []
    confidences = []
    for bbox, text, conf in ocr_results:
        raw_texts.append(text)
        confidences.append(conf)

    raw_text = " ".join(raw_texts)
    plate_number = _clean_plate_text(raw_text)

    if not plate_number:
        logger.info("Plate text empty after cleaning (raw: %r)", raw_text)
        return {"plate_number": "UNKNOWN", "source": "none", "confidence": 0.0}

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    logger.info("Extracted plate number via EasyOCR: %s (conf: %.2f)", plate_number, avg_conf)
    return {
        "plate_number": plate_number,
        "source": "easyocr",
        "confidence": round(avg_conf, 4)
    }
