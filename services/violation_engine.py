"""
TrafficVision AI — Violation Engine
=====================================
Orchestrates the full analysis pipeline:

    Input image
        → YOLO detection (helmet_detector)
        → Triple Riding detection
        → Identify violations (Helmet, TripleRiding)
        → Crop detected plate region
        → OCR on plate crop (plate_ocr)
        → Generate annotated image with bounding boxes
        → Return structured violation report
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from services.helmet_detector import detect_objects
from services.plate_ocr import extract_plate_text
from services.triple_riding_detector import detect_triple_riding
from services.visualization import annotate_image

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------
_OUTPUT_DIR: Path = Path(__file__).resolve().parent.parent / "outputs"
_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# Violation checkers
# ═══════════════════════════════════════════════════════════════════════════

def _check_helmet_non_compliance(
    detections: list[dict[str, Any]],
) -> list[str]:
    """Flag ``WithoutHelmet`` detection as a violation."""
    for det in detections:
        if det["class_name"] == "WithoutHelmet":
            return ["Helmet Non Compliance"]
    return []

# ═══════════════════════════════════════════════════════════════════════════
# Plate extraction helper
# ═══════════════════════════════════════════════════════════════════════════

def _crop_plate(
    image: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray | None:
    """Return the cropped plate region with the highest confidence."""
    plate_dets = [d for d in detections if d["class_name"] == "Plate"]
    if not plate_dets:
        return None

    # Pick the plate with the highest confidence
    best_plate = max(plate_dets, key=lambda d: d["confidence"])
    x1, y1, x2, y2 = best_plate["bbox"]

    # Clamp to image boundaries
    h, w = image.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        logger.warning("Plate crop is empty (bbox may be invalid)")
        return None

    return crop

# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def analyze_image(
    image_path: str | Path,
    save_annotated: bool = True,
) -> dict[str, Any]:
    """End-to-end analysis of a single traffic image.

    Returns
    -------
    dict[str, Any]
        {
            "helmet_violation": bool,
            "triple_riding": bool,
            "rider_count": int,
            "number_plate": str,
            "ocr_source": str,
            "ocr_confidence": float,
            "violations": list[str],
            "detections": list[dict],
            "annotated_image_path": str | None,
        }
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    # ------------------------------------------------------------------
    # Step 1 — Helmet & Plate Detection
    # ------------------------------------------------------------------
    detections = detect_objects(image_path)
    logger.info("Helmet/Plate Detections: %s", detections)

    # ------------------------------------------------------------------
    # Step 2 — Triple Riding Detection
    # ------------------------------------------------------------------
    try:
        triple_riding_result = detect_triple_riding(image_path)
    except Exception as e:
        logger.error("Triple riding detection failed: %s", e)
        # Gracefully handle failure if yolov8l model cannot be loaded
        triple_riding_result = {
            "is_violation": False,
            "max_rider_count": 0,
            "details": []
        }
    
    # ------------------------------------------------------------------
    # Step 3 — Identify Violations
    # ------------------------------------------------------------------
    violations: list[str] = []
    
    helmet_vols = _check_helmet_non_compliance(detections)
    violations.extend(helmet_vols)
    has_helmet_violation = len(helmet_vols) > 0
    
    has_triple_riding = triple_riding_result["is_violation"]
    if has_triple_riding:
        violations.append("Triple Riding")

    # ------------------------------------------------------------------
    # Step 4 — Plate OCR
    # ------------------------------------------------------------------
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    plate_crop = _crop_plate(image, detections)
    
    if plate_crop is not None:
        ocr_result = extract_plate_text(plate_crop)
        vehicle_number = ocr_result.get("plate_number", "UNKNOWN")
        ocr_source = ocr_result.get("source", "none")
        ocr_confidence = ocr_result.get("confidence", 0.0)
    else:
        logger.info("No plate detected — skipping OCR")
        vehicle_number = "UNKNOWN"
        ocr_source = "none"
        ocr_confidence = 0.0

    # ------------------------------------------------------------------
    # Step 5 — Annotated image
    # ------------------------------------------------------------------
    annotated_image_path: str | None = None

    if save_annotated:
        annotated = annotate_image(image, detections, triple_riding_result["details"])
        output_name = f"annotated_{uuid.uuid4().hex[:8]}_{image_path.stem}.jpg"
        output_path = _OUTPUT_DIR / output_name
        cv2.imwrite(str(output_path), annotated)
        annotated_image_path = str(output_path)
        logger.info("Annotated image saved to %s", output_path)

    # ------------------------------------------------------------------
    # Step 6 — Build response
    # ------------------------------------------------------------------
    report: dict[str, Any] = {
        "helmet_violation": has_helmet_violation,
        "triple_riding": has_triple_riding,
        "rider_count": triple_riding_result["max_rider_count"],
        "number_plate": vehicle_number,
        "ocr_source": ocr_source,
        "ocr_confidence": ocr_confidence,
        "violations": violations,
        "detections": detections,
        "annotated_image_path": annotated_image_path,
    }

    return report
