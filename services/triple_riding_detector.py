"""
TrafficVision AI — Triple Riding Detector
=========================================
Uses YOLOv8l to detect motorcycles and persons, and applies filtering
to associate riders with bikes and identify triple riding violations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
PERSON_CONF_MIN   = 0.3   
PERSON_AREA_MIN   = 1500   
OVERLAP_THRESHOLD = 0.23   
BIKE_EXPAND_RATIO = 0.1   
MAX_RIDERS        = 4      

_MODEL_PATH: Path = Path(__file__).resolve().parent.parent / "models" / "yolov8l.pt"
_model: YOLO | None = None

def _get_model() -> YOLO:
    global _model
    if _model is None:
        try:
            if not _MODEL_PATH.exists():
                logger.info("YOLOv8l model not found at %s. Ultralytics will download it.", _MODEL_PATH)
            
            # YOLO will automatically download to the current working directory if just given "yolov8l.pt"
            # Actually, ultralytics YOLO("yolov8l.pt") downloads to current dir. We will just pass the absolute path.
            # If the file isn't there, it might fail or download. Passing the filename directly is safer for auto-download.
            _model = YOLO(str(_MODEL_PATH) if _MODEL_PATH.exists() else "yolov8l.pt")
            logger.info("YOLOv8l loaded for Triple Riding Detection")
        except Exception:
            logger.exception("Failed to initialize YOLOv8l for Triple Riding.")
            raise
    return _model

# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _overlap_ratio(person_box: list[int], bike_box: list[int]) -> float:
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box

    ix1 = max(px1, bx1)
    iy1 = max(py1, by1)
    ix2 = min(px2, bx2)
    iy2 = min(py2, by2)

    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    person_area  = (px2 - px1) * (py2 - py1)
    return intersection / person_area

def _expand_bike_box_downward(bike_box: list[int], ratio: float = BIKE_EXPAND_RATIO) -> list[int]:
    """Expand only the bottom edge — catches rear passenger sitting behind the bike."""
    bx1, by1, bx2, by2 = bike_box
    height = by2 - by1
    return [bx1, by1, bx2, int(by2 + height * ratio)]

def _is_valid_person(box: list[int], conf: float) -> bool:
    """All three person filters in one place."""
    px1, py1, px2, py2 = box
    w = px2 - px1
    h = py2 - py1
    area = w * h

    if conf < PERSON_CONF_MIN:   
        return False
    if area < PERSON_AREA_MIN:   
        return False
    if w > h:                    
        return False
    return True

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_triple_riding(image_input: str | Path | np.ndarray) -> dict[str, Any]:
    """Detect triple riding violations on an image.

    Returns:
        dict containing:
            - is_violation: bool
            - max_rider_count: int
            - details: list of bikes and their associated rider count/boxes
    """
    if isinstance(image_input, (str, Path)):
        image_path = Path(image_input)
        if not image_path.exists():
            raise FileNotFoundError(f"Input image not found: {image_path}")
        results = _get_model().predict(source=str(image_path), verbose=False)[0]
    else:
        results = _get_model().predict(source=image_input, verbose=False)[0]

    motorcycles = []
    persons = []

    if results.boxes is not None:
        for box in results.boxes:
            if box.cls is None or box.conf is None or box.xyxy is None:
                continue
                
            cls  = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            if cls == 3:  # COCO class 3 is motorcycle
                motorcycles.append([x1, y1, x2, y2])
            elif cls == 0:  # COCO class 0 is person
                if _is_valid_person([x1, y1, x2, y2], conf):
                    persons.append([x1, y1, x2, y2, conf])

    logger.info("Valid persons after filtering: %d", len(persons))

    bike_details = []
    max_rider_count = 0
    is_violation = False

    for bike_id, bike in enumerate(motorcycles):
        bx1, by1, bx2, by2 = bike
        expanded_bike = _expand_bike_box_downward(bike)

        candidate_riders = []
        associated_rider_boxes = []

        for person in persons:
            px1, py1, px2, py2, conf = person
            ratio = _overlap_ratio([px1, py1, px2, py2], expanded_bike)  

            if ratio > OVERLAP_THRESHOLD:
                area = (px2 - px1) * (py2 - py1)
                score = conf * ratio * area
                candidate_riders.append(score)
                associated_rider_boxes.append([px1, py1, px2, py2])

        candidate_riders.sort(reverse=True)
        rider_count = len(candidate_riders)
        rider_count = min(rider_count, MAX_RIDERS)

        if rider_count > max_rider_count:
            max_rider_count = rider_count

        if rider_count >= 3:
            is_violation = True

        bike_details.append({
            "bike_id": bike_id,
            "bbox": bike,
            "rider_count": rider_count,
            "rider_boxes": associated_rider_boxes
        })

    logger.info("Triple riding check: violation=%s, max_riders=%d", is_violation, max_rider_count)
    
    return {
        "is_violation": is_violation,
        "max_rider_count": max_rider_count,
        "details": bike_details
    }
