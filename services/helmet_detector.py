"""
TrafficVision AI — Helmet & Plate Detection Service
=====================================================
Loads a custom YOLOv8 model trained on 3 classes:
    0 → Plate
    1 → WithHelmet
    2 → WithoutHelmet

The model is loaded **once** at module import time so that
subsequent calls to `detect_objects()` are fast (no re-loading).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Class mapping — single source of truth for all services
# ---------------------------------------------------------------------------
CLASS_MAP: dict[int, str] = {
    0: "Plate",
    1: "WithHelmet",
    2: "WithoutHelmet",
}

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------
_MODEL_PATH: Path = Path(__file__).resolve().parent.parent / "models" / "helmet_detector.pt"
_model: YOLO | None = None

def _get_model() -> YOLO:
    global _model
    if _model is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO model not found at {_MODEL_PATH}. "
                "Ensure 'models/helmet_detector.pt' is present in the backend root."
            )
        _model = YOLO(str(_MODEL_PATH))
        logger.info("YOLO model loaded from %s", _MODEL_PATH)
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def detect_objects(
    image_path: str | Path,
    confidence_threshold: float = 0.25,
) -> list[dict[str, Any]]:
    """Run YOLOv8 inference on a single image.

    Parameters
    ----------
    image_path : str | Path
        Absolute or relative path to the input image.
    confidence_threshold : float, optional
        Minimum confidence to keep a detection (default 0.25).

    Returns
    -------
    list[dict[str, Any]]
        Each dict contains:
            - class_id   (int)   : numeric class label
            - class_name (str)   : human-readable class name
            - confidence (float) : detection confidence [0, 1]
            - bbox       (list)  : [x1, y1, x2, y2] pixel coordinates
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Run inference — returns a list of Results objects (one per image)
    results = _get_model().predict(
        source=str(image_path),
        conf=confidence_threshold,
        verbose=False,
    )

    detections: list[dict[str, Any]] = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            class_id: int = int(box.cls[0].item())
            confidence: float = round(float(box.conf[0].item()), 4)
            bbox: list[int] = [int(coord) for coord in box.xyxy[0].tolist()]

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": CLASS_MAP.get(class_id, "Unknown"),
                    "confidence": confidence,
                    "bbox": bbox,
                }
            )

    logger.info(
        "Detected %d object(s) in %s",
        len(detections),
        image_path.name,
    )
    return detections
