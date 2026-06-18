"""
TrafficVision AI — Visualization Service
========================================
Draws bounding boxes, labels, and violation details onto evidence images.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Any

# ---------------------------------------------------------------------------
# Colour palette for annotation (BGR)
# ---------------------------------------------------------------------------
_COLOURS: dict[str, tuple[int, int, int]] = {
    "Plate":          (255, 200, 0),     # cyan-ish
    "WithHelmet":     (0, 255, 0),       # green
    "WithoutHelmet":  (0, 0, 255),       # red
}

_DEFAULT_COLOUR: tuple[int, int, int] = (255, 255, 255)

def annotate_image(
    image: np.ndarray,
    detections: list[dict[str, Any]],
    triple_riding_details: list[dict[str, Any]] | None = None
) -> np.ndarray:
    """Draw bounding boxes, labels, and triple riding info on a **copy** of *image*."""
    annotated = image.copy()

    # 1. Draw helmet and plate detections
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_name = det["class_name"]
        confidence = det["confidence"]
        colour = _COLOURS.get(class_name, _DEFAULT_COLOUR)

        # Bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        # Label background
        label = f"{class_name} {confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 4, y1), colour, -1)

        # Label text
        cv2.putText(
            annotated,
            label,
            (x1 + 2, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    # 2. Draw triple riding details if available
    if triple_riding_details:
        for detail in triple_riding_details:
            bx1, by1, bx2, by2 = detail["bbox"]
            rider_count = detail["rider_count"]
            
            # Draw bike bounding box
            # To avoid overwriting helmet boxes completely, we use a different color
            cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 0, 255), 2) # Magenta for bike

            # Draw rider boxes
            for rider_box in detail["rider_boxes"]:
                rx1, ry1, rx2, ry2 = rider_box
                cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (0, 165, 255), 2) # Orange for riders

            # Draw rider count label
            label = f"Bike {detail['bike_id']}: Riders={rider_count}"
            color = (0, 0, 255) if rider_count >= 3 else (0, 255, 255)
            
            # Offset the y position slightly if it overlaps with a helmet detection label
            cv2.putText(annotated, label, (bx1, max(30, by1 - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return annotated
