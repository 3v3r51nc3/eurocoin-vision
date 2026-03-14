from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class InferenceSettings:
    confidence_threshold: float
    iou_threshold: float
