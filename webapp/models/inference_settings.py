from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class InferenceSettings:
    weights_input: str
    confidence_threshold: float
    iou_threshold: float
