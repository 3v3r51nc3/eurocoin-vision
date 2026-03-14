from __future__ import annotations

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True, slots=True)
class CoinDetection:
    box_xyxy: tuple[int, int, int, int]
    detection_confidence: float
    material: str
    denomination: str


@dataclass(frozen=True, slots=True)
class PipelinePrediction:
    input_image: Image.Image
    detections: list[CoinDetection]
