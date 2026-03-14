from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from webapp.config.app_config import AppConfig
from webapp.models.pipeline_prediction import PipelinePrediction


class PredictionAnnotationService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def render(self, prediction: PipelinePrediction) -> np.ndarray:
        annotated = prediction.input_image.copy()
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default()

        for detection in prediction.detections:
            left, top, right, bottom = detection.box_xyxy
            label = self._config.format_label(detection.denomination)
            draw.rectangle((left, top, right, bottom), outline="#00A86B", width=3)

            text_box = draw.textbbox((left, top), label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            text_top = max(0, top - text_height - 8)
            text_bottom = text_top + text_height + 4
            text_right = left + text_width + 8

            draw.rectangle((left, text_top, text_right, text_bottom), fill="#008E5A")
            draw.text((left + 4, text_top + 2), label, fill="white", font=font)

        return np.array(annotated)
