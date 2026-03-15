from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from webapp.config.app_config import AppConfig
from webapp.models.pipeline_prediction import PipelinePrediction


class PredictionAnnotationService:
    _MIN_FONT_SIZE = 12
    _FONT_SCALE = 0.025
    _MIN_ELLIPSE_WIDTH = 4
    _ELLIPSE_WIDTH_SCALE = 0.01
    _DEFAULT_STYLE = {
        "outline": "#00A86B",
        "label_fill": "#008E5A",
        "label_outline": "#005A39",
        "text_stroke": "#004A2F",
    }
    _STYLE_BY_MATERIAL = {
        "bronze": {
            "outline": "#CD7F32",
            "label_fill": "#B86A2B",
            "label_outline": "#8A4E20",
            "text_stroke": "#5F3415",
        },
        "gold": {
            "outline": "#D4AF37",
            "label_fill": "#B8992E",
            "label_outline": "#8C7423",
            "text_stroke": "#5C4C17",
        },
        "bicolor": {
            "outline": "#35A7FF",
            "label_fill": "#1A8BE0",
            "label_outline": "#146DAF",
            "text_stroke": "#0E4B77",
        },
    }

    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def render(self, prediction: PipelinePrediction) -> np.ndarray:
        annotated = prediction.input_image.copy()
        draw = ImageDraw.Draw(annotated)
        image_w, image_h = annotated.size
        min_side = min(image_w, image_h)
        base_font_size = max(self._MIN_FONT_SIZE, int(min_side * self._FONT_SCALE))
        ellipse_width = max(self._MIN_ELLIPSE_WIDTH, int(min_side * self._ELLIPSE_WIDTH_SCALE))
        font = self._load_font(base_font_size)

        for detection in prediction.detections:
            left, top, right, bottom = detection.box_xyxy
            label = self._config.format_label(detection.denomination)
            style = self._style_for_material(detection.material)
            draw.ellipse((left, top, right, bottom), outline=style["outline"], width=ellipse_width)

            text_box = draw.textbbox((left, top), label, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            text_padding_x = max(14, base_font_size // 3)
            text_padding_y = max(10, base_font_size // 5)
            text_top = max(0, top - text_height - (2 * text_padding_y) - 8)
            text_bottom = text_top + text_height + (2 * text_padding_y)
            text_right = left + text_width + (2 * text_padding_x)

            draw.rounded_rectangle(
                (left, text_top, text_right, text_bottom),
                fill=style["label_fill"],
                radius=max(14, base_font_size // 3),
                outline=style["label_outline"],
                width=max(3, base_font_size // 10),
            )
            draw.text(
                (left + text_padding_x, text_top + text_padding_y),
                label,
                fill="white",
                font=font,
                stroke_width=max(2, base_font_size // 12),
                stroke_fill=style["text_stroke"],
            )

        return np.array(annotated)

    @staticmethod
    def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
        font_candidates = [
            "arialbd.ttf",
            "segoeuib.ttf",
            "DejaVuSans-Bold.ttf",
        ]
        for font_name in font_candidates:
            try:
                return ImageFont.truetype(font_name, size=size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _style_for_material(self, material: str) -> dict[str, str]:
        normalized = material.strip().lower()
        return self._STYLE_BY_MATERIAL.get(normalized, self._DEFAULT_STYLE)
