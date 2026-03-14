from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from ultralytics import YOLO

from webapp.models.inference_settings import InferenceSettings


class ModelInferenceService:
    @staticmethod
    @st.cache_resource(show_spinner=False)
    def _load_model(weights_path: str) -> YOLO:
        return YOLO(weights_path)

    def predict(self, image_array: np.ndarray, weights_path: Path, settings: InferenceSettings):
        model = self._load_model(str(weights_path))
        return model.predict(
            source=image_array,
            conf=settings.confidence_threshold,
            iou=settings.iou_threshold,
            verbose=False,
        )[0]
