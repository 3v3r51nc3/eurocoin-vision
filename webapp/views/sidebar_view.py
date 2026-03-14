from __future__ import annotations

import streamlit as st

from webapp.config.app_config import AppConfig
from webapp.models.inference_settings import InferenceSettings


class SidebarView:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def render(self) -> InferenceSettings:
        with st.sidebar:
            st.header("Inference Settings")
            confidence_threshold = st.slider(
                "Confidence threshold",
                min_value=0.05,
                max_value=0.95,
                value=0.25,
                step=0.05,
            )
            iou_threshold = st.slider(
                "IoU threshold",
                min_value=0.10,
                max_value=0.95,
                value=0.45,
                step=0.05,
            )

        return InferenceSettings(
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
