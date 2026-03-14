from __future__ import annotations

from pathlib import Path

import streamlit as st

from webapp.config.app_config import AppConfig
from webapp.models.inference_settings import InferenceSettings
from webapp.services.checkpoint_repository import CheckpointRepository


class SidebarView:
    def __init__(self, config: AppConfig, checkpoint_repository: CheckpointRepository) -> None:
        self._config = config
        self._checkpoint_repository = checkpoint_repository

    def render(self) -> InferenceSettings:
        weight_candidates = self._checkpoint_repository.find_weight_candidates()
        default_weights = str(weight_candidates[0]) if weight_candidates else ""

        with st.sidebar:
            st.header("Inference Settings")
            weights_input = st.text_input("Model weights (.pt)", value=default_weights)
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

            if weight_candidates:
                st.caption("Detected checkpoints:")
                for checkpoint in weight_candidates[:5]:
                    st.code(self._format_checkpoint_path(checkpoint), language="text")
            else:
                st.warning("No local best.pt checkpoint was found inside runs/.")

        return InferenceSettings(
            weights_input=weights_input,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )

    def _format_checkpoint_path(self, checkpoint: Path) -> str:
        try:
            return str(checkpoint.relative_to(self._config.base_dir))
        except ValueError:
            return str(checkpoint)
