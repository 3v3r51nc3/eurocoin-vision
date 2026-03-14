from __future__ import annotations

from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from webapp.config.app_config import AppConfig
from webapp.models.detection_report import DetectionReport
from webapp.services.image_export_service import ImageExportService


class ResultsView:
    def __init__(self, config: AppConfig, image_export_service: ImageExportService) -> None:
        self._config = config
        self._image_export_service = image_export_service

    def render(
        self,
        uploaded_file_name: str,
        input_image: Image.Image,
        annotated_image: np.ndarray,
        report: DetectionReport,
    ) -> None:
        image_col, result_col = st.columns(2)

        with image_col:
            st.subheader("Uploaded Image")
            st.image(input_image, use_container_width=True)

        with result_col:
            st.subheader("Prediction Result")
            st.image(annotated_image, use_container_width=True)
            st.download_button(
                "Download annotated image",
                data=self._image_export_service.to_png_bytes(annotated_image),
                file_name=f"{Path(uploaded_file_name).stem}_prediction.png",
                mime="image/png",
            )

        metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
        metric_col_1.metric("Detected coins", str(report.total_count))
        metric_col_2.metric(
            "Estimated total value",
            self._config.format_currency(report.total_value_cents),
        )
        metric_col_3.metric("Unique denominations", str(report.unique_denominations))

        if not report.has_detections:
            st.warning("The model did not detect any coins in this image.")
            return

        st.subheader("Coin Summary")
        st.dataframe(report.summary_table(), use_container_width=True, hide_index=True)

        st.subheader("Raw Detections")
        st.dataframe(report.detection_table(), use_container_width=True, hide_index=True)
