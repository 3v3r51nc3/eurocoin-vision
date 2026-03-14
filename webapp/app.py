from __future__ import annotations

import numpy as np
import streamlit as st
from webapp.config.app_config import AppConfig
from webapp.services.detection_report_service import DetectionReportService
from webapp.services.image_export_service import ImageExportService
from webapp.services.model_inference_service import ModelInferenceService
from webapp.services.prediction_annotation_service import PredictionAnnotationService
from webapp.services.runtime_environment_service import RuntimeEnvironmentService
from webapp.services.uploaded_image_loader import ImageLoadingError, UploadedImageLoader
from webapp.views.results_view import ResultsView
from webapp.views.sidebar_view import SidebarView


class EuroCoinVisionApp:
    def __init__(self) -> None:
        self._config = AppConfig()
        self._model_inference_service = ModelInferenceService(self._config)
        self._detection_report_service = DetectionReportService(self._config)
        self._image_export_service = ImageExportService()
        self._prediction_annotation_service = PredictionAnnotationService(self._config)
        self._runtime_environment_service = RuntimeEnvironmentService()
        self._uploaded_image_loader = UploadedImageLoader(self._config)
        self._sidebar_view = SidebarView(self._config)
        self._results_view = ResultsView(self._config, self._image_export_service)

    def run(self) -> None:
        self._runtime_environment_service.ensure_streamlit_runtime()
        st.set_page_config(page_title=self._config.page_title, layout="wide")

        st.title(self._config.page_title)
        st.caption(
            "Upload an image to detect euro coins, estimate their count, and summarize denominations."
        )

        settings = self._sidebar_view.render()

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=self._config.supported_image_types,
        )
        if uploaded_file is None:
            st.info("Upload a JPG, PNG, BMP, or WEBP image to see detections in the browser.")
            st.stop()

        try:
            input_image = self._uploaded_image_loader.load(uploaded_file)
        except ImageLoadingError as error:
            st.error(str(error))
            st.stop()

        input_array = np.array(input_image)

        with st.spinner("Running inference..."):
            try:
                prediction = self._model_inference_service.predict(
                    image_array=input_array,
                    settings=settings,
                )
            except (FileNotFoundError, KeyError, ValueError) as error:
                st.error(f"Pipeline assets could not be loaded: {error}")
                st.stop()

        annotated_rgb = self._prediction_annotation_service.render(prediction)
        report = self._detection_report_service.build(prediction)

        self._results_view.render(
            uploaded_file_name=uploaded_file.name,
            input_image=input_image,
            annotated_image=annotated_rgb,
            report=report,
        )
