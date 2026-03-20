from __future__ import annotations

from typing import Any

from PIL import Image

from webapp.config.app_config import AppConfig


class ImageLoadingError(Exception):
    pass


class UploadedImageLoader:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def load(self, uploaded_file: Any) -> Image.Image:
        """Open a Streamlit uploaded file and return it as an RGB PIL image."""
        if uploaded_file is None:
            raise ImageLoadingError("No file was uploaded.")

        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        try:
            return Image.open(uploaded_file).convert("RGB")
        except ModuleNotFoundError as error:
            if error.name == "pi_heif":
                raise ImageLoadingError(
                    "HEIC/HEIF support is unavailable in the current environment. "
                    "Install project dependencies again or upload JPG/PNG/BMP/WEBP."
                ) from error
            raise
        except (AttributeError, OSError, ValueError) as error:
            raise ImageLoadingError(
                "The uploaded file could not be read as an image. "
                f"Supported formats: {', '.join(self._config.supported_image_types).upper()}."
            ) from error
