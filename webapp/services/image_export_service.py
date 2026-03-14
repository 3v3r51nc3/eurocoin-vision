from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image


class ImageExportService:
    def to_png_bytes(self, image_array: np.ndarray) -> bytes:
        image = Image.fromarray(image_array)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
