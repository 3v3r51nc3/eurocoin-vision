from __future__ import annotations

from pathlib import Path


class AppConfig:
    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or Path(__file__).resolve().parents[2]
        self._supported_image_types = ["jpg", "jpeg", "png", "bmp", "webp"]
        self._denomination_cents = {
            "1_cent": 1,
            "2_cent": 2,
            "5_cent": 5,
            "10_cent": 10,
            "20_cent": 20,
            "50_cent": 50,
            "1_euro": 100,
            "2_euro": 200,
        }

    @property
    def base_dir(self) -> Path:
        return self._base_dir

    @property
    def runs_dir(self) -> Path:
        return self._base_dir / "runs"

    @property
    def model_weights_dir(self) -> Path:
        return self._base_dir / "model_weights"

    @property
    def page_title(self) -> str:
        return "Euro Coin Vision"

    @property
    def supported_image_types(self) -> list[str]:
        return list(self._supported_image_types)

    @property
    def denomination_cents(self) -> dict[str, int]:
        return dict(self._denomination_cents)

    def format_label(self, label: str) -> str:
        return label.replace("_", " ")

    def format_currency(self, cents: int) -> str:
        return f"EUR {cents / 100:.2f}"
