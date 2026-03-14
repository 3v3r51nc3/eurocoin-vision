from __future__ import annotations

from pathlib import Path

from webapp.config.app_config import AppConfig


class WeightsPathResolver:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def resolve(self, raw_path: str) -> Path:
        candidate = Path(raw_path).expanduser()
        if candidate.is_absolute():
            return candidate
        return (self._config.base_dir / candidate).resolve()
