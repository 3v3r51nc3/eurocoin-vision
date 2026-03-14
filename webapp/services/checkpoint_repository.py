from __future__ import annotations

from pathlib import Path

from webapp.config.app_config import AppConfig


class CheckpointRepository:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def find_weight_candidates(self) -> list[Path]:
        if not self._config.runs_dir.exists():
            return []

        return sorted(
            (
                path
                for path in self._config.runs_dir.rglob("best.pt")
                if "weights" in path.parts
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
