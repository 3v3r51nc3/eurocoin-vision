from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DetectionReport:
    summary_rows: list[dict[str, str]]
    detection_rows: list[dict[str, str]]
    total_count: int
    total_value_cents: int

    @property
    def has_detections(self) -> bool:
        return bool(self.summary_rows)

    @property
    def unique_denominations(self) -> int:
        return len(self.summary_rows)

    def summary_table(self) -> dict[str, list[str]]:
        return self._rows_to_table(self.summary_rows)

    def detection_table(self) -> dict[str, list[str]]:
        return self._rows_to_table(self.detection_rows)

    def _rows_to_table(self, rows: list[dict[str, str]]) -> dict[str, list[str]]:
        if not rows:
            return {}

        headers = rows[0].keys()
        return {header: [row[header] for row in rows] for header in headers}
