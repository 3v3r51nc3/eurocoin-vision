from __future__ import annotations

from collections import Counter

from webapp.config.app_config import AppConfig
from webapp.models.detection_report import DetectionReport
from webapp.models.pipeline_prediction import PipelinePrediction


class DetectionReportService:
    def __init__(self, config: AppConfig) -> None:
        self._config = config

    def build(self, prediction: PipelinePrediction) -> DetectionReport:
        """Aggregate pipeline detections into per-coin rows, a denomination summary, and totals."""
        counts: Counter[str] = Counter()
        detection_rows: list[dict[str, str]] = []

        if not prediction.detections:
            return DetectionReport([], [], 0, 0)

        for detection in prediction.detections:
            counts[detection.denomination] += 1
            detection_rows.append(
                {
                    "Denomination": self._config.format_label(detection.denomination),
                    "Material": self._config.format_label(detection.material),
                    "Confidence": f"{detection.detection_confidence:.2%}",
                }
            )

        summary_rows: list[dict[str, str]] = []
        total_value_cents = 0

        for label, count in sorted(counts.items(), key=self._sort_key):
            denomination_value = self._config.denomination_cents.get(label)
            subtotal_cents = denomination_value * count if denomination_value is not None else 0
            total_value_cents += subtotal_cents
            summary_rows.append(
                {
                    "Denomination": self._config.format_label(label),
                    "Count": str(count),
                    "Value per coin": (
                        self._config.format_currency(denomination_value)
                        if denomination_value is not None
                        else "Unknown"
                    ),
                    "Subtotal": (
                        self._config.format_currency(subtotal_cents)
                        if denomination_value is not None
                        else "Unknown"
                    ),
                }
            )

        return DetectionReport(
            summary_rows=summary_rows,
            detection_rows=detection_rows,
            total_count=sum(counts.values()),
            total_value_cents=total_value_cents,
        )

    def _sort_key(self, item: tuple[str, int]) -> tuple[int, str]:
        """Sort denominations by ascending face value, with unknowns last."""
        label, _ = item
        return (self._config.denomination_cents.get(label, 10**9), label)
