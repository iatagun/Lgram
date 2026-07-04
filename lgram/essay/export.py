"""
Data Collection & Export Module.

Structured export for:
  - Calibration data (machine + human scores per level)
  - Error typology (frequency, severity, L1 correlation)
  - Cross-institutional comparison
  - Research-ready anonymized datasets

Format: JSON by default, CSV for tabular exports.
Aligned with ICLE/TOEFL11 conventions where possible.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .models import CAEASReport, Essay


@dataclass
class ExportBundle:
    version: str = "0.2"
    export_date: str = field(default_factory=lambda: datetime.now().isoformat())
    total_essays: int = 0
    cefr_distribution: Dict[str, int] = field(default_factory=dict)
    reports: List[Dict[str, Any]] = field(default_factory=list)


class DataExporter:
    """
    Export research-quality data for calibration and publication.

    Usage:
        exporter = DataExporter()
        bundle = exporter.create_bundle(essays, reports)

        exporter.to_json(bundle, "calibration_data.json")
        exporter.to_csv_typology(bundle, "error_types.csv")

        # Anonymized for sharing
        anon = exporter.anonymize(bundle)
        exporter.to_json(anon, "anonymized_data.json")
    """

    def create_bundle(
        self,
        essays: List[Essay],
        reports: List[CAEASReport],
        institution_id: Optional[str] = None,
    ) -> ExportBundle:
        bundle = ExportBundle(total_essays=len(reports))

        for essay, report in zip(essays, reports):
            cefr = report.cefr_level or "unknown"

            bundle.cefr_distribution[cefr] = (
                bundle.cefr_distribution.get(cefr, 0) + 1
            )

            record = {
                "title": essay.title,
                "text_length": len(essay.text.split()),
                "estimated_cefr": cefr,
                "overall_cohesion_indicator": report.overall_score,
                "confidence_interval": list(report.confidence_interval),
                "borderline": report.borderline,
                "human_review_recommended": report.human_review_recommended,
                "trigger_count": len(report.triggers),
                "layers": [
                    {
                        "name": lr.layer_name,
                        "indicator": lr.score,
                        "confidence": (
                            list(lr.confidence_interval)
                            if lr.confidence_interval else None
                        ),
                        "evidence_count": len(lr.evidence),
                    }
                    for lr in report.layer_results
                ],
            }

            if institution_id:
                record["institution"] = institution_id

            bundle.reports.append(record)

        return bundle

    def to_json(self, bundle: ExportBundle, filepath: str) -> None:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(bundle.__dict__, f, indent=2, ensure_ascii=False, default=str)

    def to_json_string(self, bundle: ExportBundle) -> str:
        return json.dumps(bundle.__dict__, indent=2, ensure_ascii=False, default=str)

    def to_csv_typology(
        self, bundle: ExportBundle, filepath: str
    ) -> None:
        rows = [["cefr", "score", "ci_low", "ci_high", "borderline", "triggers"]]
        for r in bundle.reports:
            ci = r.get("confidence_interval", [0, 0])
            rows.append([
                r.get("estimated_cefr", ""),
                r.get("overall_cohesion_indicator", 0),
                ci[0] if len(ci) > 0 else 0,
                ci[1] if len(ci) > 1 else 0,
                r.get("borderline", False),
                r.get("trigger_count", 0),
            ])

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def anonymize(self, bundle: ExportBundle) -> ExportBundle:
        import copy
        anon = copy.deepcopy(bundle)
        for r in anon.reports:
            r.pop("title", None)
            r.pop("institution", None)
        return anon

    def compare_institutions(
        self,
        bundle_a: ExportBundle,
        bundle_b: ExportBundle,
    ) -> Dict[str, Any]:
        scores_a = [
            r["overall_cohesion_indicator"] for r in bundle_a.reports
        ]
        scores_b = [
            r["overall_cohesion_indicator"] for r in bundle_b.reports
        ]
        return {
            "institution_a": {
                "count": len(scores_a),
                "mean": sum(scores_a) / max(len(scores_a), 1),
                "min": min(scores_a) if scores_a else 0,
                "max": max(scores_a) if scores_a else 0,
            },
            "institution_b": {
                "count": len(scores_b),
                "mean": sum(scores_b) / max(len(scores_b), 1),
                "min": min(scores_b) if scores_b else 0,
                "max": max(scores_b) if scores_b else 0,
            },
        }
