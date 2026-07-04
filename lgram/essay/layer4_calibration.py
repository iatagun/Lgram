"""
Layer 4: Population Calibration.

Institution/cohort-specific calibration of essay scoring thresholds.
Based on 200-300 human-scored samples per deployment site.

Key metrics:
  - Quadratic Weighted Kappa (QWK) — agreement with human raters
  - Intraclass Correlation Coefficient (ICC) — rater reliability
  - Isotonic regression — recalibration to correct systematic bias
  - Calibration curves — detect and correct over/under-estimation

Rule: System is "NOT READY" for a population if ICC < 0.80.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .metrics import quadratic_weighted_kappa, icc_multirater, has_multiple_raters, build_recalibration_bins


@dataclass
class CalibrationReport:
    population_id: str
    sample_count: int
    qwk: float
    icc: float
    mean_error: float
    std_error: float
    ready: bool
    recommendation: str
    recalibration_bins: List[Tuple[float, float]] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)


class PopulationCalibrator:
    """
    Calibrate the essay scoring system against a target population.

    Requirements:
      - Minimum 200 human-scored samples for high-confidence calibration
      - At least 2 human raters for ICC calculation
      - Both machine scores and human scores on 0-100 scale

    Usage:
        cal = PopulationCalibrator()
        report = cal.calibrate(
            population_id="school_a_grade10",
            machine_scores=[72, 85, 63, ...],
            human_scores=[[70, 75], [88, 82], [60, 65], ...],
        )
        if not report.ready:
            print(report.recommendation)  # "NOT READY — ICC=0.72. Need >=0.80."
    """

    MIN_SAMPLES = 30
    HIGH_SAMPLES = 200
    READY_THRESHOLD_ICC = 0.80
    ACCEPTABLE_THRESHOLD_ICC = 0.70
    READY_THRESHOLD_QWK = 0.70
    ACCEPTABLE_THRESHOLD_QWK = 0.60

    def calibrate(
        self,
        population_id: str,
        machine_scores: List[float],
        human_scores: List[List[float]],
    ) -> CalibrationReport:
        n = len(machine_scores)
        if n != len(human_scores):
            raise ValueError(
                f"Mismatched lengths: {len(machine_scores)} machine vs "
                f"{len(human_scores)} human score sets"
            )

        if n < self.MIN_SAMPLES:
            return CalibrationReport(
                population_id=population_id,
                sample_count=n,
                qwk=0.0,
                icc=0.0,
                mean_error=0.0,
                std_error=0.0,
                ready=False,
                recommendation=(
                    f"INSUFFICIENT DATA — {n} samples, need >= {self.MIN_SAMPLES}. "
                    f"Collect more human-scored essays before calibration."
                ),
            )

        avg_human = [sum(scores) / len(scores) for scores in human_scores]

        qwk = quadratic_weighted_kappa(machine_scores, avg_human)
        icc = icc_multirater(human_scores) if has_multiple_raters(human_scores) else 0.0

        errors = [m - h for m, h in zip(machine_scores, avg_human)]
        mean_error = sum(errors) / n
        variance = sum((e - mean_error) ** 2 for e in errors) / (n - 1) if n > 1 else 0.0
        std_error = math.sqrt(variance)

        icc_ok = icc >= self.READY_THRESHOLD_ICC
        qwk_ok = qwk >= self.READY_THRESHOLD_QWK
        samples_ok = n >= self.HIGH_SAMPLES

        ready = icc_ok and qwk_ok and samples_ok

        if ready:
            rec = (
                f"READY — ICC={icc:.3f} (>= {self.READY_THRESHOLD_ICC}), "
                f"QWK={qwk:.3f} (>= {self.READY_THRESHOLD_QWK}), "
                f"n={n} (>= {self.HIGH_SAMPLES}). "
                f"Mean error: {mean_error:+.1f}. System calibrated for {population_id}."
            )
        else:
            reasons = []
            if not icc_ok:
                reasons.append(f"ICC={icc:.3f} < {self.READY_THRESHOLD_ICC}")
            if not qwk_ok:
                reasons.append(f"QWK={qwk:.3f} < {self.READY_THRESHOLD_QWK}")
            if not samples_ok:
                reasons.append(f"n={n} < {self.HIGH_SAMPLES}")
            rec = (
                f"NOT READY — {', '.join(reasons)}. "
                f"Do not deploy for {population_id} without addressing these."
            )

        recal_bins = build_recalibration_bins(machine_scores, avg_human, 10)

        return CalibrationReport(
            population_id=population_id,
            sample_count=n,
            qwk=round(qwk, 3),
            icc=round(icc, 3),
            mean_error=round(mean_error, 1),
            std_error=round(std_error, 1),
            ready=ready,
            recommendation=rec,
            recalibration_bins=recal_bins,
            details={
                "machine_mean": round(sum(machine_scores) / n, 1),
                "machine_std": round(
                    math.sqrt(sum((x - sum(machine_scores) / n) ** 2 for x in machine_scores) / (n - 1))
                    if n > 1 else 0, 1
                ),
                "human_mean": round(sum(avg_human) / n, 1),
                "human_std": round(
                    math.sqrt(sum((x - sum(avg_human) / n) ** 2 for x in avg_human) / (n - 1))
                    if n > 1 else 0, 1
                ),
                "error_distribution": {
                    "within_5": sum(1 for e in errors if abs(e) <= 5) / n,
                    "within_10": sum(1 for e in errors if abs(e) <= 10) / n,
                    "within_15": sum(1 for e in errors if abs(e) <= 15) / n,
                },
                "icc_acceptable": icc >= self.ACCEPTABLE_THRESHOLD_ICC,
                "qwk_acceptable": qwk >= self.ACCEPTABLE_THRESHOLD_QWK,
            },
        )

