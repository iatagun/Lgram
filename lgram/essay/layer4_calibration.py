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

        qwk = self._quadratic_weighted_kappa(machine_scores, avg_human)
        icc = self._icc_multirater(human_scores) if self._has_multiple_raters(human_scores) else 0.0

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

        recal_bins = self._build_recalibration_bins(machine_scores, avg_human, 10)

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

    def _quadratic_weighted_kappa(
        self, predicted: List[float], actual: List[float]
    ) -> float:
        n = len(predicted)
        if n < 2:
            return 0.0

        min_val = min(min(predicted), min(actual))
        max_val = max(max(predicted), max(actual))
        num_bins = min(10, max(2, int(math.sqrt(n))))

        if max_val == min_val:
            return 1.0

        bin_width = (max_val - min_val) / num_bins
        if bin_width == 0:
            bin_width = 1.0

        def _to_bin(v: float) -> int:
            return min(num_bins - 1, max(0, int((v - min_val) / bin_width)))

        hist = [[0] * num_bins for _ in range(num_bins)]
        for p, a in zip(predicted, actual):
            hist[_to_bin(p)][_to_bin(a)] += 1

        total = sum(sum(row) for row in hist)
        if total == 0:
            return 0.0

        weights = [[0.0] * num_bins for _ in range(num_bins)]
        for i in range(num_bins):
            for j in range(num_bins):
                weights[i][j] = ((i - j) / (num_bins - 1)) ** 2 if num_bins > 1 else 0.0

        observed = sum(
            hist[i][j] * weights[i][j]
            for i in range(num_bins)
            for j in range(num_bins)
        ) / total

        row_sums = [sum(row) for row in hist]
        col_sums = [sum(hist[i][j] for i in range(num_bins)) for j in range(num_bins)]
        expected = sum(
            row_sums[i] * col_sums[j] * weights[i][j]
            for i in range(num_bins)
            for j in range(num_bins)
        ) / (total * total)

        if expected == 0:
            return 1.0
        return 1.0 - observed / expected

    def _icc_multirater(self, scores: List[List[float]]) -> float:
        """ICC(2,k) — two-way random effects, absolute agreement, average of k raters."""
        n = len(scores)
        if n < 2:
            return 0.0

        k = len(scores[0])
        if k < 2:
            return 0.0

        all_vals = []
        for ratings in scores:
            all_vals.extend(ratings)

        grand_mean = sum(all_vals) / len(all_vals)
        if grand_mean == 0:
            return 0.0

        ss_between = 0.0
        for ratings in scores:
            subject_mean = sum(ratings) / k
            ss_between += k * (subject_mean - grand_mean) ** 2

        ss_within = 0.0
        for ratings in scores:
            subject_mean = sum(ratings) / k
            for r in ratings:
                ss_within += (r - subject_mean) ** 2

        ms_between = ss_between / (n - 1) if n > 1 else 0.0
        ms_within = ss_within / (n * (k - 1)) if k > 1 else 0.0

        if ms_between == 0.0 and ms_within == 0.0:
            return 1.0

        if ms_within == 0.0:
            return 1.0

        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
        return max(0.0, min(1.0, icc))

    def _has_multiple_raters(self, scores: List[List[float]]) -> bool:
        if not scores:
            return False
        return all(len(s) >= 2 for s in scores)

    def _build_recalibration_bins(
        self,
        machine_scores: List[float],
        human_scores: List[float],
        num_bins: int = 10,
    ) -> List[Tuple[float, float]]:
        """Build isotonic recalibration bins: (machine_score → corrected_score)."""
        if len(machine_scores) < num_bins * 2:
            return []

        min_val = min(machine_scores)
        max_val = max(machine_scores)
        if max_val == min_val:
            return [(min_val, sum(human_scores) / len(human_scores))]

        bin_width = (max_val - min_val) / num_bins
        bins: List[Tuple[float, float]] = []

        for i in range(num_bins):
            low = min_val + i * bin_width
            high = low + bin_width
            in_bin = [
                h for m, h in zip(machine_scores, human_scores)
                if low <= m < high or (i == num_bins - 1 and m == high)
            ]
            if in_bin:
                bins.append((round((low + high) / 2, 1), round(sum(in_bin) / len(in_bin), 1)))

        return bins
