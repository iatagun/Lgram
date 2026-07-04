"""
Shared statistical metrics for calibrators.

Quadratic Weighted Kappa (QWK) and Intraclass Correlation Coefficient (ICC)
used by both PopulationCalibrator (layer4) and CEFRCalibrator.
"""

from __future__ import annotations

import math
from typing import List, Tuple


def quadratic_weighted_kappa(predicted: List[float], actual: List[float]) -> float:
    n = len(predicted)
    if n < 2:
        return 0.0

    min_v = min(min(predicted), min(actual))
    max_v = max(max(predicted), max(actual))
    bins = min(10, max(2, int(math.sqrt(n))))

    if max_v == min_v:
        return 1.0

    bw = (max_v - min_v) / bins
    if bw == 0:
        return 1.0

    def _bin(v: float) -> int:
        return min(bins - 1, max(0, int((v - min_v) / bw)))

    hist = [[0] * bins for _ in range(bins)]
    for p, a in zip(predicted, actual):
        hist[_bin(p)][_bin(a)] += 1

    total = sum(sum(row) for row in hist)
    if total == 0:
        return 0.0

    weights = [[0.0] * bins for _ in range(bins)]
    for i in range(bins):
        for j in range(bins):
            weights[i][j] = ((i - j) / (bins - 1)) ** 2 if bins > 1 else 0.0

    observed = sum(
        hist[i][j] * weights[i][j]
        for i in range(bins) for j in range(bins)
    ) / total

    row_sums = [sum(row) for row in hist]
    col_sums = [sum(hist[i][j] for i in range(bins)) for j in range(bins)]
    expected = sum(
        row_sums[i] * col_sums[j] * weights[i][j]
        for i in range(bins) for j in range(bins)
    ) / (total * total)

    return 1.0 - observed / expected if expected > 0 else 1.0


def icc_multirater(scores: List[List[float]]) -> float:
    """ICC(2,k): two-way random effects, absolute agreement, average of k raters."""
    n = len(scores)
    if n < 2:
        return 0.0

    k = len(scores[0])
    if k < 2:
        return 0.0

    all_vals = [v for ratings in scores for v in ratings]
    grand_mean = sum(all_vals) / len(all_vals)

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


def has_multiple_raters(scores: List[List[float]]) -> bool:
    """Check if score data contains multiple independent raters."""
    return bool(scores) and all(len(s) >= 2 for s in scores)


def build_recalibration_bins(
    machine_scores: List[float],
    human_scores: List[float],
    num_bins: int = 10,
) -> List[Tuple[float, float]]:
    """Build isotonic recalibration bins for systematic bias correction."""
    if len(machine_scores) < num_bins * 2:
        return []

    min_val, max_val = min(machine_scores), max(machine_scores)
    if max_val == min_val:
        return [(min_val, sum(human_scores) / len(human_scores))]

    bw = (max_val - min_val) / num_bins
    bins: List[Tuple[float, float]] = []

    for i in range(num_bins):
        lo = min_val + i * bw
        hi = lo + bw
        in_bin = [
            h for m, h in zip(machine_scores, human_scores)
            if lo <= m < hi or (i == num_bins - 1 and m == hi)
        ]
        if in_bin:
            bins.append((round((lo + hi) / 2, 1), round(sum(in_bin) / len(in_bin), 1)))

    return bins
