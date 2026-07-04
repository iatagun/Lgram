"""
CEFR-Level Calibration Pipeline.

Per-level calibration curves for B1, B2, C1 EFL writing.
Separate thresholds per level to prevent the "better writer paradox"
(advanced students attempting complex structures should not be penalized
for taking more risks).

Key concept: Complexity-Adjusted Scoring
  - Low complexity text → cohesion is "easier" → ceiling is lower
  - High complexity text → cohesion is "harder" → threshold is more tolerant
  - This prevents simple-but-safe texts from outscoring ambitious texts
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class LevelCalibration:
    level: str
    sample_count: int
    qwk: float
    icc: float
    cohesion_threshold: float
    rough_shift_tolerance: float
    complexity_adjustment: Dict[str, float] = field(default_factory=dict)
    recalibration_bins: List[Tuple[float, float]] = field(default_factory=list)
    ready: bool = False
    recommendation: str = ""


@dataclass
class ComplexityProfile:
    sentence_count: int
    avg_sentence_length: float
    avg_clause_depth: int
    subordination_ratio: float
    vocabulary_diversity: float
    complexity_level: str
    adjustment_factor: float


class CEFRCalibrator:
    """
    Calibrates cohesion/quality thresholds per CEFR level.

    Requires: 30+ human-scored essays per level for initial calibration.
    Full production readiness: 200+ per level for high confidence.

    Complexity adjustment: texts with higher syntactic complexity
    (more clauses, longer sentences, more subordination) get a
    tolerance bonus — their cohesion is harder to maintain, so
    thresholds are relaxed proportionally.
    """

    MIN_SAMPLES_PER_LEVEL = 30
    HIGH_SAMPLES_PER_LEVEL = 200
    READY_THRESHOLD_QWK = 0.70
    READY_THRESHOLD_ICC = 0.75

    def calibrate_level(
        self,
        level: str,
        machine_scores: List[float],
        human_scores: List[List[float]],
        text_complexities: Optional[List[ComplexityProfile]] = None,
    ) -> LevelCalibration:
        n = len(machine_scores)
        if n != len(human_scores):
            raise ValueError("Mismatched machine/human score lengths")

        if n < self.MIN_SAMPLES_PER_LEVEL:
            return LevelCalibration(
                level=level,
                sample_count=n,
                qwk=0.0,
                icc=0.0,
                cohesion_threshold=0.0,
                rough_shift_tolerance=0.0,
                ready=False,
                recommendation=(
                    f"INSUFFICIENT DATA for {level}: {n} samples, "
                    f"need >= {self.MIN_SAMPLES_PER_LEVEL}."
                ),
            )

        avg_human = [sum(s) / len(s) for s in human_scores]
        qwk = self._qwk(machine_scores, avg_human)
        icc = self._icc(human_scores) if self._has_multiple_raters(human_scores) else 0.0

        cohesion_threshold, rough_shift_tolerance = self._derive_thresholds(
            machine_scores, avg_human
        )

        complexity_adj = self._compute_complexity_adjustment(
            text_complexities, machine_scores, avg_human
        ) if text_complexities else {}

        ready = (
            n >= self.HIGH_SAMPLES_PER_LEVEL
            and qwk >= self.READY_THRESHOLD_QWK
            and icc >= (self.READY_THRESHOLD_ICC if icc > 0 else self.READY_THRESHOLD_QWK)
        )

        if ready:
            rec = f"READY: {level} calibrated with n={n}, QWK={qwk:.3f}, ICC={icc:.3f}."
        elif n >= self.MIN_SAMPLES_PER_LEVEL:
            rec = f"PILOT: {level} has {n} samples (need {self.HIGH_SAMPLES_PER_LEVEL} for production). QWK={qwk:.3f}."
        else:
            rec = f"NOT READY: {level} needs >= {self.MIN_SAMPLES_PER_LEVEL} samples. Current: {n}."

        bins = self._build_bins(machine_scores, avg_human)

        return LevelCalibration(
            level=level,
            sample_count=n,
            qwk=round(qwk, 3),
            icc=round(icc, 3),
            cohesion_threshold=round(cohesion_threshold, 3),
            rough_shift_tolerance=round(rough_shift_tolerance, 3),
            complexity_adjustment=complexity_adj,
            recalibration_bins=bins,
            ready=ready,
            recommendation=rec,
        )

    def calibrate_all_levels(
        self, data: Dict[str, Dict[str, Any]]
    ) -> Dict[str, LevelCalibration]:
        """
        data = {
            "B1": {
                "machine_scores": [...],
                "human_scores": [[...], [...]],
                "complexities": [ComplexityProfile(...), ...],  # optional
            },
            ...
        }
        """
        results = {}
        for level in ["B1", "B2", "C1"]:
            if level in data:
                d = data[level]
                results[level] = self.calibrate_level(
                    level,
                    d["machine_scores"],
                    d["human_scores"],
                    d.get("complexities"),
                )
        return results

    def assess_complexity(self, text: str) -> ComplexityProfile:
        """
        Estimate syntactic complexity for adjustment factor.

        Returns ComplexityProfile with:
          - complexity_level: "low" | "medium" | "high"
          - adjustment_factor: 1.0 = no adjustment, >1.0 = more tolerance
        """
        sentences = _split_sentences(text)
        words = text.split()
        sent_count = len(sentences)
        word_count = len(words)

        if sent_count == 0:
            return ComplexityProfile(
                sentence_count=0, avg_sentence_length=0,
                avg_clause_depth=0, subordination_ratio=0,
                vocabulary_diversity=0, complexity_level="low",
                adjustment_factor=1.0,
            )

        avg_len = word_count / sent_count
        clause_count = sum(
            1 for sent in sentences
            for w in sent.lower().split()
            if w in {"because", "although", "while", "whereas", "unless",
                     "since", "which", "who", "whom", "whose", "that",
                     "after", "before", "when", "if", "though"}
        )
        sub_ratio = clause_count / max(sent_count, 1)

        types = len(set(w.lower() for w in words if w.isalpha()))
        vocab_div = types / max(word_count, 1)

        if avg_len > 18 and sub_ratio > 0.4 and vocab_div > 0.6:
            level = "high"
            adj = 1.15
        elif avg_len > 12 and sub_ratio > 0.2:
            level = "medium"
            adj = 1.05
        else:
            level = "low"
            adj = 0.95

        return ComplexityProfile(
            sentence_count=sent_count,
            avg_sentence_length=round(avg_len, 1),
            avg_clause_depth=int(sub_ratio * sent_count),
            subordination_ratio=round(sub_ratio, 3),
            vocabulary_diversity=round(vocab_div, 3),
            complexity_level=level,
            adjustment_factor=adj,
        )

    def _derive_thresholds(
        self, machine: List[float], human: List[float]
    ) -> Tuple[float, float]:
        if not machine:
            return (0.50, 0.30)
        mean_m = sum(machine) / len(machine)
        mean_h = sum(human) / len(human)
        threshold = (mean_m + mean_h) / 200.0
        tolerance = (mean_m / 100.0) * 0.3
        return (max(0.3, min(0.8, threshold)), max(0.15, min(0.4, tolerance)))

    def _compute_complexity_adjustment(
        self,
        complexities: List[ComplexityProfile],
        machine: List[float],
        human: List[float],
    ) -> Dict[str, float]:
        if len(complexities) != len(machine):
            return {}
        by_level: Dict[str, List[float]] = defaultdict(list)
        for c, m, h in zip(complexities, machine, human):
            by_level.setdefault(c.complexity_level, []).append(h - m)

        result = {}
        for level, diffs in by_level.items():
            if diffs:
                mean_diff = sum(diffs) / len(diffs)
                result[level] = round(mean_diff, 1)
        return result

    def _qwk(self, predicted: List[float], actual: List[float]) -> float:
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

        w = [[0.0] * bins for _ in range(bins)]
        for i in range(bins):
            for j in range(bins):
                w[i][j] = ((i - j) / (bins - 1)) ** 2 if bins > 1 else 0.0

        obs = sum(hist[i][j] * w[i][j] for i in range(bins) for j in range(bins)) / total
        row_sums = [sum(row) for row in hist]
        col_sums = [sum(hist[i][j] for i in range(bins)) for j in range(bins)]
        exp = sum(
            row_sums[i] * col_sums[j] * w[i][j]
            for i in range(bins) for j in range(bins)
        ) / (total * total)

        return 1.0 - obs / exp if exp > 0 else 1.0

    def _icc(self, scores: List[List[float]]) -> float:
        n = len(scores)
        if n < 2:
            return 0.0
        k = len(scores[0])
        if k < 2:
            return 0.0
        gm = sum(sum(r) for r in scores) / (n * k)
        ssb = sum(k * (sum(r) / k - gm) ** 2 for r in scores)
        ssw = sum(sum((v - sum(r) / k) ** 2 for v in r) for r in scores)
        msb = ssb / (n - 1) if n > 1 else 0.0
        msw = ssw / (n * (k - 1)) if k > 1 else 0.0
        if msw == 0:
            return 1.0 if msb > 0 else 0.0
        return max(0.0, min(1.0, (msb - msw) / (msb + (k - 1) * msw)))

    def _has_multiple_raters(self, scores: List[List[float]]) -> bool:
        return bool(scores) and all(len(s) >= 2 for s in scores)

    def _build_bins(
        self, machine: List[float], human: List[float]
    ) -> List[Tuple[float, float]]:
        n = min(10, max(2, len(machine) // 5))
        if len(machine) < n * 2:
            return []
        min_v, max_v = min(machine), max(machine)
        if max_v == min_v:
            return [(min_v, sum(human) / len(human))]
        bw = (max_v - min_v) / n
        bins: List[Tuple[float, float]] = []
        for i in range(n):
            lo, hi = min_v + i * bw, min_v + (i + 1) * bw
            in_bin = [h for m, h in zip(machine, human) if lo <= m < hi or (i == n - 1 and m == hi)]
            if in_bin:
                bins.append((round((lo + hi) / 2, 1), round(sum(in_bin) / len(in_bin), 1)))
        return bins


def _split_sentences(text: str) -> List[str]:
    import re
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
