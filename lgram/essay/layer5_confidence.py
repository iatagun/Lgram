"""
Layer 5: Confidence & Transparency.

Produces confidence information, observation text, and teacher-review
recommendations. Ensures the system provides evidence — it never claims
to make decisions. The teacher's judgment is final.

Key outputs:
  - Overall confidence interval (e.g., "estimated indicator 72 ± 5")
  - Borderline detection (near attention thresholds)
  - Teacher review recommendations
  - Traceable, citation-style evidence summary
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from .models import LayerResult


class ConfidenceLayer:
    """
    Computes aggregate confidence intervals, borderline detection,
    and teacher review recommendations from Layer 1-3 results.
    """

    DEFAULT_BORDERLINE_MARGIN = 5.0
    CRITICAL_BOUNDARIES = [50.0, 60.0, 65.0, 70.0, 75.0, 80.0, 90.0]
    REVIEW_SCORE_GAP = 20.0

    def __init__(self, borderline_margin: float = 5.0):
        self._borderline_margin = borderline_margin

    def analyze(
        self,
        overall_score: float,
        layer_results: List[LayerResult],
        rubric_weights: Optional[List[float]] = None,
    ) -> dict:
        """
        Returns:
          - confidence_interval: (low, high)
          - borderline: bool
          - triggers: list of trigger reasons
          - human_review_recommended: bool
          - justification: str
        """
        ci = self._aggregate_confidence(overall_score, layer_results, rubric_weights)
        borderline = self._detect_borderline(overall_score, ci)
        triggers = self._detect_triggers(overall_score, layer_results)
        review = borderline or bool(triggers)
        justification = self._build_justification(
            overall_score, ci, layer_results, triggers
        )

        return {
            "confidence_interval": ci,
            "borderline": borderline,
            "triggers": triggers,
            "human_review_recommended": review,
            "justification": justification,
        }

    def _aggregate_confidence(
        self,
        overall_score: float,
        layer_results: List[LayerResult],
        rubric_weights: Optional[List[float]] = None,
    ) -> Tuple[float, float]:
        """Combine layer confidence intervals into an overall CI."""
        if not layer_results:
            sem = 5.0
            return (max(0, overall_score - sem * 2), min(100, overall_score + sem * 2))

        if rubric_weights is None:
            rubric_weights = [1.0 / len(layer_results)] * len(layer_results)

        weighted_variance = 0.0
        total_weight = 0.0

        for i, lr in enumerate(layer_results):
            w = rubric_weights[i] if i < len(rubric_weights) else 1.0
            total_weight += w
            sem = self._layer_sem(lr)
            weighted_variance += w * (sem**2)

        if total_weight > 0:
            weighted_variance /= total_weight

        sem = math.sqrt(weighted_variance) if weighted_variance > 0 else 3.0
        margin = sem * 2
        low = max(0.0, overall_score - margin)
        high = min(100.0, overall_score + margin)

        if high - low < 2.0:
            pad = 1.0
            low = max(0.0, overall_score - pad)
            high = min(100.0, overall_score + pad)

        return (round(low, 1), round(high, 1))

    def _layer_sem(self, lr: LayerResult) -> float:
        ci = lr.confidence_interval
        if ci is None:
            return 5.0
        return (ci[1] - ci[0]) / 4.0

    def _detect_borderline(self, score: float, ci: Tuple[float, float]) -> bool:
        """Detect if indicator is near a common decision threshold."""
        for boundary in self.CRITICAL_BOUNDARIES:
            if abs(score - boundary) <= self._borderline_margin:
                return True
        low, high = ci
        for boundary in self.CRITICAL_BOUNDARIES:
            if low <= boundary <= high:
                return True
        return False

    def _detect_triggers(
        self, overall_score: float, layer_results: List[LayerResult]
    ) -> List[str]:
        """Detect conditions that should trigger teacher review."""
        triggers = []

        scores = [lr.score for lr in layer_results]
        if scores and (max(scores) - min(scores)) > self.REVIEW_SCORE_GAP:
            largest = ""
            smallest = ""
            for lr in layer_results:
                if lr.score == max(scores):
                    largest = lr.layer_name
                if lr.score == min(scores):
                    smallest = lr.layer_name
            triggers.append(
                f"Indicator gap > {self.REVIEW_SCORE_GAP:.0f} between "
                f"'{largest}' ({max(scores):.0f}) and "
                f"'{smallest}' ({min(scores):.0f}) — may warrant teacher attention"
            )

        for lr in layer_results:
            if lr.score < 40:
                triggers.append(
                    f"'{lr.layer_name}' indicator notably low ({lr.score:.0f}/100)"
                )
            elif lr.score < 55:
                if lr.confidence_interval:
                    ci_lo, ci_hi = lr.confidence_interval
                    if ci_hi - ci_lo > 30:
                        triggers.append(
                            f"'{lr.layer_name}' borderline ({lr.score:.0f}) "
                            f"with wide uncertainty ({ci_lo:.0f}–{ci_hi:.0f})"
                        )

        if overall_score < 30:
            triggers.append(
                f"Overall cohesion indicator notably low ({overall_score:.0f}/100)"
            )

        return triggers

    def _build_justification(
        self,
        overall_score: float,
        ci: Tuple[float, float],
        layer_results: List[LayerResult],
        triggers: List[str],
    ) -> str:
        """Build traceable evidence summary for teacher review."""
        parts = [
            f"Cohesion analysis complete. Estimated cohesion: {overall_score:.1f}/100 "
            f"(CI: {ci[0]:.0f}–{ci[1]:.0f}).",
            "",
            "Observations by dimension:",
        ]

        for lr in layer_results:
            ci_str = ""
            if lr.confidence_interval:
                ci_str = f" [{lr.confidence_interval[0]:.0f}–{lr.confidence_interval[1]:.0f}]"
            parts.append(f"  • {lr.layer_name}: {lr.score:.1f}/100{ci_str}")
            for e in lr.evidence[:2]:
                parts.append(f"    -> {e}")

        if triggers:
            parts.append("")
            parts.append("Suggested teacher review points:")
            for t in triggers:
                parts.append(f"  [!] {t}")

        parts.append("")
        parts.append(
            "NOTE: This is automated feedback for teacher consideration. "
            "No automated tool can replace professional judgment in writing evaluation. "
            "The teacher's assessment is the final authority."
        )

        return "\n".join(parts)
