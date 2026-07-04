"""
CAEAS Grader: Coherence-Aware Essay Assessment System.

Five-layer evidence-based essay scoring:
  Layer 1: Content & Argument (LLM-judge, rubric-specific)
  Layer 2: Cohesion & Organization (Lgram, segment-aware)
  Layer 3: Surface Quality (grammar, vocab, readability)
  Layer 4: Population Calibration (institution-specific thresholds)
  Layer 5: Confidence & Transparency (justification, uncertainty, human trigger)

Architecture principle: The system is an EVIDENCE PROVIDER, not an AUTHORITY.
Final decisions always rest with human reviewers. Borderline cases are
automatically flagged for human review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .layer1_content import MockContentJudge
from .layer2_cohesion import CohesionLayer
from .layer3_surface import SurfaceLayer
from .layer4_calibration import PopulationCalibrator, CalibrationReport
from .layer5_confidence import ConfidenceLayer
from .models import (
    CAEASReport,
    ContentJudge,
    Essay,
    LayerResult,
    RubricCriterion,
)


DEFAULT_RUBRIC = [
    RubricCriterion(
        name="Content & Argument",
        weight=0.40,
        description="Thesis clarity, evidence use, reasoning quality, topic development",
    ),
    RubricCriterion(
        name="Organization & Cohesion",
        weight=0.35,
        description="Paragraph structure, sentence flow, transition quality, logical ordering",
    ),
    RubricCriterion(
        name="Language & Conventions",
        weight=0.25,
        description="Grammar, vocabulary range, sentence variety, readability, spelling",
    ),
]


class CAEASGrader:
    """
    Coherence-Aware Essay Assessment System.

    Usage:
        grader = CAEASGrader()
        report = grader.grade(Essay(text="...", title="My Essay"))
        print(report.justification)

        # With population calibration:
        cal = grader.calibrator.calibrate(
            population_id="school_a",
            machine_scores=[...],
            human_scores=[[...], [...]],
        )
        if cal.ready:
            grader.set_calibration(cal)
    """

    def __init__(
        self,
        content_judge: Optional[ContentJudge] = None,
        rubric: Optional[List[RubricCriterion]] = None,
        surface_model: str = "en_core_web_md",
        borderline_margin: float = 5.0,
    ):
        self._content_judge = content_judge or MockContentJudge()
        self.rubric = rubric or DEFAULT_RUBRIC
        self._cohesion = CohesionLayer()
        self._surface = SurfaceLayer(model=surface_model)
        self.calibrator = PopulationCalibrator()
        self._confidence = ConfidenceLayer(borderline_margin=borderline_margin)
        self._calibration: Optional[CalibrationReport] = None

    @property
    def calibration_ready(self) -> bool:
        return self._calibration is not None and self._calibration.ready

    def set_calibration(self, report: CalibrationReport) -> None:
        self._calibration = report

    def set_content_judge(self, judge: ContentJudge) -> None:
        self._content_judge = judge

    def grade(self, essay: Essay) -> CAEASReport:
        """
        Run all five layers and produce a calibrated essay assessment.

        Returns CAEASReport with:
          - overall_score: 0-100
          - confidence_interval: (low, high)
          - verdict: human-readable summary
          - justification: detailed traceable explanation
          - human_review_recommended: whether to trigger review
        """
        l1 = self._content_judge.evaluate(essay, self.rubric)
        l2 = self._cohesion.evaluate(essay)
        l3 = self._surface.evaluate(essay)

        layer_results = [l1, l2, l3]
        weights = [c.weight for c in self.rubric[:3]]

        total_w = sum(weights)
        if total_w > 0:
            weights = [w / total_w for w in weights]

        raw_score = sum(lr.score * w for lr, w in zip(layer_results, weights))

        overall_score = raw_score
        if self._calibration and self._calibration.ready:
            overall_score = self._apply_calibration(raw_score)

        overall_score = round(overall_score, 1)
        overall_score = max(0.0, min(100.0, overall_score))

        conf = self._confidence.analyze(
            overall_score, layer_results, weights
        )

        verdict = self._build_verdict(
            overall_score, conf["borderline"], conf["human_review_recommended"]
        )

        if self._calibration is None or not self._calibration.ready:
            conf["triggers"].insert(
                0,
                "WARNING: No population calibration. General thresholds only. "
                "Do NOT use for high-stakes decisions without calibration."
            )

        return CAEASReport(
            overall_score=overall_score,
            confidence_interval=conf["confidence_interval"],
            layer_results=layer_results,
            verdict=verdict,
            justification=conf["justification"],
            triggers=conf["triggers"],
            human_review_recommended=conf["human_review_recommended"],
            borderline=conf["borderline"],
            essay=essay,
        )

    def _apply_calibration(self, raw_score: float) -> float:
        if self._calibration is None:
            return raw_score
        bins = self._calibration.recalibration_bins
        if not bins:
            return raw_score - self._calibration.mean_error

        if raw_score <= bins[0][0]:
            return max(0.0, raw_score + (bins[0][1] - bins[0][0]))

        if raw_score >= bins[-1][0]:
            return min(100.0, raw_score + (bins[-1][1] - bins[-1][0]))

        for i in range(len(bins) - 1):
            if bins[i][0] <= raw_score <= bins[i + 1][0]:
                t = (raw_score - bins[i][0]) / (bins[i + 1][0] - bins[i][0])
                corrected = bins[i][1] + t * (bins[i + 1][1] - bins[i][1])
                return max(0.0, min(100.0, corrected))

        return raw_score

    def _build_verdict(
        self, score: float, borderline: bool, review: bool
    ) -> str:
        if review:
            if borderline and score >= 60:
                return (
                    f"BORDERLINE ({score:.0f}/100) — near grade boundary. "
                    f"Human review recommended."
                )
            elif score < 40:
                return (
                    f"REVIEW ({score:.0f}/100) — significant issues detected "
                    f"across multiple dimensions."
                )
            else:
                return (
                    f"REVIEW ({score:.0f}/100) — inconsistencies or high "
                    f"uncertainty detected."
                )
        else:
            if score >= 80:
                return f"STRONG ({score:.0f}/100) — good quality across all dimensions."
            elif score >= 65:
                return f"ADEQUATE ({score:.0f}/100) — meets baseline expectations."
            elif score >= 50:
                return f"DEVELOPING ({score:.0f}/100) — needs improvement in some areas."
            else:
                return f"WEAK ({score:.0f}/100) — significant weaknesses detected."

    def grade_batch(
        self, essays: List[Essay]
    ) -> List[CAEASReport]:
        return [self.grade(e) for e in essays]
