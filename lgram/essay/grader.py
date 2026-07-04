"""
CAEAS Grader: Coherence-Aware Essay Assessment System.

Target domain: EFL (English as a Foreign Language) writing assignments.
Primary L1: Turkish.

Five-layer evidence-based essay scoring:
  Layer 1: Content & Argument (LLM-judge, rubric-specific)
  Layer 2: Cohesion & Organization (Lgram, segment-aware)
  Layer 3: Surface Quality (grammar, vocab, readability)
  Layer 4: Population/CEFR Calibration
  Layer 5: Confidence & Transparency

Supplementary: L1 Transfer Analysis (Turkish-specific patterns)
  - Pro-drop → missing subject pronouns
  - Gender-neutral → he/she confusion
  - Article-less → a/an/the errors
  - SOV → verb-final transfer

Architecture principle: The system is an EVIDENCE PROVIDER, not an AUTHORITY.
Final decisions always rest with human reviewers. Borderline cases are
automatically flagged for human review.

Comparison benchmark: Yavuz (2025) — EFL teachers vs ChatGPT/Bard on
5-dimension rubric (grammar, content, organization, style, mechanics).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .efl import (
    CEFR_PROFILES,
    EFL_RUBRIC,
    L1TransferAnalyzer,
    L1TransferReport,
    estimate_cefr_level,
    get_cefr_profile,
)
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


class CAEASGrader:
    """
    Coherence-Aware Essay Assessment System for EFL writing.

    Default mode: EFL with 5-dimension rubric.
    Can also operate in general mode with custom rubric.

    Usage:
        # EFL mode (default) — Turkish L1, CEFR-aware
        grader = CAEASGrader()
        report = grader.grade(Essay(text="...", title="My Essay"))

        # With CEFR level set explicitly
        grader = CAEASGrader(cefr_level="B2")
        report = grader.grade(essay)

        # With population calibration:
        cal = grader.calibrator.calibrate(
            population_id="university_prep_b2",
            machine_scores=[...],
            human_scores=[[...], [...]],
        )
        if cal.ready:
            grader.set_calibration(cal)

        # General mode (no EFL defaults)
        grader = CAEASGrader()
        grader.use_efl = False
    """

    def __init__(
        self,
        content_judge: Optional[ContentJudge] = None,
        rubric: Optional[List[RubricCriterion]] = None,
        surface_model: str = "en_core_web_md",
        borderline_margin: float = 5.0,
        cefr_level: Optional[str] = None,
        l1_language: Optional[str] = None,
    ):
        self._content_judge = content_judge or MockContentJudge()
        self._rubric = rubric or EFL_RUBRIC
        self._cohesion = CohesionLayer()
        self._surface = SurfaceLayer(model=surface_model)
        self.calibrator = PopulationCalibrator()
        self._confidence = ConfidenceLayer(borderline_margin=borderline_margin)
        self._calibration: Optional[CalibrationReport] = None
        self._l1_analyzer = L1TransferAnalyzer() if l1_language == "tr" else None

        self.use_efl = True
        self.cefr_level: Optional[str] = cefr_level
        self.l1_language: Optional[str] = l1_language or "tr"

    @property
    def rubric(self) -> List[RubricCriterion]:
        return self._rubric

    @property
    def calibration_ready(self) -> bool:
        return self._calibration is not None and self._calibration.ready

    def set_calibration(self, report: CalibrationReport) -> None:
        self._calibration = report

    def set_content_judge(self, judge: ContentJudge) -> None:
        self._content_judge = judge

    def set_rubric(self, rubric: List[RubricCriterion]) -> None:
        self._rubric = rubric

    def enable_l1_analysis(self, language: str = "tr") -> None:
        if language == "tr":
            self._l1_analyzer = L1TransferAnalyzer()
            self.l1_language = language

    def grade(self, essay: Essay) -> CAEASReport:
        """
        Run assessment layers and produce an EFL-calibrated report.

        EFL mode adds:
          - CEFR level estimation (if not explicitly set)
          - CEFR-aware thresholds for cohesion/rubric expectations
          - L1 transfer analysis (Turkish, if enabled)
        """
        cefr_level = self.cefr_level
        detected_level = None

        if self.use_efl and cefr_level is None:
            detected_level, _ = estimate_cefr_level(essay.text)
            cefr_level = detected_level

        cefr_profile = None
        if self.use_efl and cefr_level:
            cefr_profile = get_cefr_profile(cefr_level)

        l1 = self._content_judge.evaluate(essay, self._rubric)
        l2 = self._cohesion.evaluate(essay)
        l3 = self._surface.evaluate(essay)

        layer_results = [l1, l2, l3]

        weights = [c.weight for c in self._rubric]
        total_w = sum(weights)
        if total_w > 0:
            weights = [w / total_w for w in weights]

        raw_score = sum(lr.score * w for lr, w in zip(layer_results, weights))

        overall_score = raw_score
        if self._calibration and self._calibration.ready:
            overall_score = self._apply_calibration(raw_score)

        overall_score = round(overall_score, 1)
        overall_score = max(0.0, min(100.0, overall_score))

        raw_details: Dict[str, Any] = {}

        l1_transfer = None
        if self._l1_analyzer:
            l1_transfer = self._l1_analyzer.analyze(essay.text)
            raw_details["l1_transfer"] = {
                "language": self.l1_language,
                "score": l1_transfer.overall_transfer_score,
                "pro_drop_count": len(l1_transfer.pro_drop_issues),
                "gender_issues": len(l1_transfer.gender_pronoun_issues),
                "article_adequacy": l1_transfer.article_estimate.get("adequacy", 1.0),
                "summary": l1_transfer.summary,
            }

        if self.use_efl and cefr_profile:
            raw_details["cefr"] = {
                "level": cefr_level,
                "detected": detected_level,
                "expected_range": cefr_profile["expected_score_range"],
                "cohesion_threshold": cefr_profile["cohesion_threshold"],
            }

        conf = self._confidence.analyze(
            overall_score, layer_results, weights
        )

        verdict = self._build_efl_verdict(
            overall_score, conf, cefr_level, cefr_profile, l1_transfer
        )

        if not self.calibration_ready:
            if self.use_efl and cefr_profile:
                conf["triggers"].insert(
                    0,
                    f"INFO: Using general CEFR {cefr_level} thresholds. "
                    f"Calibrate with {self.calibrator.HIGH_SAMPLES}+ human-scored "
                    f"essays from your institution for production use."
                )
            else:
                conf["triggers"].insert(
                    0,
                    "WARNING: No population calibration. General thresholds only. "
                    "Do NOT use for high-stakes decisions without calibration."
                )

        if l1_transfer and l1_transfer.overall_transfer_score < 0.5:
            conf["triggers"].append(
                f"L1 transfer score low ({l1_transfer.overall_transfer_score:.2f}) — "
                f"significant Turkish transfer patterns detected"
            )

        report = CAEASReport(
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

        return report

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

    def _build_efl_verdict(
        self,
        score: float,
        conf: Dict[str, Any],
        cefr_level: Optional[str],
        cefr_profile: Optional[Dict[str, Any]],
        l1_transfer: Optional[L1TransferReport],
    ) -> str:
        if conf["human_review_recommended"]:
            if conf["borderline"] and score >= 60:
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

        suffix = ""
        if cefr_level and cefr_profile:
            lo, hi = cefr_profile["expected_score_range"]
            if score >= hi:
                above = "above" if score > hi + 5 else "at upper bound of"
                suffix = f" ({above} {cefr_level} expectations)"
            elif score < lo:
                below = "below" if score < lo - 5 else "at lower bound of"
                suffix = f" ({below} {cefr_level} expectations)"
            else:
                suffix = f" (within {cefr_level} range)"

        if score >= 85:
            return f"EXCELLENT ({score:.0f}/100){suffix} — strong performance across all dimensions."
        elif score >= 70:
            return f"GOOD ({score:.0f}/100){suffix} — solid performance with minor areas for improvement."
        elif score >= 55:
            return f"ADEQUATE ({score:.0f}/100){suffix} — meets baseline expectations with some weaknesses."
        elif score >= 40:
            return f"DEVELOPING ({score:.0f}/100){suffix} — several dimensions need attention."
        else:
            return f"WEAK ({score:.0f}/100){suffix} — significant weaknesses across multiple dimensions."

    def grade_batch(
        self, essays: List[Essay]
    ) -> List[CAEASReport]:
        return [self.grade(e) for e in essays]
