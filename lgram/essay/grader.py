"""
CAEAS: Cohesion-Aware Writing Feedback Tool.

Target domain: EFL (English as a Foreign Language) writing.
Primary L1: Turkish learners at CEFR B1-B2-C1 levels.

THIS IS A FEEDBACK TOOL, NOT A GRADING SYSTEM.
It provides evidence for teacher consideration. The teacher's
professional judgment is the final authority.

Architecture:
  PreFilter: Grammar/cohesion disambiguation (optional LanguageTool)
  Layer 1: Content Analysis (rubric-specific)
  Layer 2: Cohesion & Organization (Lgram, segment-aware)
  Layer 3: Surface Quality (grammar, vocab, readability)
  Layer 4: CEFR / Population Calibration
  Layer 5: Confidence & Transparency

Supplementary: L1 Transfer Analysis (Turkish-specific patterns)
  - Pro-drop → missing subject pronouns
  - Gender-neutral → he/she confusion
  - Article-less → a/an/the errors
  - SOV → verb-final transfer

Comparison benchmark: Yavuz (2025) — EFL teachers vs ChatGPT/Bard on
5-dimension rubric (grammar, content, organization, style, mechanics).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .cefr_calibration import CEFRCalibrator, ComplexityProfile
from .efl import (
    CEFR_PROFILES,
    EFL_RUBRIC,
    L1TransferAnalyzer,
    L1TransferReport,
    estimate_cefr_level,
    get_cefr_profile,
)
from .layer1_content import MockContentAnalyzer
from .layer2_cohesion import CohesionLayer
from .layer3_surface import SurfaceLayer
from .layer4_calibration import PopulationCalibrator, CalibrationReport
from .layer5_confidence import ConfidenceLayer
from .models import (
    CAEASReport,
    ContentAnalyzer,
    Essay,
    LayerResult,
    RubricCriterion,
)

try:
    from .layer_grammar import GrammarLayer
    _grammar_available = True
except ImportError:
    GrammarLayer = None
    _grammar_available = False

try:
    from .layer_mechanics import MechanicsLayer
    _mechanics_available = True
except ImportError:
    MechanicsLayer = None
    _mechanics_available = False

try:
    from .layer_llm_content import LLMContentAnalyzer
    _llm_available = True
except ImportError:
    LLMContentAnalyzer = None
    _llm_available = False


class CAEASGrader:
    """
    Cohesion-Aware Writing Feedback Tool for EFL.

    Default mode: EFL with 5-dimension rubric, CEFR-aware.
    Can also operate in general mode with custom rubric.

    IMPORTANT: This tool provides EVIDENCE, not GRADES.
    The numeric indicator is an estimate with a confidence interval.
    Teacher judgment is always the final authority.

    Usage:
        # EFL mode (default) — Turkish L1, CEFR-aware
        grader = CAEASGrader()
        report = grader.analyze(Essay(text="...", title="My Essay"))
        print(report.justification)  # teacher-facing evidence

        # With LanguageTool prefilter (more accurate grammar/cohesion split)
        grader = CAEASGrader(use_prefilter=True)
    """

    def __init__(
        self,
        content_analyzer: Optional[ContentAnalyzer] = None,
        rubric: Optional[List[RubricCriterion]] = None,
        surface_model: str = "en_core_web_md",
        borderline_margin: float = 5.0,
        cefr_level: Optional[str] = None,
        l1_language: Optional[str] = None,
        use_grammar: bool = True,
        use_mechanics: bool = True,
        use_llm: bool = True,
    ):
        self._rubric = rubric or EFL_RUBRIC
        self._cohesion = CohesionLayer(model=surface_model)
        self._surface = SurfaceLayer(model=surface_model)
        self.calibrator = PopulationCalibrator()
        self._cefr_calibrator = CEFRCalibrator()
        self._confidence = ConfidenceLayer(borderline_margin=borderline_margin)
        self._calibration: Optional[CalibrationReport] = None

        self._grammar = GrammarLayer() if use_grammar and GrammarLayer else None
        self._mechanics = MechanicsLayer() if use_mechanics and MechanicsLayer else None

        llm = None
        if use_llm and LLMContentAnalyzer:
            llm = LLMContentAnalyzer()
            if llm.available:
                self._content_analyzer: ContentAnalyzer = llm
            else:
                self._content_analyzer = content_analyzer or MockContentAnalyzer()
                llm = None
        else:
            self._content_analyzer = content_analyzer or MockContentAnalyzer()

        if self._grammar and llm and llm.client:
            self._grammar._llm_base_url = str(llm.client.base_url).rstrip("/")
            self._grammar._llm_model = llm.model_name

        self.use_efl = True
        self.cefr_level: Optional[str] = cefr_level
        self.l1_language: Optional[str] = l1_language or "tr"

        self._l1_analyzer = L1TransferAnalyzer() if self.l1_language == "tr" else None

    @property
    def rubric(self) -> List[RubricCriterion]:
        return self._rubric

    @property
    def calibration_ready(self) -> bool:
        return self._calibration is not None and self._calibration.ready

    def set_calibration(self, report: CalibrationReport) -> None:
        self._calibration = report

    def set_content_analyzer(self, analyzer: ContentAnalyzer) -> None:
        self._content_analyzer = analyzer

    def set_rubric(self, rubric: List[RubricCriterion]) -> None:
        self._rubric = rubric

    def enable_l1_analysis(self, language: str = "tr") -> None:
        if language == "tr":
            self._l1_analyzer = L1TransferAnalyzer()
            self.l1_language = language
        else:
            raise ValueError(
                f"Unsupported L1 language: {language}. "
                f"Currently supported: tr (Turkish)."
            )

    def enable_prefilter(self) -> None:  # deprecated, kept for compat
        pass

    def grade(self, essay: Essay) -> CAEASReport:
        return self.analyze(essay)

    def analyze(self, essay: Essay) -> CAEASReport:
        """
        Analyze essay across all 5 rubric dimensions.

        Layers:
          1. Grammar (LanguageTool)  → rubric Grammar
          2. Content (LLM/heuristic) → rubric Content
          3. Cohesion (Lgram)        → rubric Organization
          4. Surface (readability)   → rubric Style
          5. Mechanics (spellcheck)  → rubric Mechanics

        Primary metric: cohesion_score (pure Layer 3, 0-100)
        Supplementary: composite_indicator (all 5 layers blended)
        """
        cefr_level = self.cefr_level
        detected_level = None

        if self.use_efl and cefr_level is None:
            detected_level, _ = estimate_cefr_level(essay.text)
            cefr_level = detected_level

        cefr_profile = None
        if self.use_efl and cefr_level:
            cefr_profile = get_cefr_profile(cefr_level)

        complexity = self._cefr_calibrator.assess_complexity(essay.text)

        l_grammar = self._grammar.evaluate(essay) if self._grammar else _placeholder("Grammar")
        l_content = self._content_analyzer.analyze(essay, self._rubric)
        l_cohesion = self._cohesion.evaluate(essay)
        l_surface = self._surface.evaluate(essay)
        l_mechanics = self._mechanics.evaluate(essay) if self._mechanics else _placeholder("Mechanics")

        if self.use_efl and "cefr_estimate" in l_content.raw_details:
            llm_cefr = l_content.raw_details.get("cefr_estimate", "")
            valid_cefr = {"A1", "A2", "B1", "B2", "C1", "C2"}
            if llm_cefr and llm_cefr in valid_cefr:
                if llm_cefr in ("A1", "A2"):
                    llm_cefr = "B1"
                cefr_level = llm_cefr
                cefr_profile = get_cefr_profile(cefr_level)

        layer_results = [l_grammar, l_content, l_cohesion, l_surface, l_mechanics]
        weights = [c.weight for c in self._rubric]
        total_w = sum(weights)
        if total_w > 0:
            weights = [w / total_w for w in weights]

        composite = sum(lr.score * w for lr, w in zip(layer_results, weights))
        composite = composite * 0.5 + l_cohesion.score * 0.5

        if complexity.adjustment_factor != 1.0:
            composite *= complexity.adjustment_factor
            composite = min(100.0, composite)

        composite = round(composite, 1)
        composite = max(0.0, min(100.0, composite))

        cohesion_score = l_cohesion.score

        if self._calibration and self._calibration.ready:
            cohesion_score = self._apply_calibration(cohesion_score)
            composite = self._apply_calibration(composite)

        cohesion_score = round(cohesion_score, 1)
        cohesion_score = max(0.0, min(100.0, cohesion_score))

        raw_details: Dict[str, Any] = {
            "composite_indicator": composite,
            "complexity": {
                "level": complexity.complexity_level,
                "adjustment": complexity.adjustment_factor,
                "avg_sentence_length": complexity.avg_sentence_length,
            },
        }

        if self._grammar and not self._grammar.available:
            raw_details["grammar_warning"] = (
                "LanguageTool not available. Install: pip install language-tool-python "
                "(requires Java 8+). Grammar scoring disabled."
            )

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
            cohesion_score, layer_results, weights
        )

        suggestion = self._build_suggestion(
            cohesion_score, conf, cefr_level, cefr_profile, l1_transfer
        )

        if not self.calibration_ready:
            if self.use_efl and cefr_profile:
                conf["triggers"].insert(
                    0,
                    f"INFO: Using general CEFR {cefr_level} reference ranges. "
                    f"Institution-specific calibration ({self.calibrator.HIGH_SAMPLES}+ "
                    f"teacher-scored essays) recommended before production use."
                )
            else:
                conf["triggers"].insert(
                    0,
                    "INFO: No institution-specific calibration. General reference "
                    "ranges only. Not suitable for high-stakes decisions."
                )

        if l1_transfer and l1_transfer.overall_transfer_score < 0.5:
            conf["triggers"].append(
                f"L1 transfer patterns detected (score: {l1_transfer.overall_transfer_score:.2f}) "
                f"— some cohesion observations may reflect Turkish L1 transfer rather than "
                f"writing quality issues"
            )

        return CAEASReport(
            cohesion_score=cohesion_score,
            composite_indicator=composite,
            confidence_interval=conf["confidence_interval"],
            layer_results=layer_results,
            suggestion=suggestion,
            justification=conf["justification"],
            triggers=conf["triggers"],
            teacher_review_recommended=conf["human_review_recommended"],
            borderline=conf["borderline"],
            essay=essay,
            cefr_level=cefr_level or "",
            cefr_detected=detected_level is not None,
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

    def _build_suggestion(
        self,
        indicator: float,
        conf: Dict[str, Any],
        cefr_level: Optional[str],
        cefr_profile: Optional[Dict[str, Any]],
        l1_transfer: Optional[L1TransferReport],
    ) -> str:
        if conf["human_review_recommended"]:
            if conf["borderline"] and indicator >= 60:
                return (
                    f"Estimated indicator {indicator:.0f}/100 is near a decision threshold. "
                    f"Teacher review is recommended to determine the appropriate level."
                )
            elif indicator < 40:
                return (
                    f"Cohesion indicators suggest multiple areas needing attention "
                    f"({indicator:.0f}/100). Recommend focusing on grammar and paragraph "
                    f"structure before addressing cohesion."
                )
            else:
                return (
                    f"Inconsistent indicators detected ({indicator:.0f}/100). "
                    f"Teacher review suggested to assess which dimensions need focus."
                )

        suffix = ""
        if cefr_level and cefr_profile:
            lo, hi = cefr_profile["expected_score_range"]
            if indicator >= hi:
                suffix = f" (above typical {cefr_level} range)"
            elif indicator < lo:
                suffix = f" (below typical {cefr_level} range)"
            else:
                suffix = f" (within typical {cefr_level} range)"

        if indicator >= 85:
            return f"Strong cohesion indicators ({indicator:.0f}/100){suffix}."
        elif indicator >= 70:
            return f"Solid cohesion indicators ({indicator:.0f}/100){suffix} with minor areas to review."
        elif indicator >= 55:
            return f"Adequate cohesion ({indicator:.0f}/100){suffix} with some areas for development."
        elif indicator >= 40:
            return f"Developing cohesion ({indicator:.0f}/100){suffix} — several dimensions need attention."
        else:
            return f"Limited cohesion indicators ({indicator:.0f}/100){suffix} — multiple dimensions need development."

    def analyze_batch(self, essays: List[Essay]) -> List[CAEASReport]:
        return [self.analyze(e) for e in essays]

    grade_batch = analyze_batch


def _placeholder(name: str) -> LayerResult:
    return LayerResult(
        layer_name=name,
        score=50.0,
        normalized_score=0.5,
        raw_details={"note": f"{name} layer not loaded (optional dependency missing)"},
        evidence=[f"{name} layer not available — install optional dependencies"],
        confidence_interval=(40.0, 60.0),
    )
