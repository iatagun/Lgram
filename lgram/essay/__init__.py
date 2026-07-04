"""
CAEAS: Coherence-Aware Essay Assessment System.

Target domain: EFL (English as a Foreign Language) writing assessment.
Primary L1: Turkish learners at CEFR B1-B2-C1 levels.

Multi-layer evidence system:
  Layer 1: Content & Argument (LLM-judge, rubric-specific)
  Layer 2: Cohesion & Organization (Lgram, segment-aware)
  Layer 3: Surface Quality (grammar, vocab, readability)
  Layer 4: Population/CEFR Calibration
  Layer 5: Confidence & Transparency (justification, uncertainty, human trigger)

Supplementary: L1 Transfer Analysis (Turkish-specific)
  - Pro-drop → missing subjects, gender-neutral → he/she confusion
  - Article-less → a/an/the errors, SOV → verb-final transfer

Standard EFL rubric (5 dimensions):
  Grammar, Content, Organization, Style & Expression, Mechanics

Comparison benchmark: Yavuz (2025) — EFL teachers vs ChatGPT/Bard.
"""

from .efl import (
    EFL_RUBRIC,
    CEFR_PROFILES,
    L1TransferAnalyzer,
    L1TransferReport,
    estimate_cefr_level,
    get_cefr_profile,
)
from .models import (
    Essay,
    RubricCriterion,
    LayerResult,
    CAEASReport,
    ContentJudge,
)
from .grader import CAEASGrader
from .layer1_content import MockContentJudge
from .layer2_cohesion import CohesionLayer, SegmentAnalyzer
from .layer3_surface import SurfaceLayer
from .layer4_calibration import PopulationCalibrator, CalibrationReport
from .layer5_confidence import ConfidenceLayer

__all__ = [
    "Essay",
    "RubricCriterion",
    "LayerResult",
    "CAEASReport",
    "ContentJudge",
    "MockContentJudge",
    "CAEASGrader",
    "CohesionLayer",
    "SegmentAnalyzer",
    "SurfaceLayer",
    "PopulationCalibrator",
    "CalibrationReport",
    "ConfidenceLayer",
    "EFL_RUBRIC",
    "CEFR_PROFILES",
    "L1TransferAnalyzer",
    "L1TransferReport",
    "estimate_cefr_level",
    "get_cefr_profile",
]
