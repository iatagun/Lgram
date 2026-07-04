"""
CAEAS: Cohesion-Aware Writing Feedback Tool.

Target domain: EFL (English as a Foreign Language) writing.
Primary L1: Turkish learners at CEFR B1-B2-C1 levels.

This is a FEEDBACK tool, not a grading system.
It provides evidence for teacher consideration — the teacher's
professional judgment is always the final authority.

Multi-layer evidence system:
  Layer 1: Content Analysis (rubric-specific)
  Layer 2: Cohesion & Organization (Lgram, segment-aware)
  Layer 3: Surface Quality (grammar, vocab, readability)
  Layer 4: CEFR / Population Calibration
  Layer 5: Confidence & Transparency

PreFilter: Grammar/cohesion disambiguation (spaCy fragility mitigation)
Supplementary: L1 Transfer Analysis (Turkish-specific patterns)

Standard EFL rubric (5 dimensions):
  Grammar, Content, Organization, Style & Expression, Mechanics

Comparison benchmark: Yavuz (2025) — EFL teachers vs ChatGPT/Bard.
"""

from .cefr_calibration import CEFRCalibrator, LevelCalibration, ComplexityProfile
from .efl import (
    EFL_RUBRIC,
    CEFR_PROFILES,
    L1TransferAnalyzer,
    L1TransferReport,
    estimate_cefr_level,
    get_cefr_profile,
)
from .export import DataExporter, ExportBundle
from .models import (
    Essay,
    RubricCriterion,
    LayerResult,
    CAEASReport,
    ContentAnalyzer,
    ContentJudge,
)
from .grader import CAEASGrader
from .layer1_content import MockContentAnalyzer, MockContentAnalyzer as MockContentJudge
from .layer2_cohesion import CohesionLayer, SegmentAnalyzer
from .layer3_surface import SurfaceLayer
from .layer4_calibration import PopulationCalibrator, CalibrationReport
from .layer5_confidence import ConfidenceLayer
from .prefilter import PreFilter, PreFilterReport
from .typology import ErrorTypology, TypologyReport, ErrorCategory

__all__ = [
    "Essay",
    "RubricCriterion",
    "LayerResult",
    "CAEASReport",
    "ContentAnalyzer",
    "ContentJudge",
    "MockContentAnalyzer",
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
    "PreFilter",
    "PreFilterReport",
    "CEFRCalibrator",
    "LevelCalibration",
    "ComplexityProfile",
    "DataExporter",
    "ExportBundle",
    "ErrorTypology",
    "TypologyReport",
    "ErrorCategory",
]
