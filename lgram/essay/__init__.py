"""
CAEAS: Cohesion-Aware Essay Analysis System.

Target domain: EFL (English as a Foreign Language) writing.
Primary L1: Turkish learners at CEFR B1-B2-C1 levels.

This is a FEEDBACK tool, not a grading system.
It provides evidence for teacher consideration — the teacher's
professional judgment is always the final authority.

5-layer full rubric system:
  Grammar — LanguageTool (6000+ rules, real grammar check)
  Content — LM Studio / local LLM (no API costs, private)
  Organization — Lgram Centering Theory (cohesion)
  Style — Vocabulary + readability metrics
  Mechanics — pyspellchecker (spelling, punctuation, capitalization)

Supplementary: L1 Transfer Analysis (Turkish-specific patterns)
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

try:
    from .layer_grammar import GrammarLayer
except ImportError:
    GrammarLayer = None
try:
    from .layer_mechanics import MechanicsLayer
except ImportError:
    MechanicsLayer = None
try:
    from .layer_llm_content import LLMContentAnalyzer
except ImportError:
    LLMContentAnalyzer = None

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
    "GrammarLayer",
    "MechanicsLayer",
    "LLMContentAnalyzer",
]
