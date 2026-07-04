"""
CAEAS: Coherence-Aware Essay Assessment System.

Multi-layer evidence system for essay assessment:
  Layer 1: Content & Argument (LLM-judge, rubric-specific)
  Layer 2: Cohesion & Organization (Lgram, segment-aware)
  Layer 3: Surface Quality (grammar, vocab, readability)
  Layer 4: Population Calibration (institution-specific thresholds)
  Layer 5: Confidence & Transparency (justification, uncertainty, human trigger)
"""

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
]
