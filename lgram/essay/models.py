"""
Data models for CAEAS — Cohesion-Aware Essay Analysis System.

NOT an assessment system. NOT a grader.
This tool provides evidence for teacher consideration.
The teacher's professional judgment is the final authority.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Essay:
    text: str
    title: str = ""
    genre: str = "essay"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.text or not self.text.strip():
            raise ValueError("Essay text must be non-empty.")

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ") + ("..." if len(self.text) > 80 else "")
        return f"Essay(title={self.title!r}, text={preview!r})"


@dataclass
class RubricCriterion:
    name: str
    weight: float
    description: str
    max_score: float = 100.0


@dataclass
class LayerResult:
    layer_name: str
    score: float
    normalized_score: float
    raw_details: Dict[str, Any] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)
    confidence_interval: Optional[Tuple[float, float]] = None


@dataclass
class CAEASReport:
    cohesion_score: float
    composite_indicator: float
    confidence_interval: Tuple[float, float]
    layer_results: List[LayerResult]
    suggestion: str
    justification: str
    triggers: List[str] = field(default_factory=list)
    teacher_review_recommended: bool = False
    borderline: bool = False
    essay: Optional[Essay] = None
    cefr_level: str = ""
    cefr_detected: bool = False

    @property
    def overall_cohesion_indicator(self) -> float:
        return self.cohesion_score

    @property
    def overall_score(self) -> float:
        return self.cohesion_score

    @property
    def human_review_recommended(self) -> bool:
        return self.teacher_review_recommended

    @property
    def verdict(self) -> str:
        return self.suggestion

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cohesion_score": self.cohesion_score,
            "composite_indicator": self.composite_indicator,
            "confidence_interval": list(self.confidence_interval),
            "suggestion": self.suggestion,
            "justification": self.justification,
            "borderline": self.borderline,
            "teacher_review_recommended": self.teacher_review_recommended,
            "triggers": self.triggers,
            "layers": [
                {
                    "name": lr.layer_name,
                    "indicator": lr.score,
                    "normalized": lr.normalized_score,
                    "confidence": (
                        list(lr.confidence_interval) if lr.confidence_interval else None
                    ),
                    "evidence": lr.evidence,
                }
                for lr in self.layer_results
            ],
        }


class ContentAnalyzer(ABC):
    """Pluggable content analysis interface (Layer 1)."""

    @abstractmethod
    def analyze(self, essay: Essay, rubric: List[RubricCriterion]) -> LayerResult:
        ...

ContentJudge = ContentAnalyzer
