"""
Data models for CAEAS essay assessment system.
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
    overall_score: float
    confidence_interval: Tuple[float, float]
    layer_results: List[LayerResult]
    verdict: str
    justification: str
    triggers: List[str] = field(default_factory=list)
    human_review_recommended: bool = False
    borderline: bool = False
    essay: Optional[Essay] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "confidence_interval": list(self.confidence_interval),
            "verdict": self.verdict,
            "justification": self.justification,
            "borderline": self.borderline,
            "human_review_recommended": self.human_review_recommended,
            "triggers": self.triggers,
            "layers": [
                {
                    "name": lr.layer_name,
                    "score": lr.score,
                    "normalized": lr.normalized_score,
                    "confidence": (
                        list(lr.confidence_interval) if lr.confidence_interval else None
                    ),
                    "evidence": lr.evidence,
                }
                for lr in self.layer_results
            ],
        }


class ContentJudge(ABC):
    """Pluggable content/argument evaluator (Layer 1)."""

    @abstractmethod
    def evaluate(self, essay: Essay, rubric: List[RubricCriterion]) -> LayerResult:
        ...
