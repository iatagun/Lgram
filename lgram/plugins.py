"""
Plugin system for analysis methods.

Each analysis method (EntityGrid, TextTiling, LexicalChain, etc.) registers
as a plugin. TextAnalyzer discovers and invokes them through the registry.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type


@dataclass
class AnalysisResult:
    name: str
    score: float
    data: Dict[str, Any]


class AnalysisPlugin(ABC):
    """Base class for analysis plugins."""

    name: str = "base"
    description: str = ""

    @abstractmethod
    def analyze(self, text: str, analyzer: Any) -> AnalysisResult:
        ...


class PluginRegistry:
    """Registry for analysis method plugins."""

    def __init__(self):
        self._plugins: Dict[str, Type[AnalysisPlugin]] = {}

    def register(self, plugin_class: Type[AnalysisPlugin]) -> None:
        instance = plugin_class()
        self._plugins[instance.name] = plugin_class

    def get(self, name: str) -> Optional[Type[AnalysisPlugin]]:
        return self._plugins.get(name)

    def list(self) -> List[str]:
        return list(self._plugins.keys())

    def run_all(self, text: str, analyzer: Any) -> Dict[str, AnalysisResult]:
        results: Dict[str, AnalysisResult] = {}
        for name, plugin_cls in self._plugins.items():
            try:
                plugin = plugin_cls()
                results[name] = plugin.analyze(text, analyzer)
            except Exception:
                logging.getLogger(__name__).exception("Plugin %s failed", name)
        return results

    @property
    def names(self) -> List[str]:
        return self.list()


registry = PluginRegistry()


def register_plugin(cls: Type[AnalysisPlugin]) -> Type[AnalysisPlugin]:
    registry.register(cls)
    return cls


class EntityGridPlugin(AnalysisPlugin):
    name = "entity_grid"
    description = "Entity Grid Model (Barzilay & Lapata 2005)"

    def analyze(self, text: str, analyzer: Any) -> AnalysisResult:
        result = analyzer.entity_grid_score(text)
        return AnalysisResult(
            name=self.name,
            score=result.score,
            data={"entities": result.entities, "matrix": result.matrix},
        )


class LexicalChainPlugin(AnalysisPlugin):
    name = "lexical_chain"
    description = "Lexical Chain Approximation"

    def analyze(self, text: str, analyzer: Any) -> AnalysisResult:
        score = analyzer.lexical_chain_score(text)
        return AnalysisResult(name=self.name, score=score, data={})


class CohesionTrendPlugin(AnalysisPlugin):
    name = "cohesion_trend"
    description = "Sliding Window Cohesion Trend"

    def analyze(self, text: str, analyzer: Any) -> AnalysisResult:
        result = analyzer.cohesion_trend(text)
        return AnalysisResult(
            name=self.name,
            score=result["mean"],
            data={"trend": result["trend"], "windows": result["windows"]},
        )


class CohesionHeatmapPlugin(AnalysisPlugin):
    name = "cohesion_heatmap"
    description = "NxN Sentence Similarity Heatmap"

    def analyze(self, text: str, analyzer: Any) -> AnalysisResult:
        result = analyzer.cohesion_heatmap(text, ascii_render=False)
        return AnalysisResult(
            name=self.name,
            score=1.0 - (result["weak_count"] / max(len(result["matrix"]), 1)),
            data={"weak_pairs": result["weak_pairs"]},
        )


class ReadabilityPlugin(AnalysisPlugin):
    name = "readability"
    description = "Flesch Reading Ease"

    def analyze(self, text: str, analyzer: Any) -> AnalysisResult:
        result = analyzer.readability_score(text)
        return AnalysisResult(
            name=self.name,
            score=min(result["flesch_reading_ease"] / 100.0, 1.0),
            data=result,
        )


class TextTilingPlugin(AnalysisPlugin):
    name = "texttiling"
    description = "TextTiling Segmentation (Hearst 1994)"

    def analyze(self, text: str, analyzer: Any) -> AnalysisResult:
        boundaries = analyzer.texttile_segments(text)
        return AnalysisResult(
            name=self.name,
            score=1.0 / max(len(boundaries), 1),
            data={"boundaries": boundaries, "segment_count": len(boundaries)},
        )


class CohesionGraphPlugin(AnalysisPlugin):
    name = "cohesion_graph"
    description = "Sentence Cohesion Graph"

    def analyze(self, text: str, analyzer: Any) -> AnalysisResult:
        result = analyzer.build_cohesion_graph(text)
        return AnalysisResult(
            name=self.name,
            score=result.density,
            data={
                "density": result.density,
                "avg_similarity": result.avg_similarity,
                "communities": result.communities,
                "central_sentences": result.central_sentences,
            },
        )


class DiscoursePlugin(AnalysisPlugin):
    name = "discourse"
    description = "Discourse Relation Analysis (PDTB-lite)"

    def analyze(self, text: str, analyzer: Any) -> AnalysisResult:
        from lgram.discourse import DiscourseAnalyzer
        da = DiscourseAnalyzer(analyzer.nlp)
        result = da.analyze(text)
        return AnalysisResult(
            name=self.name,
            score=result.cohesion_score,
            data={
                "relation_density": result.relation_density,
                "dominant_relation": result.dominant_relation,
                "connective_count": result.connective_count,
                "distribution": result.relation_distribution,
            },
        )


_default_plugins: List[Type[AnalysisPlugin]] = [
    EntityGridPlugin,
    LexicalChainPlugin,
    CohesionTrendPlugin,
    CohesionHeatmapPlugin,
    ReadabilityPlugin,
    TextTilingPlugin,
    CohesionGraphPlugin,
    DiscoursePlugin,
]


def register_default_plugins() -> PluginRegistry:
    for plugin_cls in _default_plugins:
        registry.register(plugin_cls)
    return registry
