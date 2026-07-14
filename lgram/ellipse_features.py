"""
Feature extraction for the ELLIPSE supervised experiment.

Extracts three interpretable feature groups per essay:

  A (surface):     word/sentence/paragraph counts, sentence length stats,
                   Guiraud lexical diversity, run-on flag
  B (connective):  discourse connective density and diversity, pronoun density
  C (centering):   Centering Theory transition distribution, current cohesion
                   score, entity-grid score, entity density

The supervised experiment (ellipse_supervised.py) tests whether group C adds
incremental validity over A and B against human cohesion ratings. Extraction
is cached per text_id so long runs can be resumed.

Usage:
    python -m lgram.ellipse_features --sample 1000
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .ellipse_benchmark import DEFAULT_CSV, EllipseBenchmark

DEFAULT_FEATURES_CACHE = Path("benchmark_data/ellipse_features_cache.json")

FEATURES_VERSION = "1.0"

# common English discourse connectives (explicit cohesive devices)
CONNECTIVES = frozenset(
    {
        "however",
        "therefore",
        "moreover",
        "furthermore",
        "consequently",
        "nevertheless",
        "nonetheless",
        "meanwhile",
        "although",
        "though",
        "whereas",
        "because",
        "since",
        "thus",
        "hence",
        "also",
        "additionally",
        "besides",
        "finally",
        "firstly",
        "secondly",
        "thirdly",
        "first",
        "second",
        "third",
        "next",
        "then",
        "instead",
        "otherwise",
        "similarly",
        "likewise",
        "overall",
        "furthermore",
        "despite",
        "unless",
        "while",
    }
)

PRONOUNS = frozenset(
    {
        "he",
        "she",
        "it",
        "they",
        "we",
        "i",
        "you",
        "him",
        "her",
        "them",
        "us",
        "me",
        "his",
        "hers",
        "its",
        "their",
        "theirs",
        "our",
        "ours",
        "this",
        "that",
        "these",
        "those",
    }
)

SURFACE_FEATURES = [
    "words",
    "sentences",
    "paragraphs",
    "avg_sent_len",
    "std_sent_len",
    "guiraud",
    "runon",
]
CONNECTIVE_FEATURES = [
    "conn_density",
    "conn_unique",
    "pronoun_density",
]
CENTERING_FEATURES = [
    "pct_continue",
    "pct_retain",
    "pct_smooth_shift",
    "pct_rough_shift",
    "pct_establish",
    "cohesion_score",
    "entity_grid",
    "entities_per_sent",
]
ALL_FEATURES = SURFACE_FEATURES + CONNECTIVE_FEATURES + CENTERING_FEATURES


def _tokenize_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[a-zA-Z']+", text.lower()) if w]


def surface_features(text: str, report) -> Dict[str, float]:
    words = _tokenize_words(text)
    n = len(words)
    sents = max(report.sentence_count, 1)
    lens: List[int] = []
    for s in re.split(r"[.!?]+", text):
        w = len(s.split())
        if w:
            lens.append(w)
    mean_len = sum(lens) / len(lens) if lens else 0.0
    var = (
        sum((x - mean_len) ** 2 for x in lens) / (len(lens) - 1)
        if len(lens) > 1
        else 0.0
    )
    return {
        "words": float(n),
        "sentences": float(report.sentence_count),
        "paragraphs": float(report.paragraph_count),
        "avg_sent_len": n / sents,
        "std_sent_len": math.sqrt(var),
        "guiraud": len(set(words)) / math.sqrt(n) if n else 0.0,
        # run-on: spaCy sees almost no sentence boundaries in a long text
        "runon": 1.0 if (report.sentence_count <= 2 and n > 120) else 0.0,
    }


def connective_features(text: str) -> Dict[str, float]:
    words = _tokenize_words(text)
    n = max(len(words), 1)
    conn = [w for w in words if w in CONNECTIVES]
    pron = [w for w in words if w in PRONOUNS]
    return {
        "conn_density": 100.0 * len(conn) / n,
        "conn_unique": float(len(set(conn))),
        "pronoun_density": 100.0 * len(pron) / n,
    }


def centering_features(report, grid_score: float) -> Dict[str, float]:
    dist = report.transition_distribution or {}
    total_entities = 0
    sent_count = 0
    for para in report.paragraphs:
        for sa in para.sentences:
            total_entities += len(sa.entities)
            sent_count += 1
    return {
        "pct_continue": dist.get("Continue", 0.0),
        "pct_retain": dist.get("Retain", 0.0),
        "pct_smooth_shift": dist.get("Smooth-Shift", 0.0),
        "pct_rough_shift": dist.get("Rough-Shift", 0.0),
        "pct_establish": dist.get("Establish", 0.0),
        "cohesion_score": report.overall_cohesion,
        "entity_grid": grid_score,
        "entities_per_sent": total_entities / max(sent_count, 1),
    }


class EllipseFeatureExtractor:
    """Extract and cache the full feature set for ELLIPSE essays."""

    def __init__(
        self,
        csv_path: Path = DEFAULT_CSV,
        cache_path: Path = DEFAULT_FEATURES_CACHE,
        model: str = "en_core_web_md",
    ):
        self.bench = EllipseBenchmark(csv_path=csv_path, model=model)
        self.cache_path = Path(cache_path)
        self.model = model
        self._ta = None

    def _analyzer(self):
        if self._ta is None:
            from .analyzer import TextAnalyzer

            self._ta = TextAnalyzer(self.model)
        return self._ta

    def _load_cache(self) -> Dict[str, Dict[str, float]]:
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if (
            data.get("features_version") != FEATURES_VERSION
            or data.get("model") != self.model
        ):
            return {}
        entries = data.get("features", {})
        return entries if isinstance(entries, dict) else {}

    def _save_cache(self, features: Dict[str, Dict[str, float]]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(
            json.dumps(
                {
                    "features_version": FEATURES_VERSION,
                    "model": self.model,
                    "features": features,
                }
            ),
            encoding="utf-8",
        )

    def extract_one(self, text: str) -> Dict[str, float]:
        report = self._analyzer().analyze(text, include_clauses=False)
        try:
            grid = self._analyzer().entity_grid_score(text)
            grid_score = float(grid.score)
        except Exception:
            grid_score = 0.0

        feats: Dict[str, float] = {}
        feats.update(surface_features(text, report))
        feats.update(connective_features(text))
        feats.update(centering_features(report, grid_score))
        return feats

    def extract_sample(
        self,
        sample_size: Optional[int] = None,
        seed: int = 42,
        progress: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """Extract features for a deterministic sample; returns {tid: features}."""
        rows = self.bench.load_rows()
        rng = random.Random(seed)
        rng.shuffle(rows)
        if sample_size is not None:
            rows = rows[:sample_size]

        cache = self._load_cache()
        out: Dict[str, Dict[str, float]] = {}
        dirty = 0
        t0 = time.time()

        for i, row in enumerate(rows):
            tid = row["text_id_kaggle"]
            if tid not in cache:
                cache[tid] = self.extract_one(row["full_text"])
                dirty += 1
                if dirty % 20 == 0:
                    self._save_cache(cache)
                    if progress:
                        rate = (i + 1) / max(time.time() - t0, 0.001)
                        eta_min = (len(rows) - i - 1) / max(rate, 0.001) / 60
                        print(
                            f"  {i + 1}/{len(rows)} extracted "
                            f"({rate:.2f}/s, ETA {eta_min:.0f}m)",
                            file=sys.stderr,
                            flush=True,
                        )
            out[tid] = cache[tid]

        if dirty:
            self._save_cache(cache)
        return out


def main() -> int:
    parser = argparse.ArgumentParser(description="ELLIPSE feature extraction")
    parser.add_argument("--sample", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="en_core_web_md")
    args = parser.parse_args()

    ex = EllipseFeatureExtractor(model=args.model)
    feats = ex.extract_sample(sample_size=args.sample, seed=args.seed)
    print(f"extracted features for {len(feats)} essays -> {ex.cache_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
