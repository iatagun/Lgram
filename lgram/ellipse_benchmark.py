"""
ELLIPSE Corpus Benchmark — external validation of the cohesion metric.

ELLIPSE (Crossley et al. 2023) contains ~3,900 essays by English language
learners, human-scored on six analytic dimensions INCLUDING COHESION
(1.0-5.0 in 0.5 steps, double-rated). This is the most direct public test
of CAEAS's core claim: that its Centering Theory cohesion score tracks
human judgments of cohesion.

GATE 1 criteria (see docs/CAEAS_DEVELOPMENT_PLAN.md):
  - Pearson r(machine cohesion, human cohesion) >= 0.5
  - AND machine r must beat the word-count baseline by >= 0.1
    (a metric that loses to essay length carries no real signal)

Data: https://github.com/scrosseye/ELLIPSE-Corpus (public)
Expected at benchmark_data/ellipse_train.csv (gitignored).

Usage:
    python -m lgram.ellipse_benchmark --sample 200
    python -m lgram.ellipse_benchmark --sample 200 --model en_core_web_md

Scores are cached per text_id in benchmark_data/ellipse_scores_cache.json,
so repeated runs accumulate toward the full corpus.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .essay.metrics import quadratic_weighted_kappa

DEFAULT_CSV = Path("benchmark_data/ellipse_train.csv")
DEFAULT_CACHE = Path("benchmark_data/ellipse_scores_cache.json")

# bump when the cohesion scoring formula changes — cached scores from an
# older formula must not be mixed into a new run
SCORING_VERSION = "2.3"


def pearson(x: List[float], y: List[float]) -> float:
    n = len(x)
    if n < 3:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return 0.0
    return cov / (sx * sy)


def _ranks(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(x: List[float], y: List[float]) -> float:
    if len(x) < 3:
        return 0.0
    return pearson(_ranks(x), _ranks(y))


@dataclass
class EllipseResult:
    n: int
    pearson_r: float
    spearman_rho: float
    qwk: float
    baseline_wordcount_r: float
    baseline_sentcount_r: float
    gate1_pass: bool
    model: str
    scoring_version: str = SCORING_VERSION
    details: Dict[str, float] = field(default_factory=dict)


class EllipseBenchmark:
    """Correlate CAEAS cohesion scores with ELLIPSE human cohesion ratings."""

    def __init__(
        self,
        csv_path: Path = DEFAULT_CSV,
        cache_path: Path = DEFAULT_CACHE,
        model: str = "en_core_web_md",
    ):
        self.csv_path = Path(csv_path)
        self.cache_path = Path(cache_path)
        self.model = model
        self._ta = None

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def load_rows(self) -> List[Dict[str, str]]:
        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"{self.csv_path} not found. Download ELLIPSE_Final_github_train.csv "
                f"from https://github.com/scrosseye/ELLIPSE-Corpus and place it there."
            )
        with open(self.csv_path, encoding="utf-8") as f:
            return [
                row
                for row in csv.DictReader(f)
                if row.get("full_text", "").strip() and row.get("Cohesion")
            ]

    def _load_cache(self) -> Dict[str, Dict[str, float]]:
        if not self.cache_path.exists():
            return {}
        try:
            data = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        if data.get("scoring_version") != SCORING_VERSION:
            return {}
        if data.get("model") != self.model:
            return {}
        entries = data.get("scores", {})
        return entries if isinstance(entries, dict) else {}

    def _save_cache(self, scores: Dict[str, Dict[str, float]]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "scoring_version": SCORING_VERSION,
            "model": self.model,
            "scores": scores,
        }
        self.cache_path.write_text(json.dumps(payload), encoding="utf-8")

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _analyzer(self):
        if self._ta is None:
            from .analyzer import TextAnalyzer

            self._ta = TextAnalyzer(self.model)
        return self._ta

    def score_sample(
        self,
        sample_size: Optional[int] = None,
        seed: int = 42,
        progress: bool = True,
    ) -> List[Dict[str, float]]:
        """Score a (deterministic) sample of essays, using/updating the cache.

        Returns one record per essay:
        {machine, human_cohesion, human_overall, words, sentences}
        """
        rows = self.load_rows()
        rng = random.Random(seed)
        rng.shuffle(rows)
        if sample_size is not None:
            rows = rows[:sample_size]

        cache = self._load_cache()
        records: List[Dict[str, float]] = []
        dirty = 0
        t0 = time.time()

        for i, row in enumerate(rows):
            tid = row["text_id_kaggle"]
            cached = cache.get(tid)
            if cached is None:
                report = self._analyzer().analyze(row["full_text"])
                cached = {
                    "machine": report.overall_cohesion,
                    "words": float(report.word_count),
                    "sentences": float(report.sentence_count),
                }
                cache[tid] = cached
                dirty += 1
                if dirty % 25 == 0:
                    self._save_cache(cache)
            records.append(
                {
                    "machine": cached["machine"],
                    "human_cohesion": float(row["Cohesion"]),
                    "human_overall": float(row["Overall"]),
                    "words": cached["words"],
                    "sentences": cached["sentences"],
                }
            )
            if progress and (i + 1) % 25 == 0:
                rate = (i + 1) / max(time.time() - t0, 0.001)
                print(
                    f"  {i + 1}/{len(rows)} scored ({rate:.1f}/s)",
                    file=sys.stderr,
                    flush=True,
                )

        if dirty:
            self._save_cache(cache)
        return records

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, sample_size: Optional[int] = None, seed: int = 42
    ) -> EllipseResult:
        records = self.score_sample(sample_size=sample_size, seed=seed)

        machine = [r["machine"] for r in records]
        human = [r["human_cohesion"] for r in records]
        words = [r["words"] for r in records]
        sents = [r["sentences"] for r in records]

        r_machine = pearson(machine, human)
        rho = spearman(machine, human)
        qwk = quadratic_weighted_kappa(machine, human)
        r_words = pearson(words, human)
        r_sents = pearson(sents, human)

        gate1 = r_machine >= 0.5 and (r_machine - abs(r_words)) >= 0.1

        return EllipseResult(
            n=len(records),
            pearson_r=round(r_machine, 4),
            spearman_rho=round(rho, 4),
            qwk=round(qwk, 4),
            baseline_wordcount_r=round(r_words, 4),
            baseline_sentcount_r=round(r_sents, 4),
            gate1_pass=gate1,
            model=self.model,
            details={
                "machine_vs_overall_r": round(
                    pearson(machine, [r["human_overall"] for r in records]), 4
                ),
                "machine_mean": round(sum(machine) / len(machine), 4),
                "human_mean": round(sum(human) / len(human), 4),
            },
        )

    def report(self, result: EllipseResult) -> str:
        lines = [
            "=" * 64,
            "  ELLIPSE COHESION BENCHMARK (external validation)",
            f"  n={result.n}  |  model={result.model}  |  "
            f"scoring v{result.scoring_version}",
            "=" * 64,
            "  Machine cohesion vs human cohesion:",
            f"    Pearson r     : {result.pearson_r:+.3f}",
            f"    Spearman rho  : {result.spearman_rho:+.3f}",
            f"    QWK           : {result.qwk:+.3f}",
            "  Baselines (must be beaten by >= 0.10):",
            f"    word count r  : {result.baseline_wordcount_r:+.3f}",
            f"    sent count r  : {result.baseline_sentcount_r:+.3f}",
            f"  Machine vs human OVERALL r: "
            f"{result.details.get('machine_vs_overall_r', 0):+.3f}",
            "-" * 64,
            f"  GATE 1: {'PASS' if result.gate1_pass else 'FAIL'} "
            f"(need r >= 0.50 and margin >= 0.10 over word-count baseline)",
            "=" * 64,
        ]
        return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="ELLIPSE cohesion benchmark")
    parser.add_argument("--sample", type=int, default=200, help="sample size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default="en_core_web_md")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    args = parser.parse_args()

    bench = EllipseBenchmark(csv_path=Path(args.csv), model=args.model)
    result = bench.evaluate(sample_size=args.sample, seed=args.seed)
    print(bench.report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
