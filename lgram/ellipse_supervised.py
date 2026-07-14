"""
Supervised ablation experiment: do Centering Theory features add
incremental validity for predicting human cohesion ratings?

Three nested ridge regression models, 5-fold cross-validated on ELLIPSE:

  A: surface features only        (length, sentence stats, lexical diversity)
  B: A + connective features      (discourse markers, pronoun density)
  C: B + centering features       (transition distribution, cohesion score,
                                   entity grid, entity density)

The scientific question is the C-vs-B margin: if centering features add
nothing on held-out data, the Centering-based scalar carries no incremental
signal for learner-essay cohesion as humans rate it — and the honest
conclusion is a reposition or a negative-result publication (see
docs/CAEAS_DEVELOPMENT_PLAN.md, GATE 1).

Pure numpy (already a spaCy dependency) — no sklearn.

Usage:
    python -m lgram.ellipse_features --sample 1000   # extract first
    python -m lgram.ellipse_supervised
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .ellipse_benchmark import DEFAULT_CSV, pearson
from .ellipse_features import (
    ALL_FEATURES,
    CENTERING_FEATURES,
    CONNECTIVE_FEATURES,
    SURFACE_FEATURES,
    EllipseFeatureExtractor,
)
from .essay.metrics import quadratic_weighted_kappa

FEATURE_GROUPS: Dict[str, List[str]] = {
    "A_surface": SURFACE_FEATURES,
    "B_surface+connective": SURFACE_FEATURES + CONNECTIVE_FEATURES,
    "C_full+centering": ALL_FEATURES,
    "C_only_centering": CENTERING_FEATURES,
}

RIDGE_LAMBDAS = [0.01, 0.1, 1.0, 10.0, 100.0]


@dataclass
class AblationResult:
    group: str
    features: List[str]
    n: int
    cv_pearson_mean: float
    cv_pearson_std: float
    cv_qwk_mean: float
    best_lambda: float
    weights: Dict[str, float] = field(default_factory=dict)


def _standardize(
    X: np.ndarray, mean: np.ndarray = None, std: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        std[std == 0] = 1.0
    return (X - mean) / std, mean, std


def _ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    n_feat = X.shape[1]
    A = X.T @ X + lam * np.eye(n_feat)
    return np.linalg.solve(A, X.T @ y)


def _kfold_indices(n: int, k: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [idx[i::k] for i in range(k)]


def cross_validate(
    X: np.ndarray, y: np.ndarray, k: int = 5, seed: int = 7
) -> Tuple[float, float, float, float]:
    """Returns (pearson_mean, pearson_std, qwk_mean, best_lambda)."""
    folds = _kfold_indices(len(y), k, seed)
    best = (-2.0, 0.0, 0.0, RIDGE_LAMBDAS[0])

    for lam in RIDGE_LAMBDAS:
        rs: List[float] = []
        qwks: List[float] = []
        for i in range(k):
            test_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(k) if j != i])

            Xtr, mean, std = _standardize(X[train_idx])
            Xte, _, _ = _standardize(X[test_idx], mean, std)
            ytr = y[train_idx]
            y_mean = ytr.mean()

            w = _ridge_fit(Xtr, ytr - y_mean, lam)
            pred = Xte @ w + y_mean

            rs.append(pearson(pred.tolist(), y[test_idx].tolist()))
            qwks.append(quadratic_weighted_kappa(pred.tolist(), y[test_idx].tolist()))

        r_mean = float(np.mean(rs))
        if r_mean > best[0]:
            best = (r_mean, float(np.std(rs)), float(np.mean(qwks)), lam)

    return best


def run_ablation(
    csv_path: Path = DEFAULT_CSV,
    target: str = "Cohesion",
    seed: int = 7,
) -> List[AblationResult]:
    ex = EllipseFeatureExtractor(csv_path=csv_path)
    features = ex._load_cache()
    if len(features) < 100:
        raise RuntimeError(
            f"Only {len(features)} essays in feature cache; run "
            f"'python -m lgram.ellipse_features --sample 1000' first."
        )

    with open(csv_path, encoding="utf-8") as f:
        rows = {r["text_id_kaggle"]: r for r in csv.DictReader(f)}

    tids = [t for t in features if t in rows and rows[t].get(target)]
    y = np.array([float(rows[t][target]) for t in tids])

    results: List[AblationResult] = []
    for group, feat_names in FEATURE_GROUPS.items():
        X = np.array([[features[t][fn] for fn in feat_names] for t in tids])
        r_mean, r_std, qwk_mean, lam = cross_validate(X, y, seed=seed)

        # full-data fit for interpretable weights (standardized scale)
        Xs, _, _ = _standardize(X)
        w = _ridge_fit(Xs, y - y.mean(), lam)
        weights = {fn: round(float(wi), 4) for fn, wi in zip(feat_names, w)}

        results.append(
            AblationResult(
                group=group,
                features=feat_names,
                n=len(tids),
                cv_pearson_mean=round(r_mean, 4),
                cv_pearson_std=round(r_std, 4),
                cv_qwk_mean=round(qwk_mean, 4),
                best_lambda=lam,
                weights=weights,
            )
        )
    return results


def report(results: List[AblationResult], target: str = "Cohesion") -> str:
    lines = [
        "=" * 68,
        f"  ELLIPSE SUPERVISED ABLATION — target: human {target}",
        f"  n={results[0].n}  |  5-fold CV ridge  |  held-out Pearson r",
        "=" * 68,
    ]
    by_group = {r.group: r for r in results}
    for r in results:
        lines.append(
            f"  {r.group:24s} r = {r.cv_pearson_mean:+.3f} "
            f"(±{r.cv_pearson_std:.3f})  QWK = {r.cv_qwk_mean:+.3f}  "
            f"λ={r.best_lambda}"
        )

    a = by_group.get("A_surface")
    b = by_group.get("B_surface+connective")
    c = by_group.get("C_full+centering")
    if a and b and c:
        lines.append("-" * 68)
        lines.append(
            f"  connective increment (B-A): {b.cv_pearson_mean - a.cv_pearson_mean:+.3f}"
        )
        lines.append(
            f"  centering  increment (C-B): {c.cv_pearson_mean - b.cv_pearson_mean:+.3f}"
        )
        lines.append("-" * 68)
        verdict = (
            "centering features ADD incremental signal"
            if c.cv_pearson_mean - b.cv_pearson_mean >= 0.03
            else "centering features add NO meaningful increment"
        )
        lines.append(f"  VERDICT: {verdict}")

    lines.append("=" * 68)
    lines.append("  Top standardized weights (C model):")
    if c:
        top = sorted(c.weights.items(), key=lambda kv: -abs(kv[1]))[:8]
        for fn, wi in top:
            lines.append(f"    {fn:20s} {wi:+.3f}")
    lines.append("=" * 68)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="ELLIPSE supervised ablation")
    parser.add_argument("--target", default="Cohesion")
    parser.add_argument("--csv", default=str(DEFAULT_CSV))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    results = run_ablation(csv_path=Path(args.csv), target=args.target, seed=args.seed)
    print(report(results, target=args.target))
    return 0


if __name__ == "__main__":
    sys.exit(main())
