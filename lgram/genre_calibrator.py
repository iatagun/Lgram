"""
Genre Threshold Calibrator — empirically derive transition patterns
for different text types using real corpora.

Output: per-genre Rough-Shift ranges, Continue/Rough distributions,
and recommended thresholds for genre-aware analysis.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .analyzer import TextAnalyzer


@dataclass
class GenreProfile:
    name: str
    sample_count: int
    rough_shifts: List[float]
    continue_rates: List[float]
    retain_rates: List[float]
    smooth_rates: List[float]
    mean_rough: float = 0.0
    std_rough: float = 0.0
    p25_rough: float = 0.0
    p75_rough: float = 0.0
    iqr_rough: float = 0.0
    tukey_upper: float = 0.0
    mean_continue: float = 0.0
    mean_retain: float = 0.0
    mean_smooth: float = 0.0
    confidence: str = "low"
    recommendation: str = ""


class GenreCalibrator:
    """
    Calibrate genre thresholds from real text samples.

    IMPORTANT: Calibration corpus must be DIFFERENT from evaluation corpus.
    Using the same data for both = circular validation. Split corpus into
    train (calibration) and test (evaluation) sets.

    Threshold method: Tukey's fence (p75 + 1.5 * IQR).
    Minimum recommended: 20 samples per genre for high confidence.
    """

    def __init__(self, model: str = "en_core_web_md"):
        self.ta = TextAnalyzer(model, similarity_threshold=0.35)

    def calibrate(
        self, corpus: Dict[str, List[str]]
    ) -> List[GenreProfile]:
        """
        corpus: {genre_name: [text1, text2, ...]}

        Returns calibrated GenreProfile per genre with:
        - Mean, std, p25, p75 for Rough-Shift
        - Recommended normal ranges
        """
        profiles: List[GenreProfile] = []

        for genre_name, texts in corpus.items():
            roughs: List[float] = []
            continues: List[float] = []
            retains: List[float] = []
            smooths: List[float] = []

            for text in texts:
                if not text.strip():
                    continue
                r = self.ta.analyze(text)
                d = r.transition_distribution
                roughs.append(d.get("Rough-Shift", 0))
                continues.append(d.get("Continue", 0))
                retains.append(d.get("Retain", 0))
                smooths.append(d.get("Smooth-Shift", 0))

            if len(roughs) < 3:
                continue

            profile = GenreProfile(
                name=genre_name,
                sample_count=len(roughs),
                rough_shifts=roughs,
                continue_rates=continues,
                retain_rates=retains,
                smooth_rates=smooths,
            )

            # Statistics
            profile.mean_rough = round(self._mean(roughs), 3)
            profile.std_rough = round(self._std(roughs, profile.mean_rough), 3)
            profile.p25_rough = round(self._percentile(sorted(roughs), 25), 3)
            profile.p75_rough = round(self._percentile(sorted(roughs), 75), 3)
            profile.iqr_rough = round(profile.p75_rough - profile.p25_rough, 3)
            # Tukey's fence: p75 + 1.5 * IQR (standard outlier detection)
            profile.tukey_upper = round(min(profile.p75_rough + 1.5 * profile.iqr_rough, 1.0), 3)
            profile.mean_continue = round(self._mean(continues), 3)
            profile.mean_retain = round(self._mean(retains), 3)
            profile.mean_smooth = round(self._mean(smooths), 3)

            # Confidence level
            if len(roughs) >= 20:
                profile.confidence = "high"
            elif len(roughs) >= 10:
                profile.confidence = "medium"
            else:
                profile.confidence = "low"

            # Recommendation
            profile.recommendation = (
                f"Rough-Shift {profile.p25_rough:.1%}-{profile.p75_rough:.1%} is normal "
                f"(IQR={profile.iqr_rough:.1%}). "
                f"Flag if Rough > {profile.tukey_upper:.1%} (Tukey: p75 + 1.5*IQR)."
            )
            if profile.confidence != "high":
                profile.recommendation += (
                    f" [CONFIDENCE: {profile.confidence.upper()} — need >=20 samples. "
                    f"Current n={len(roughs)}.]"
                )

            profiles.append(profile)

        return profiles

    def report(self, profiles: List[GenreProfile]) -> str:
        """Generate human-readable calibration report."""
        lines = [
            "=" * 70,
            "  GENRE CALIBRATION REPORT",
            "=" * 70,
            "  Method: Tukey's fence (p75 + 1.5*IQR)",
            "  Minimum: 20 samples/genre for high confidence",
            "  WARNING: Use separate corpus for evaluation (not calibration data)",
        ]

        for p in profiles:
            lines.append(f"\n  {p.name.upper()} (n={p.sample_count})")
            lines.append(f"  {'-'*50}")

            # Rough-Shift distribution
            rough_bar = self._histogram(p.rough_shifts, width=40)
            lines.append(f"  Rough-Shift:  mean={p.mean_rough:.3f}  std={p.std_rough:.3f}  "
                        f"p25={p.p25_rough:.3f}  p75={p.p75_rough:.3f}")
            lines.append(f"                {rough_bar}")

            # Other metrics
            lines.append(f"  Continue:     mean={p.mean_continue:.3f}")
            lines.append(f"  Retain:       mean={p.mean_retain:.3f}")
            lines.append(f"  Smooth-Shift: mean={p.mean_smooth:.3f}")

            # Recommendation
            lines.append(f"\n  Recommendation: {p.recommendation}")

        # Cross-genre comparison
        if len(profiles) >= 2:
            lines.append(f"\n{'='*70}")
            lines.append(f"  CROSS-GENRE COMPARISON")
            lines.append(f"{'='*70}")
            lines.append(f"\n  {'Genre':15s} {'n':>4s} {'Rough mean':>8s} {'Rough std':>8s} "
                        f"{'p25':>6s} {'p75':>6s} {'Continue mean':>10s}")
            lines.append(f"  {'-'*60}")
            for p in sorted(profiles, key=lambda x: x.mean_rough):
                lines.append(f"  {p.name:15s} {p.sample_count:>4d} "
                            f"{p.mean_rough:>8.1%} {p.std_rough:>8.1%} "
                            f"{p.p25_rough:>6.1%} {p.p75_rough:>6.1%} "
                            f"{p.mean_continue:>10.1%}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Statistics helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mean(values: List[float]) -> float:
        return sum(values) / max(len(values), 1)

    @staticmethod
    def _std(values: List[float], mean: float) -> float:
        if len(values) < 2:
            return 0.0
        return math.sqrt(sum((x - mean) ** 2 for x in values) / (len(values) - 1))

    @staticmethod
    def _percentile(sorted_values: List[float], p: float) -> float:
        if not sorted_values:
            return 0.0
        k = (len(sorted_values) - 1) * p / 100.0
        f = int(k)
        c = k - f
        if f + 1 < len(sorted_values):
            return sorted_values[f] + c * (sorted_values[f + 1] - sorted_values[f])
        return sorted_values[f]

    @staticmethod
    def _histogram(values: List[float], width: int = 40) -> str:
        if not values:
            return ""
        bins = [0] * 10
        for v in values:
            idx = min(int(v * 10), 9)
            bins[idx] += 1
        max_bin = max(bins) or 1
        chars = "_.-:=+*#"
        bar = "".join(
            chars[min(int(b / max_bin * 7), 7)] for b in bins
        )
        return bar
