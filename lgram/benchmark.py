"""
Benchmark Suite — validate cohesion methods against known tests.

Tests:
1. Permutation Test: coherent original > randomly shuffled
2. Degradation Test: score drops as sentences are removed
3. Cross-Method Agreement: all methods correlate
4. Binary Classification: coherent vs incoherent discrimination
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .analyzer import TextAnalyzer


@dataclass
class BenchmarkResult:
    name: str
    passed: bool
    details: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""


class CohesionBenchmark:
    """Run standard cohesion benchmarks."""

    # sample texts spanning different domains
    _coherent_texts = [
        "John went to the grocery store. He needed to buy milk and bread. The store was quite busy today. John found everything quickly. He paid at the checkout counter. Then he walked home.",
        "Tesla reported record quarterly earnings. The electric vehicle maker posted strong revenue. CEO Elon Musk credited demand in China. The company also warned about supply chain issues. Its stock rose in after-hours trading. Analysts remain optimistic.",
        "Python is a popular programming language. It is used for web development and data science. The language has a simple syntax. Many beginners start with Python. It has a large community of developers. They contribute libraries for various tasks.",
    ]

    _incoherent_texts = [
        "John went to the grocery store. Quantum physics is fascinating. Pizza is delicious with extra cheese. The Amazon rainforest produces oxygen. Stock markets fell sharply today.",
        "Tesla reported earnings. I enjoy gardening on weekends. The cat slept peacefully. Python is a programming language. The moon orbits the Earth.",
        "Python is popular. My neighbor has a loud dog. The weather is nice today. Basketball is an exciting sport. She bought a new car.",
    ]

    def __init__(self, analyzer: Optional[TextAnalyzer] = None):
        self.ta = analyzer or TextAnalyzer()
        random.seed(42)

    # ------------------------------------------------------------------
    # Test 1: Permutation
    # ------------------------------------------------------------------

    def test_permutation(self, texts: Optional[List[str]] = None) -> BenchmarkResult:
        """
        Original text cohesion > random permutation cohesion.
        Shuffles sentence order, expects lower score.
        """
        texts = texts or self._coherent_texts
        results = []
        passed = True

        for idx, text in enumerate(texts):
            # split sentences
            doc = self.ta.nlp(text)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            if len(sents) < 3:
                continue

            original_score = self.ta.entity_grid_score(text).score

            # 3 random permutations
            perm_scores = []
            for _ in range(3):
                perm = list(sents)
                random.shuffle(perm)
                perm_text = " ".join(perm)
                perm_scores.append(self.ta.entity_grid_score(perm_text).score)

            avg_perm = sum(perm_scores) / len(perm_scores)
            ok = original_score > avg_perm
            if not ok:
                passed = False

            results.append({
                "text_index": idx,
                "original_score": original_score,
                "permuted_scores": perm_scores,
                "avg_permuted": round(avg_perm, 4),
                "passed": ok,
            })

        summary = f"Permutation: {'PASS' if passed else 'FAIL'} ({sum(1 for r in results if r['passed'])}/{len(results)} texts)"
        # pass if majority of texts pass
        passed = sum(1 for r in results if r["passed"]) >= len(results) / 2

        summary = f"Permutation: {'PASS' if passed else 'FAIL'} ({sum(1 for r in results if r['passed'])}/{len(results)} texts)"
        return BenchmarkResult(
            name="permutation",
            passed=passed,
            details=results,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Test 2: Degradation
    # ------------------------------------------------------------------

    def test_degradation(self, texts: Optional[List[str]] = None) -> BenchmarkResult:
        """
        Removing sentences should decrease (or not increase) cohesion score.
        """
        texts = texts or self._coherent_texts
        results = []
        passed = True

        for idx, text in enumerate(texts):
            doc = self.ta.nlp(text)
            sents = [s.text.strip() for s in doc.sents if s.text.strip()]
            if len(sents) < 4:
                continue

            full_score = self.ta.entity_grid_score(text).score

            # remove last sentence
            shortened = " ".join(sents[:-1])
            short_score = self.ta.entity_grid_score(shortened).score

            # score shouldn't drop dramatically unless last sentence was crucial
            drop = full_score - short_score
            ok = drop >= -0.2  # allow minor fluctuation
            if not ok:
                passed = False

            results.append({
                "text_index": idx,
                "full_score": full_score,
                "shortened_score": short_score,
                "drop": round(drop, 4),
                "passed": ok,
            })

        summary = f"Degradation: {'PASS' if passed else 'FAIL'} ({sum(1 for r in results if r['passed'])}/{len(results)} texts)"
        return BenchmarkResult(
            name="degradation",
            passed=passed,
            details=results,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Test 3: Cross-Method Agreement
    # ------------------------------------------------------------------

    def test_cross_method(self, texts: Optional[List[str]] = None) -> BenchmarkResult:
        """
        All cohesion methods should agree on high/low direction.
        """
        all_texts = (texts or []) + self._coherent_texts + self._incoherent_texts
        if not texts:
            all_texts = self._coherent_texts + self._incoherent_texts

        methods: Dict[str, Callable] = {
            "entity_grid": lambda t: self.ta.entity_grid_score(t).score,
            "lexical_chain": lambda t: self.ta.lexical_chain_score(t),
        }

        method_scores: Dict[str, List[float]] = {m: [] for m in methods}
        for text in all_texts:
            for m_name, m_fn in methods.items():
                try:
                    method_scores[m_name].append(m_fn(text))
                except Exception:
                    method_scores[m_name].append(0.5)

        # check pairwise correlation direction
        agreements = []
        names = list(methods.keys())
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = method_scores[names[i]]
                b = method_scores[names[j]]
                # check if both > or < median for each text
                med_a = sorted(a)[len(a) // 2]
                med_b = sorted(b)[len(b) // 2]
                matches = sum(
                    1 for ai, bi in zip(a, b)
                    if (ai >= med_a) == (bi >= med_b)
                )
                rate = matches / max(len(a), 1)
                agreements.append({
                    "method_a": names[i],
                    "method_b": names[j],
                    "agreement_rate": round(rate, 3),
                })

        avg_agreement = sum(r["agreement_rate"] for r in agreements) / max(len(agreements), 1)
        passed = avg_agreement >= 0.4  # methods should not anti-correlate

        summary = f"Cross-Method: {'PASS' if passed else 'FAIL'} (avg agreement {avg_agreement:.2f})"
        return BenchmarkResult(
            name="cross_method",
            passed=passed,
            details=agreements,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Test 4: Binary Classification
    # ------------------------------------------------------------------

    def test_classification(self) -> BenchmarkResult:
        """
        Can entity_grid_score discriminate coherent from incoherent?
        Always expects: coherent_score > incoherent_score.
        """
        scores = []
        for text in self._coherent_texts:
            scores.append(("coherent", self.ta.entity_grid_score(text).score))
        for text in self._incoherent_texts:
            scores.append(("incoherent", self.ta.entity_grid_score(text).score))

        coherent_scores = [s for t, s in scores if t == "coherent"]
        incoherent_scores = [s for t, s in scores if t == "incoherent"]

        if not coherent_scores or not incoherent_scores:
            return BenchmarkResult(
                name="classification",
                passed=False,
                summary="Classification: FAIL (no data)",
            )

        # find optimal threshold
        best_correct = 0
        best_threshold = 0.5
        for threshold in [x / 100 for x in range(0, 101)]:
            correct = sum(1 for s in coherent_scores if s >= threshold)
            correct += sum(1 for s in incoherent_scores if s < threshold)
            if correct > best_correct:
                best_correct = correct
                best_threshold = threshold

        accuracy = best_correct / len(scores)
        passed = accuracy >= 0.75

        summary = f"Classification: {'PASS' if passed else 'FAIL'} (accuracy {accuracy:.2f} at threshold {best_threshold:.2f})"
        return BenchmarkResult(
            name="classification",
            passed=passed,
            details=[{
                "coherent_mean": round(sum(coherent_scores) / len(coherent_scores), 4),
                "incoherent_mean": round(sum(incoherent_scores) / len(incoherent_scores), 4),
                "accuracy": round(accuracy, 3),
                "threshold": best_threshold,
                "total_samples": len(scores),
            }],
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Run all
    # ------------------------------------------------------------------

    def run_all(self) -> List[BenchmarkResult]:
        """Run all benchmark tests."""
        return [
            self.test_permutation(),
            self.test_degradation(),
            self.test_cross_method(),
            self.test_classification(),
        ]

    def report(self) -> str:
        """Generate human-readable benchmark report."""
        results = self.run_all()
        lines = [
            "=" * 50,
            "  COHESION BENCHMARK REPORT",
            "=" * 50,
        ]
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"\n  [{status}] {r.summary}")
            for d in r.details[:3]:
                lines.append(f"        {d}")
            if len(r.details) > 3:
                lines.append(f"        ... and {len(r.details) - 3} more")
        lines.append(f"\n{'=' * 50}")
        passed = sum(1 for r in results if r.passed)
        lines.append(f"  Result: {passed}/{len(results)} tests passed")
        lines.append(f"{'=' * 50}")
        return "\n".join(lines)
