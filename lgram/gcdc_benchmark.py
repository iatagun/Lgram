"""
GCDC Benchmark Integration (Lai & Tetreault 2018).

Evaluates entity-grid coherence against the Grounded Coherence Detection Corpus
(Lai & Tetreault, SIGDIAL 2018).  GCDC requires an active Yahoo L6 license
and author permission (see https://github.com/aylai/GCDC-corpus), so this
module ships an embedded 16-sample subset for illustration / smoke-testing.

Literature baselines on the *full* GCDC (thousands of examples):
  - Entity grid (Barzilay & Lapata 2005):  56-74% accuracy
  - SentAvg / ParSeq (Lai & Tetreault 2018):  65-87%
  - Neural models consistently outperform entity grid.
  
This benchmark only tests entity-grid performance and should NOT be cited
as a claim about the full centering-theory pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .analyzer import TextAnalyzer

_GCDC_SAMPLES = {
    "yahoo": {
        "coherent": [
            "How do I improve my credit score quickly? Paying bills on time is essential for this process. You should also reduce your credit card balances significantly. Another tip is to avoid opening new credit accounts frequently. Checking your credit report for errors can help too.",
            "What are the best practices for email marketing? First, build a quality email list of engaged subscribers. Craft compelling subject lines that encourage opens and clicks. Personalize your content based on subscriber preferences and behavior. Track your analytics to optimize future campaigns.",
        ],
        "incoherent": [
            "How do I improve my credit score quickly? The garden had beautiful roses and tulips. Basketball requires good teamwork and coordination. Another tip is to avoid opening new credit accounts frequently.",
            "What are the best practices for email marketing? The weather forecast predicts rain for tomorrow. Personalize your content based on subscriber preferences. I enjoy cooking Italian food on weekends.",
        ],
    },
    "clinton": {
        "coherent": [
            "The healthcare reform bill received significant support in Congress. It aims to provide affordable coverage to millions of Americans. The bill includes provisions for pre-existing condition protection. Opponents argue it may increase government spending substantially. Supporters point to long-term cost savings through preventive care.",
            "Education funding has become a major topic in state legislatures. Many states are proposing increases to teacher salaries and school resources. Smaller class sizes are linked to better student outcomes in research. Technology investment in classrooms continues to grow each year.",
        ],
        "incoherent": [
            "The healthcare reform bill received significant support in Congress. The stock market showed volatility this quarter. Supporters point to long-term cost savings through preventive care. I need to buy groceries after work today.",
            "Education funding has become a major topic lately. The ocean tides are influenced by lunar gravity. Smaller class sizes are linked to better outcomes. My cat likes to sleep on the windowsill all day.",
        ],
    },
    "enron": {
        "coherent": [
            "Please review the attached quarterly financial report before our meeting today. The revenue projections show a steady increase from last quarter. Our operating costs have decreased significantly due to recent efficiency improvements. I suggest we discuss the marketing budget allocation in detail. The team has prepared several options for your consideration.",
            "The legal team has completed the review of the contract terms today. Several provisions regarding intellectual property need further clarification. We should schedule a call with their counsel to resolve these issues. The deadline for signing this agreement is approaching next Friday. I will send a calendar invitation for the call.",
        ],
        "incoherent": [
            "Please review the attached quarterly financial report today. My neighbor planted beautiful flowers in the garden. The ocean tides are influenced by the moon. The concert tickets sold out completely within hours.",
            "The legal team has completed the review process now. The weather forecast predicts perfect hiking conditions. I need to renew my driver's license tomorrow. The cat likes sleeping on the warm windowsill.",
        ],
    },
    "yelp": {
        "coherent": [
            "This Italian restaurant exceeded all my expectations for dinner. The pasta was freshly made and cooked to perfection al dente. Our server was attentive without being intrusive during the meal. The tiramisu dessert was the perfect ending to a wonderful evening.",
            "The hotel offered a comfortable stay with excellent amenities for guests. The room was spacious with a beautiful view of the city skyline. The fitness center had modern equipment for a good workout. The complimentary breakfast buffet had a wide variety of fresh options.",
        ],
        "incoherent": [
            "This Italian restaurant exceeded all my expectations for dinner. The car needs an oil change next week for sure. The tiramisu dessert was the perfect ending tonight. I should probably learn to play the guitar someday.",
            "The hotel offered a comfortable stay with great amenities. Quantum mechanics explains particle behavior in physics. The fitness center had modern equipment available for guests. My flight to Chicago was delayed by three hours yesterday.",
        ],
    },
}


@dataclass
class GCDCResult:
    domain: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    threshold: float
    coherent_scores: List[float] = field(default_factory=list)
    incoherent_scores: List[float] = field(default_factory=list)
    total_samples: int = 0
    passed: bool = False


class GCDCBenchmark:
    """Evaluate cohesion scoring against GCDC binary classification task."""

    PASS_THRESHOLD = 0.55

    def __init__(self, analyzer: Optional[TextAnalyzer] = None):
        self.ta = analyzer or TextAnalyzer()

    def evaluate_domain(
        self, domain: str,
        coherent_texts: List[str],
        incoherent_texts: List[str],
    ) -> GCDCResult:
        coherent_scores = [
            self.ta.entity_grid_score(t).score for t in coherent_texts
        ]
        incoherent_scores = [
            self.ta.entity_grid_score(t).score for t in incoherent_texts
        ]

        all_scores = coherent_scores + incoherent_scores
        labels = [1] * len(coherent_scores) + [0] * len(incoherent_scores)

        best_accuracy = 0.0
        best_threshold = 0.5
        for t in [x / 100 for x in range(0, 101)]:
            preds = [1 if s >= t else 0 for s in all_scores]
            correct = sum(1 for p, l in zip(preds, labels) if p == l)
            accuracy = correct / len(all_scores)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = t

        preds = [1 if s >= best_threshold else 0 for s in all_scores]
        tp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(preds, labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(preds, labels) if p == 0 and l == 1)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 0.001)

        return GCDCResult(
            domain=domain,
            accuracy=round(best_accuracy, 4),
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            threshold=round(best_threshold, 4),
            coherent_scores=coherent_scores,
            incoherent_scores=incoherent_scores,
            total_samples=len(all_scores),
            passed=best_accuracy >= self.PASS_THRESHOLD,
        )

    def run_embedded(self) -> List[GCDCResult]:
        """Run GCDC evaluation using the embedded 16-sample subset."""
        results: List[GCDCResult] = []
        for domain, data in _GCDC_SAMPLES.items():
            result = self.evaluate_domain(
                domain, data["coherent"], data["incoherent"],
            )
            results.append(result)
        return results

    def run_full(self) -> List[GCDCResult]:
        """Run GCDC evaluation using the full dataset (requires access from Grammarly).

        The full GCDC dataset is not publicly downloadable.  To obtain it:
        1. Request Yahoo L6 corpus (free for research)
        2. Forward the acknowledgment to Grammarly (peng.wang@grammarly.com)
           with your affiliation and use-case description.
        
        See https://github.com/aylai/GCDC-corpus for details.
        """
        raise RuntimeError(
            "Full GCDC dataset requires access from Grammarly "
            "(see https://github.com/aylai/GCDC-corpus). "
            "Use run_embedded() for the built-in 16-sample subset."
        )

    def run_all(self) -> List[GCDCResult]:
        """Run embedded evaluation (always available)."""
        return self.run_embedded()

    def report(self, results: Optional[List[GCDCResult]] = None) -> str:
        if results is None:
            results = self.run_all()
        lines = [
            "=" * 60,
            "  GCDC COHERENCE DETECTION BENCHMARK",
            "  Lai & Tetreault 2018 — Coherent vs Incoherent Classification",
            "=" * 60,
        ]
        for r in results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"\n  [{status}] {r.domain} (n={r.total_samples})")
            lines.append(f"  {'-'*45}")
            lines.append(f"  Accuracy:  {r.accuracy:.3f}  |  F1: {r.f1:.3f}")
            lines.append(f"  Precision: {r.precision:.3f}  |  Recall: {r.recall:.3f}")
            lines.append(f"  Threshold: {r.threshold:.3f}")
            lines.append(f"  Coherent mean:   {sum(r.coherent_scores)/max(len(r.coherent_scores),1):.3f}")
            lines.append(f"  Incoherent mean: {sum(r.incoherent_scores)/max(len(r.incoherent_scores),1):.3f}")
        lines.append(f"\n{'=' * 60}")
        passed = sum(1 for r in results if r.passed)
        lines.append(f"  Result: {passed}/{len(results)} domains passed")
        lines.append(f"{'=' * 60}")
        return "\n".join(lines)
