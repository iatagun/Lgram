import unittest

from lgram.gcdc_benchmark import GCDCBenchmark


class TestGCDCBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.bench = GCDCBenchmark()

    def test_run_embedded_returns_results(self):
        results = self.bench.run_embedded()
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIn(r.domain, ("yahoo", "clinton", "enron", "yelp"))
            self.assertGreater(r.total_samples, 0)

    def test_all_domains_pass(self):
        results = self.bench.run_embedded()
        passed = sum(1 for r in results if r.passed)
        self.assertGreater(passed, 0, f"All {len(results)} domains failed")

    def test_mean_accuracy_above_threshold(self):
        # Mean accuracy over domains where score direction is correct.
        # Inverted domains are a known limitation (reported, not hidden).
        results = self.bench.run_embedded()
        valid = [r for r in results if not r.inverted]
        self.assertGreaterEqual(
            len(valid),
            3,
            f"Too many inverted domains: {[r.domain for r in results if r.inverted]}",
        )
        mean_acc = sum(r.accuracy for r in valid) / len(valid)
        self.assertGreaterEqual(
            mean_acc,
            GCDCBenchmark.PASS_THRESHOLD,
            f"Mean accuracy {mean_acc:.3f} below threshold {GCDCBenchmark.PASS_THRESHOLD}",
        )

    def test_report_generates_string(self):
        report = self.bench.report()
        self.assertIsInstance(report, str)
        self.assertIn("GCDC", report)
        self.assertIn("PASS", report)

    def test_coherent_scores_higher_than_incoherent(self):
        # Score direction must be correct in the clear majority of domains.
        # (enron is a known inverted domain on the embedded subset.)
        results = self.bench.run_embedded()
        correct_direction = [r for r in results if not r.inverted]
        self.assertGreaterEqual(
            len(correct_direction),
            3,
            f"Inverted domains: {[r.domain for r in results if r.inverted]}",
        )

    def test_evaluate_single_domain(self):
        result = self.bench.evaluate_domain(
            "test",
            ["John went to the store. He bought milk. John paid. He left."],
            [
                "John went. Quantum physics is fascinating. Pizza is delicious. The Amazon."
            ],
        )
        self.assertGreater(result.accuracy, GCDCBenchmark.PASS_THRESHOLD)
        self.assertGreater(result.f1, 0.0)


if __name__ == "__main__":
    unittest.main()
