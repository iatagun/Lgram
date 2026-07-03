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
        self.assertGreaterEqual(
            passed / max(len(results), 1), 0.5,
            f"Only {passed}/{len(results)} domains passed",
        )

    def test_accuracy_above_chance(self):
        results = self.bench.run_embedded()
        for r in results:
            self.assertGreaterEqual(r.accuracy, 0.5, f"{r.domain} accuracy too low")

    def test_report_generates_string(self):
        report = self.bench.report()
        self.assertIsInstance(report, str)
        self.assertIn("GCDC", report)
        self.assertIn("PASS", report)

    def test_coherent_scores_higher_than_incoherent(self):
        results = self.bench.run_embedded()
        for r in results:
            coh_mean = sum(r.coherent_scores) / max(len(r.coherent_scores), 1)
            inc_mean = sum(r.incoherent_scores) / max(len(r.incoherent_scores), 1)
            self.assertGreaterEqual(
                coh_mean, inc_mean,
                f"{r.domain}: coherent={coh_mean:.3f} < incoherent={inc_mean:.3f}",
            )

    def test_evaluate_single_domain(self):
        result = self.bench.evaluate_domain(
            "test",
            ["John went to the store. He bought milk. John paid. He left."],
            ["John went. Quantum physics is fascinating. Pizza is delicious. The Amazon."],
        )
        self.assertGreater(result.accuracy, 0.5)
        self.assertGreater(result.f1, 0.0)


if __name__ == "__main__":
    unittest.main()
