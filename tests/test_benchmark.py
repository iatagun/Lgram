import unittest

from lgram import TextAnalyzer
from lgram.benchmark import CohesionBenchmark


class TestCohesionBenchmark(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ta = TextAnalyzer()
        cls.bench = CohesionBenchmark(cls.ta)

    def test_permutation_passes(self):
        result = self.bench.test_permutation()
        self.assertIsNotNone(result.name)
        self.assertIsNotNone(result.summary)
        self.assertIsInstance(result.passed, bool)
        self.assertGreater(len(result.details), 0)

    def test_degradation_passes(self):
        result = self.bench.test_degradation()
        self.assertIsNotNone(result.name)
        self.assertIsInstance(result.passed, bool)
        self.assertGreater(len(result.details), 0)

    def test_cross_method_agreement_positive(self):
        result = self.bench.test_cross_method()
        self.assertIn("agreement", result.summary.lower())
        self.assertIsInstance(result.passed, bool)

    def test_classification_discriminates(self):
        result = self.bench.test_classification()
        self.assertIsNotNone(result.summary)
        self.assertIsInstance(result.passed, bool)
        self.assertGreater(len(result.details), 0)
        self.assertGreater(result.details[0]["total_samples"], 0)

    def test_run_all_returns_four_results(self):
        results = self.bench.run_all()
        self.assertEqual(len(results), 4)
        for r in results:
            self.assertIsInstance(r.passed, bool)
            self.assertIsNotNone(r.name)

    def test_report_generates_string(self):
        report = self.bench.report()
        self.assertIsInstance(report, str)
        self.assertIn("BENCHMARK", report)
        self.assertIn("PASS", report)

    def test_coherent_vs_incoherent_discrimination(self):
        result = self.bench.test_classification()
        details = result.details[0]
        self.assertGreater(
            details["coherent_mean"],
            details["incoherent_mean"],
        )

    def test_performance_with_custom_texts(self):
        custom = [
            "Alice went to the park. She sat on a bench. Alice read a book. She enjoyed it very much. The weather was beautiful. Alice stayed until sunset.",
        ]
        result = self.bench.test_permutation(custom)
        self.assertGreater(len(result.details), 0)
        all_ok = all(d["passed"] for d in result.details)
        self.assertTrue(all_ok)


if __name__ == "__main__":
    unittest.main()
