import unittest

from lgram.ellipse_benchmark import EllipseResult, pearson, spearman


class TestCorrelations(unittest.TestCase):

    def test_pearson_perfect(self):
        x = [1.0, 2.0, 3.0, 4.0]
        self.assertAlmostEqual(pearson(x, x), 1.0)

    def test_pearson_inverse(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [4.0, 3.0, 2.0, 1.0]
        self.assertAlmostEqual(pearson(x, y), -1.0)

    def test_pearson_no_variance(self):
        self.assertEqual(pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]), 0.0)

    def test_pearson_too_short(self):
        self.assertEqual(pearson([1.0, 2.0], [1.0, 2.0]), 0.0)

    def test_spearman_monotonic_nonlinear(self):
        # monotonic but nonlinear: spearman should be 1.0
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [1.0, 4.0, 9.0, 16.0, 25.0]
        self.assertAlmostEqual(spearman(x, y), 1.0)

    def test_spearman_ties(self):
        x = [1.0, 2.0, 2.0, 3.0]
        y = [1.0, 2.0, 2.0, 3.0]
        self.assertAlmostEqual(spearman(x, y), 1.0)


class TestGate1Logic(unittest.TestCase):

    def _result(self, r, baseline):
        gate = r >= 0.5 and (r - abs(baseline)) >= 0.1
        return EllipseResult(
            n=100,
            pearson_r=r,
            spearman_rho=r,
            qwk=r,
            baseline_wordcount_r=baseline,
            baseline_sentcount_r=baseline,
            gate1_pass=gate,
            model="test",
        )

    def test_gate_passes_when_strong_and_beats_baseline(self):
        self.assertTrue(self._result(0.6, 0.3).gate1_pass)

    def test_gate_fails_when_weak(self):
        self.assertFalse(self._result(0.4, 0.1).gate1_pass)

    def test_gate_fails_when_baseline_wins(self):
        # r 0.55 but word count alone gets 0.5 — no real signal margin
        self.assertFalse(self._result(0.55, 0.5).gate1_pass)


if __name__ == "__main__":
    unittest.main()
