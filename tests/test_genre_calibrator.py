import unittest

from lgram.genre_calibrator import GenreCalibrator


class TestGenreCalibrator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.calibrator = GenreCalibrator()

    def _corpus(self):
        return {
            "narrative": [
                "John went to the store. He bought milk. John paid at the counter. He walked home. The sun was setting.",
                "Alice went to the park. She met Bob there. They talked for hours. Alice was very happy. They went home together.",
                "Tom opened the door. He stepped inside. The room was dark. Tom turned on the light. He sat down on the sofa.",
            ],
            "expository": [
                "Climate change affects global temperatures. Scientists agree on greenhouse gas emissions. Rising seas threaten coastal cities. Renewable energy offers a solution.",
                "Python is a programming language. It supports object-oriented design. Many developers use Python daily. The community is very active.",
                "Quantum computing uses quantum bits or qubits. These qubits can exist in superposition states. This allows parallel computation. Quantum computers may solve problems that classical computers cannot.",
            ],
        }

    def test_calibrate_returns_profiles(self):
        corpus = self._corpus()
        profiles = self.calibrator.calibrate(corpus)
        self.assertGreater(len(profiles), 0)
        for p in profiles:
            self.assertIn(p.name, corpus)
            self.assertGreater(p.sample_count, 0)

    def test_profile_has_statistics(self):
        corpus = self._corpus()
        profiles = self.calibrator.calibrate(corpus)
        for p in profiles:
            self.assertGreaterEqual(p.mean_rough, 0.0)
            self.assertGreaterEqual(p.mean_continue, 0.0)
            self.assertIn(p.confidence, ("low", "medium", "high"))

    def test_report_generates_string(self):
        corpus = self._corpus()
        profiles = self.calibrator.calibrate(corpus)
        report = self.calibrator.report(profiles)
        self.assertIsInstance(report, str)
        self.assertIn("GENRE", report)

    def test_tukey_fence_valid(self):
        corpus = self._corpus()
        profiles = self.calibrator.calibrate(corpus)
        for p in profiles:
            self.assertGreaterEqual(p.tukey_upper, 0.0)
            self.assertLessEqual(p.tukey_upper, 1.0)
            self.assertGreaterEqual(p.iqr_rough, 0.0)

    def test_confidence_levels(self):
        corpus = {
            "small": ["A short text."],
            "medium": [
                "First text. Second sentence.",
                "Another text. Another sentence.",
            ],
            "large": [
                f"Text number {i}. Continuing with more words." for i in range(25)
            ],
        }
        profiles = self.calibrator.calibrate(corpus)
        conf_map = {p.name: p.confidence for p in profiles}
        for name in ("small",):
            self.assertIn(conf_map.get(name, "none"), ("low", "none"))

    def test_empty_corpus_survives(self):
        profiles = self.calibrator.calibrate({"empty": []})
        self.assertEqual(len(profiles), 0)

    def test_recommendation_string(self):
        corpus = self._corpus()
        profiles = self.calibrator.calibrate(corpus)
        for p in profiles:
            self.assertIsInstance(p.recommendation, str)
            self.assertIn("Rough-Shift", p.recommendation)


if __name__ == "__main__":
    unittest.main()
