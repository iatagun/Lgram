"""
Unit tests for TextAnalyzer analysis layer.
"""

import unittest

from lgram import TextAnalyzer


class TestTextAnalyzer(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ta = TextAnalyzer()

    def test_analyze(self):
        r = self.ta.analyze("John went to the store. He bought milk. John paid. He left.")
        self.assertGreater(r.sentence_count, 0)
        self.assertIn(r.quality, ("high", "medium", "low", "insufficient_data"))

    def test_insufficient_data(self):
        r = self.ta.analyze("")
        self.assertEqual(r.quality, "insufficient_data")
        r2 = self.ta.analyze("Hello.")
        self.assertEqual(r2.quality, "insufficient_data")

    def test_entity_grid_score(self):
        g = self.ta.entity_grid_score("John went. He bought milk. John paid. He left.")
        self.assertGreaterEqual(g.score, 0.0)
        self.assertLessEqual(g.score, 1.0)
        self.assertGreater(len(g.entities), 0)
        self.assertGreater(len(g.matrix), 0)

    def test_texttile_segments(self):
        b = self.ta.texttile_segments("A. B. C. D. E. F.")
        self.assertIn(0, b)

    def test_hybrid_boundaries(self):
        h = self.ta.hybrid_boundaries("John went. He bought. The weather changed. Stocks fell.")
        self.assertIn("centering", h)
        self.assertIn("texttile", h)
        self.assertIn("high_confidence", h)

    def test_build_cohesion_graph(self):
        g = self.ta.build_cohesion_graph("John went. He bought. John paid.")
        self.assertGreaterEqual(g.density, 0.0)
        self.assertLessEqual(g.density, 1.0)
        self.assertGreater(len(g.communities), 0)

    def test_lexical_chain_score(self):
        s = self.ta.lexical_chain_score("The cat sat. The cat slept. The cat purred.")
        self.assertGreater(s, 0.0)
        s2 = self.ta.lexical_chain_score("Cats. Dogs. Birds. Fish.")
        self.assertLess(s2, 1.0)

    def test_cohesion_trend(self):
        t = self.ta.cohesion_trend("John went. He bought. John paid. He left.", window=2)
        self.assertIn(t["trend"], ("improving", "declining", "stable"))
        self.assertGreater(len(t["windows"]), 0)

    def test_cohesion_heatmap(self):
        h = self.ta.cohesion_heatmap("John went to the store. He bought milk. John paid with cash.", ascii_render=True)
        self.assertGreater(len(h["matrix"]), 0)

    def test_readability_score(self):
        r = self.ta.readability_score("This is a simple test. It has two sentences.")
        self.assertIn("flesch_reading_ease", r)
        self.assertGreater(r["words"], 0)

    def test_combined_score(self):
        s = self.ta.combined_score("John went. He bought milk. John paid.")
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_suggest_improvements(self):
        s = self.ta.suggest_improvements("John went. Pizza is good. Aliens exist.")
        self.assertGreater(len(s), 0)

    def test_annotate_weak_points(self):
        a = self.ta.annotate_weak_points("John went. Pizza is good. Aliens exist.")
        self.assertIn("WEAK", a)

    def test_diff_cohesion(self):
        d = self.ta.diff_cohesion("John went. He bought. John paid.", "John went. He bought.")
        self.assertIn(d["verdict"], ("improved", "declined", "similar"))
        self.assertIn("delta", d)

    def test_analyze_batch(self):
        b = self.ta.analyze_batch(
            ["John went. He bought.", "Cats. Dogs. Birds."],
            labels=["Good", "Bad"],
        )
        self.assertEqual(b["count"], 2)
        self.assertIn("rankings", b)


if __name__ == "__main__":
    unittest.main()
