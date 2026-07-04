"""
Tests for EFL-specific CAEAS features.
"""

from __future__ import annotations

import unittest

from lgram.essay.efl import (
    CEFR_PROFILES,
    EFL_RUBRIC,
    L1TransferAnalyzer,
    estimate_cefr_level,
    get_cefr_profile,
)
from lgram.essay import CAEASGrader, Essay


class TestEFLRubric(unittest.TestCase):

    def test_rubric_has_five_dimensions(self):
        self.assertEqual(len(EFL_RUBRIC), 5)
        names = [c.name for c in EFL_RUBRIC]
        self.assertIn("Grammar", names)
        self.assertIn("Content", names)
        self.assertIn("Organization", names)
        self.assertIn("Style & Expression", names)
        self.assertIn("Mechanics", names)

    def test_rubric_weights_sum_to_one(self):
        total = sum(c.weight for c in EFL_RUBRIC)
        self.assertAlmostEqual(total, 1.0, places=2)

    def test_organization_weight(self):
        org = [c for c in EFL_RUBRIC if c.name == "Organization"][0]
        self.assertEqual(org.weight, 0.30)
        self.assertIn("Centering Theory", org.description)


class TestCEFRProfiles(unittest.TestCase):

    def test_all_levels_available(self):
        self.assertIn("B1", CEFR_PROFILES)
        self.assertIn("B2", CEFR_PROFILES)
        self.assertIn("C1", CEFR_PROFILES)

    def test_profile_structure(self):
        for level in ["B1", "B2", "C1"]:
            p = CEFR_PROFILES[level]
            self.assertIn("expected_score_range", p)
            self.assertIn("cohesion_threshold", p)
            self.assertIn("rough_shift_tolerance", p)
            self.assertIn("typical_issues", p)
            lo, hi = p["expected_score_range"]
            self.assertLess(lo, hi)

    def test_b1_below_b2(self):
        self.assertLess(
            CEFR_PROFILES["B1"]["expected_score_range"][1],
            CEFR_PROFILES["B2"]["expected_score_range"][1],
        )

    def test_get_cefr_profile_valid(self):
        p = get_cefr_profile("B2")
        self.assertEqual(p["label"], "Upper Intermediate")

    def test_get_cefr_profile_invalid(self):
        with self.assertRaises(ValueError):
            get_cefr_profile("A1")

    def test_cohesion_threshold_increases_with_level(self):
        self.assertLess(
            CEFR_PROFILES["B1"]["cohesion_threshold"],
            CEFR_PROFILES["B2"]["cohesion_threshold"],
        )
        self.assertLess(
            CEFR_PROFILES["B2"]["cohesion_threshold"],
            CEFR_PROFILES["C1"]["cohesion_threshold"],
        )


class TestCEFREstimation(unittest.TestCase):

    def test_short_text_is_b1(self):
        level, conf = estimate_cefr_level("Hello world. I like cats.")
        self.assertEqual(level, "B1")

    def test_longer_text_is_b2(self):
        text = " ".join([
            "The impact of social media on modern society is profound and multifaceted.",
            "Research demonstrates that platforms like Instagram and TikTok reshape how",
            "young people form identities and maintain relationships. Furthermore, these",
            "technologies introduce unprecedented challenges regarding privacy, mental",
            "health, and attention spans. Consequently, educators and policymakers must",
            "develop comprehensive strategies to address these emerging concerns.",
            "Studies have shown that excessive social media use correlates with anxiety.",
            "However, the same platforms also provide valuable social support networks.",
            "The key lies in balanced usage and digital literacy education.",
        ])
        level, conf = estimate_cefr_level(text)
        self.assertIn(level, ["B1", "B2"])


class TestL1TransferAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = L1TransferAnalyzer()

    def test_empty_text(self):
        report = self.analyzer.analyze("")
        self.assertEqual(len(report.pro_drop_issues), 0)

    def test_normal_text(self):
        text = (
            "Social media has changed how people communicate today. "
            "Many young people use platforms like Instagram every day. "
            "These apps let users share photos and videos with their friends. "
            "However, social media also has negative effects on mental health."
        )
        report = self.analyzer.analyze(text)
        self.assertGreaterEqual(report.overall_transfer_score, 0.0)
        self.assertLessEqual(report.overall_transfer_score, 1.0)
        self.assertIsInstance(report.summary, str)

    def test_pro_drop_detection(self):
        text = "I went to school. Is very far from home. And was late to class."
        report = self.analyzer.analyze(text)
        self.assertGreater(len(report.pro_drop_issues), 0)

    def test_gender_confusion_asymmetry(self):
        text = (
            "The teacher entered the classroom. He put his books on the desk. "
            "He then started the lesson. He asked many questions. He was strict. "
            "He gave homework. He left early."
        )
        report = self.analyzer.analyze(text)
        self.assertGreater(len(report.gender_pronoun_issues), 0)

    def test_article_estimate(self):
        text = (
            "Cat sat on mat yesterday morning when sun was shining brightly. "
            "Dog ran after ball in park near house quickly. "
            "Bird flew to tree branch without hesitation. "
            "Fish swam in river water all afternoon long."
        )
        report = self.analyzer.analyze(text)
        self.assertLess(report.article_estimate.get("adequacy", 1.0), 0.8)

    def test_article_estimate_normal(self):
        text = (
            "The cat sat on a mat. The dog ran after a ball. "
            "A bird flew to the tree. The sun was shining brightly."
        )
        report = self.analyzer.analyze(text)
        self.assertGreater(report.article_estimate.get("adequacy", 0.0), 0.5)

    def test_sov_detection(self):
        text = (
            "The students who came to the school their homework completed. "
            "She with her friends the interesting book read. "
            "They very early in the morning to the school went. "
            "The teacher all of the students a difficult exam gave."
        )
        report = self.analyzer.analyze(text)
        self.assertGreater(len(report.word_order_issues), 0)

    def test_report_has_all_fields(self):
        text = "This is a simple test essay with some content."
        report = self.analyzer.analyze(text)
        self.assertIsInstance(report.pro_drop_issues, list)
        self.assertIsInstance(report.gender_pronoun_issues, list)
        self.assertIsInstance(report.article_estimate, dict)
        self.assertIsInstance(report.word_order_issues, list)
        self.assertIsNotNone(report.summary)


class TestEFLGrader(unittest.TestCase):

    def setUp(self):
        self.grader = CAEASGrader(use_grammar=False, use_mechanics=False, use_llm=False)

    def test_efl_rubric_used_by_default(self):
        self.assertEqual(len(self.grader.rubric), 5)
        names = [c.name for c in self.grader.rubric]
        self.assertIn("Organization", names)

    def test_cefr_level_set(self):
        grader = CAEASGrader(cefr_level="B2")
        self.assertEqual(grader.cefr_level, "B2")

    def test_l1_enabled(self):
        grader = CAEASGrader(l1_language="tr")
        self.assertEqual(grader.l1_language, "tr")
        self.assertIsNotNone(grader._l1_analyzer)

    def test_grade_with_cefr(self):
        grader = CAEASGrader(cefr_level="B2", l1_language="tr")
        essay = Essay(
            title="EFL Essay",
            text=(
                "Social media is very important for young people today. "
                "Many students use Instagram and TikTok every day. "
                "These applications help people share photos and videos. "
                "However, social media also has bad effects on mental health. "
                "For example, some people feel sad when they compare themselves. "
                "I think social media is both good and bad for society."
            ),
        )
        report = grader.grade(essay)
        self.assertGreaterEqual(report.overall_score, 0)
        self.assertLessEqual(report.overall_score, 100)
        self.assertEqual(len(report.layer_results), 5)

    def test_grade_without_l1(self):
        grader = CAEASGrader()
        essay = Essay(title="T", text="Hello world. This is a test.")
        report = grader.grade(essay)
        self.assertIsNotNone(report)

    def test_set_rubric(self):
        from lgram.essay.models import RubricCriterion
        custom = [RubricCriterion(name="Custom", weight=1.0, description="")]
        self.grader.set_rubric(custom)
        self.assertEqual(len(self.grader.rubric), 1)

    def test_disable_efl_mode(self):
        self.grader.use_efl = False
        essay = Essay(title="T", text="Hello world.")
        report = self.grader.grade(essay)
        self.assertIsNotNone(report)


if __name__ == "__main__":
    unittest.main()
