"""
Tests for CAEAS essay grading system.
"""

from __future__ import annotations

import unittest

from lgram.essay import (
    CAEASGrader,
    CalibrationReport,
    CohesionLayer,
    ConfidenceLayer,
    Essay,
    LayerResult,
    MockContentJudge,
    PopulationCalibrator,
    SurfaceLayer,
)


class TestModels(unittest.TestCase):

    def test_essay_creation(self):
        e = Essay(text="Hello world.", title="Test")
        self.assertEqual(e.title, "Test")
        self.assertEqual(e.genre, "essay")

    def test_layer_result(self):
        lr = LayerResult(
            layer_name="Test",
            score=75.0,
            normalized_score=0.75,
            evidence=["ok"],
            confidence_interval=(70.0, 80.0),
        )
        self.assertEqual(lr.layer_name, "Test")
        self.assertEqual(lr.score, 75.0)
        self.assertEqual(lr.confidence_interval, (70.0, 80.0))

    def test_caeas_report_to_dict(self):
        lr = LayerResult(
            layer_name="Test",
            score=75.0,
            normalized_score=0.75,
            evidence=["ok"],
            confidence_interval=(70.0, 80.0),
        )
        from lgram.essay.models import CAEASReport as CAEASReport_model
        r = CAEASReport_model(
            cohesion_score=75.0,
            composite_indicator=70.0,
            confidence_interval=(70.0, 80.0),
            layer_results=[lr],
            suggestion="OK",
            justification="test",
        )
        d = r.to_dict()
        self.assertEqual(d["cohesion_score"], 75.0)
        self.assertEqual(d["composite_indicator"], 70.0)
        self.assertEqual(d["suggestion"], "OK")


class TestMockContentJudge(unittest.TestCase):

    def setUp(self):
        self.judge = MockContentJudge()

    def test_empty_text(self):
        with self.assertRaises(ValueError):
            Essay(text="", title="Empty")

    def test_single_sentence(self):
        essay = Essay(text="Hello world.", title="Short")
        result = self.judge.evaluate(essay, [])
        self.assertLess(result.score, 50)

    def test_reasonable_essay(self):
        text = (
            "Social media has changed how people communicate today. "
            "Many young people use platforms like Instagram every day. "
            "These apps let users share photos and videos easily. "
            "However, social media also has negative effects on mental health. "
            "For example, studies show that comparison leads to anxiety. "
            "In conclusion, social media is both good and bad for society."
        )
        essay = Essay(text=text, title="Social Media")
        result = self.judge.evaluate(essay, [])
        self.assertTrue(0 <= result.score <= 100)
        self.assertIsInstance(result.evidence, list)

    def test_weak_essay(self):
        text = "I like dogs. Pizza is good. The weather is nice. Cars go fast."
        essay = Essay(text=text, title="Random")
        result = self.judge.evaluate(essay, [])
        self.assertLess(result.score, 60)

    def test_result_has_evidence(self):
        essay = Essay(text="This is a very short essay without much content.", title="Short")
        result = self.judge.evaluate(essay, [])
        self.assertGreater(len(result.evidence), 0)


class TestCohesionLayer(unittest.TestCase):

    def setUp(self):
        self.layer = CohesionLayer()

    def test_coherent_essay(self):
        text = (
            "Alice went to the park. She sat on a bench. "
            "Alice read a book. She enjoyed it very much. "
            "The weather was beautiful. Alice stayed until sunset."
        )
        essay = Essay(text=text, title="Coherent")
        result = self.layer.evaluate(essay)
        self.assertTrue(0 <= result.score <= 100)
        self.assertIsNotNone(result.confidence_interval)
        self.assertGreater(result.score, 40)

    def test_incoherent_essay(self):
        text = (
            "Quantum physics is fascinating. My neighbor has a dog. "
            "The Roman Empire fell in 476. Pizza is delicious. "
            "Basketball requires coordination. Beethoven was a composer."
        )
        essay = Essay(text=text, title="Incoherent")
        result = self.layer.evaluate(essay)
        self.assertTrue(0 <= result.score <= 100)
        self.assertLess(result.score, 80)

    def test_single_paragraph(self):
        essay = Essay(text="Hello world.", title="Tiny")
        result = self.layer.evaluate(essay)
        self.assertTrue(0 <= result.score <= 100)

    def test_multi_paragraph(self):
        text = (
            "First paragraph with some content here.\n\n"
            "Second paragraph with different content there.\n\n"
            "Third paragraph concluding the discussion."
        )
        essay = Essay(text=text, title="Multi")
        result = self.layer.evaluate(essay)
        self.assertGreaterEqual(result.raw_details["segment_count"], 3)


class TestSurfaceLayer(unittest.TestCase):

    def setUp(self):
        self.layer = SurfaceLayer()

    def test_empty_text(self):
        with self.assertRaises(ValueError):
            Essay(text="", title="Empty")

    def test_short_text(self):
        essay = Essay(text="Hello world.", title="Short")
        result = self.layer.evaluate(essay)
        self.assertTrue(0 <= result.score <= 100)

    def test_reasonable_text(self):
        text = (
            "The impact of social media on modern society is profound and multifaceted. "
            "Research demonstrates that platforms like Instagram and TikTok reshape how "
            "young people form identities and maintain relationships. Furthermore, these "
            "technologies introduce unprecedented challenges regarding privacy, mental "
            "health, and attention spans. Consequently, educators and policymakers must "
            "develop comprehensive strategies to address these emerging concerns."
        )
        essay = Essay(text=text, title="Academic")
        result = self.layer.evaluate(essay)
        self.assertTrue(0 <= result.score <= 100)
        details = result.raw_details
        self.assertIn("readability", details)
        self.assertIn("vocabulary", details)
        self.assertIn("sentence_variety", details)
        self.assertIn("grammar_complexity", details)

    def test_ttr_computes(self):
        essay = Essay(
            text="The cat sat on the mat. The cat was happy. Cats are great pets.",
            title="TTR Test",
        )
        result = self.layer.evaluate(essay)
        ttr = result.raw_details["vocabulary"]["ttr"]
        self.assertGreater(ttr, 0.0)
        self.assertLessEqual(ttr, 1.0)


class TestPopulationCalibrator(unittest.TestCase):

    def setUp(self):
        self.cal = PopulationCalibrator()

    def test_insufficient_samples(self):
        report = self.cal.calibrate(
            "test_pop",
            machine_scores=[70.0, 80.0],
            human_scores=[[72.0, 75.0], [78.0, 82.0]],
        )
        self.assertFalse(report.ready)
        self.assertIn("INSUFFICIENT", report.recommendation)

    def test_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            self.cal.calibrate(
                "test_pop",
                machine_scores=[70.0, 80.0, 90.0],
                human_scores=[[70.0, 75.0], [80.0, 85.0]],
            )

    def test_perfect_agreement(self):
        n = 50
        scores = [float(i * 2) for i in range(n)]
        human = [[s, s + 0.1] for s in scores]
        report = self.cal.calibrate("perfect", scores, human)
        self.assertGreater(report.qwk, 0.85)
        self.assertGreater(report.icc, 0.85)

    def test_poor_agreement(self):
        n = 50
        machine = [50.0 + i for i in range(n)]
        human = [[100.0 - i, 100.0 - i + 2] for i in range(n)]
        report = self.cal.calibrate("poor", machine, human)
        self.assertLess(report.qwk, 0.5)

    def test_calibration_bins(self):
        n = 50
        machine = [float(i) for i in range(50, 100)]
        human = [[float(i + 5), float(i + 3)] for i in range(50, 100)]
        report = self.cal.calibrate("binned", machine, human)
        self.assertGreater(len(report.recalibration_bins), 0)

    def test_single_rater(self):
        n = 50
        scores = [75.0] * n
        human = [[s] for s in scores]
        report = self.cal.calibrate("single_rater", scores, human)
        self.assertEqual(report.icc, 0.0)


class TestConfidenceLayer(unittest.TestCase):

    def setUp(self):
        self.conf = ConfidenceLayer(borderline_margin=5.0)

    def test_aggregate_ci(self):
        lrs = [
            LayerResult(
                layer_name="A",
                score=70.0,
                normalized_score=0.7,
                confidence_interval=(65.0, 75.0),
            ),
            LayerResult(
                layer_name="B",
                score=80.0,
                normalized_score=0.8,
                confidence_interval=(76.0, 84.0),
            ),
        ]
        result = self.conf.analyze(75.0, lrs, [0.5, 0.5])
        self.assertIn("confidence_interval", result)
        ci = result["confidence_interval"]
        self.assertLessEqual(ci[0], ci[1])

    def test_borderline_detection(self):
        lrs = [
            LayerResult(
                layer_name="A",
                score=51.0,
                normalized_score=0.51,
                confidence_interval=(48.0, 54.0),
            ),
        ]
        result = self.conf.analyze(51.0, lrs)
        self.assertTrue(result["borderline"])
        self.assertTrue(result["human_review_recommended"])

    def test_clear_score(self):
        lrs = [
            LayerResult(
                layer_name="A",
                score=96.0,
                normalized_score=0.96,
                confidence_interval=(94.0, 98.0),
            ),
        ]
        result = self.conf.analyze(96.0, lrs)
        self.assertFalse(result["borderline"])

    def test_wide_gap_trigger(self):
        lrs = [
            LayerResult(
                layer_name="A",
                score=90.0,
                normalized_score=0.9,
                confidence_interval=(88.0, 92.0),
            ),
            LayerResult(
                layer_name="B",
                score=40.0,
                normalized_score=0.4,
                confidence_interval=(35.0, 45.0),
            ),
        ]
        result = self.conf.analyze(65.0, lrs)
        self.assertTrue(result["human_review_recommended"])
        self.assertGreater(len(result["triggers"]), 0)

    def test_justification_produced(self):
        lrs = [
            LayerResult(
                layer_name="A",
                score=72.0,
                normalized_score=0.72,
                evidence=["Good structure"],
                confidence_interval=(68.0, 76.0),
            ),
        ]
        result = self.conf.analyze(72.0, lrs)
        self.assertIn("Cohesion analysis complete", result["justification"])


class TestCAEASGrader(unittest.TestCase):

    def setUp(self):
        self.grader = CAEASGrader(use_grammar=False, use_mechanics=False, use_llm=False)

    def test_grade_coherent_essay(self):
        essay = Essay(
            title="Test",
            text=(
                "Alice went to the park. She sat on a bench. "
                "Alice read a book. She enjoyed it very much. "
                "The weather was beautiful. Alice stayed until sunset."
            ),
        )
        report = self.grader.grade(essay)
        self.assertGreaterEqual(report.overall_score, 0)
        self.assertLessEqual(report.overall_score, 100)
        self.assertEqual(len(report.layer_results), 5)
        self.assertIsNotNone(report.verdict)
        self.assertIsNotNone(report.justification)
        self.assertIsInstance(report.to_dict(), dict)

    def test_grade_weak_essay(self):
        essay = Essay(
            title="Weak",
            text="I like dogs. Pizza is good. The weather is nice.",
        )
        report = self.grader.grade(essay)
        self.assertLess(report.overall_score, 80)

    def test_batch_grading(self):
        essays = [
            Essay(title=f"Essay {i}", text=f"This is essay number {i} with some content. It has multiple sentences for testing purposes.")
            for i in range(3)
        ]
        reports = self.grader.grade_batch(essays)
        self.assertEqual(len(reports), 3)
        for r in reports:
            self.assertGreaterEqual(r.overall_score, 0)

    def test_set_content_judge(self):
        grader = CAEASGrader()
        judge = MockContentJudge()
        grader.set_content_analyzer(judge)
        essay = Essay(title="T", text="Hello world.")
        report = grader.analyze(essay)
        self.assertIsNotNone(report)

    def test_uncalibrated_warning(self):
        essay = Essay(title="T", text="Hello world. This is a test.")
        report = self.grader.grade(essay)
        triggers = report.triggers
        has_warning = any(
            "No population calibration" in t or "Using general CEFR" in t
            for t in triggers
        )
        self.assertTrue(has_warning)

    def test_recalibration_applied(self):
        n = 200
        machine = [75.0] * n
        human = [[75.0, 75.0]] * n
        cal = self.grader.calibrator.calibrate("test", machine, human)
        self.grader.set_calibration(cal)
        self.assertTrue(self.grader.calibration_ready)

    def test_report_triggers(self):
        essay = Essay(title="T", text="A. B. C.")
        report = self.grader.grade(essay)
        self.assertIsInstance(report.triggers, list)
        self.assertIsInstance(report.teacher_review_recommended, bool)
        self.assertIsInstance(report.borderline, bool)


class TestDiscriminantValidity(unittest.TestCase):
    """Sanity check: system MUST differentiate good vs bad essays on pure cohesion."""

    def setUp(self):
        self.grader = CAEASGrader(use_grammar=False, use_mechanics=False, use_llm=False)

    def test_cohesion_differentiates_good_vs_bad(self):
        good = Essay(title="Good", text=(
            "Technology has transformed education in many important ways. "
            "First, online learning platforms have made education accessible to "
            "students in remote areas. For example, students in rural regions "
            "can now attend virtual classes from top universities. "
            "Second, digital tools enable personalized learning experiences. "
            "Teachers can track individual student progress and adjust their "
            "instruction accordingly. However, technology also presents challenges. "
            "Not all students have reliable internet access, which creates a "
            "digital divide. Furthermore, excessive screen time may affect "
            "students concentration and health. In conclusion, while technology "
            "offers significant benefits for education, schools must address "
            "equity concerns and establish healthy usage guidelines."
        ))
        bad = Essay(title="Bad", text=(
            "Technology is good. I like computers. My neighbor has a dog. "
            "The weather today is sunny. Schools should have more technology. "
            "Pizza is my favorite food. Students use the internet for research."
        ))
        gr = self.grader.analyze(good)
        br = self.grader.analyze(bad)

        self.assertGreater(
            gr.cohesion_score, br.cohesion_score,
            f"Good essay cohesion ({gr.cohesion_score:.0f}) must exceed "
            f"bad essay cohesion ({br.cohesion_score:.0f})"
        )

        self.assertGreater(
            gr.cohesion_score - br.cohesion_score, 10,
            f"Cohesion gap ({gr.cohesion_score - br.cohesion_score:.0f}) "
            f"too small — good ({gr.cohesion_score:.0f}) vs bad ({br.cohesion_score:.0f})"
        )

        self.assertGreater(
            gr.composite_indicator, br.composite_indicator,
            f"Composite indicator: good ({gr.composite_indicator:.0f}) must exceed "
            f"bad ({br.composite_indicator:.0f})"
        )

    def test_cohesion_solo_signal_stronger_than_composite(self):
        good = Essay(title="Good", text=(
            "Alice went to the park yesterday afternoon with her dog. "
            "She sat on a bench near the lake and watched the ducks. "
            "Alice then read a book that she had borrowed from the library. "
            "She enjoyed the story very much because it was exciting. "
            "The weather was beautiful with a gentle breeze. "
            "Alice stayed at the park until the sun began to set."
        ))
        bad = Essay(title="Bad", text=(
            "Alice likes apples. Quantum physics explains particles. "
            "My car needs an oil change soon. Beethoven wrote symphonies. "
            "The weather report predicts rain tomorrow afternoon."
        ))
        gr = self.grader.analyze(good)
        br = self.grader.analyze(bad)

        self.assertGreater(gr.cohesion_score, br.cohesion_score + 15)

    def test_cohesion_fields_exist(self):
        essay = Essay(title="T", text="A simple test. With multiple sentences. For cohesion checking.")
        report = self.grader.analyze(essay)
        self.assertTrue(hasattr(report, "cohesion_score"))
        self.assertTrue(hasattr(report, "composite_indicator"))
        self.assertGreaterEqual(report.cohesion_score, 0)
        self.assertLessEqual(report.cohesion_score, 100)
        self.assertGreaterEqual(report.composite_indicator, 0)
        self.assertLessEqual(report.composite_indicator, 100)

    def test_single_foreign_sentence_localized(self):
        clean = Essay(title="Clean", text=(
            "Technology has transformed education in many important ways. "
            "First, online learning platforms have made education accessible to "
            "students in remote areas. For example, students in rural regions "
            "can now attend virtual classes from top universities. "
            "Second, digital tools enable personalized learning experiences. "
            "However, technology also presents challenges. "
            "In conclusion, while technology offers significant benefits, "
            "schools must address equity concerns."
        ))
        contaminated = Essay(title="Contaminated", text=(
            "Technology has transformed education in many important ways. "
            "First, online learning platforms have made education accessible to "
            "students in remote areas. For example, students in rural regions "
            "can now attend virtual classes from top universities. "
            "Second, digital tools enable personalized learning experiences. "
            "Pizza is my favorite food with extra cheese. "
            "However, technology also presents challenges. "
            "In conclusion, while technology offers significant benefits, "
            "schools must address equity concerns."
        ))
        gr = self.grader.analyze(clean)
        cr = self.grader.analyze(contaminated)

        self.assertGreater(gr.cohesion_score, cr.cohesion_score,
            f"Clean ({gr.cohesion_score:.0f}) must exceed contaminated ({cr.cohesion_score:.0f})")

        self.assertGreater(gr.cohesion_score - cr.cohesion_score, 5,
            f"Single foreign sentence should drop cohesion by >5pts, "
            f"got {gr.cohesion_score - cr.cohesion_score:.0f}")


if __name__ == "__main__":
    unittest.main()
