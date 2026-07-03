import os
import tempfile
import unittest
from unittest.mock import patch

from lgram.cli import main


class TestCLI(unittest.TestCase):

    def test_version_command(self):
        with patch("sys.argv", ["cli", "version"]):
            result = main()
            self.assertEqual(result, 0)

    def test_info_command(self):
        with patch("sys.argv", ["cli", "info"]):
            result = main()
            self.assertEqual(result, 0)

    def test_no_command_prints_help(self):
        with patch("sys.argv", ["cli"]):
            result = main()
            self.assertEqual(result, 1)

    def test_unknown_command(self):
        with patch("sys.argv", ["cli", "nonexistent"]):
            try:
                main()
                self.fail("Expected SystemExit")
            except SystemExit:
                pass

    def test_score_text(self):
        with patch("sys.argv", [
            "cli", "score", "-t",
            "John went to the store. He bought milk. John paid. He left.",
        ]):
            result = main()
            self.assertEqual(result, 0)

    def test_analyze_text(self):
        with patch("sys.argv", [
            "cli", "analyze", "-t",
            "John went to the store. He bought milk. John paid. He left.",
        ]):
            result = main()
            self.assertEqual(result, 0)

    def test_clauses_text(self):
        with patch("sys.argv", [
            "cli", "clauses", "-t",
            "John went to the store because he needed milk, but the store was closed.",
        ]):
            result = main()
            self.assertEqual(result, 0)

    def test_full_text(self):
        with patch("sys.argv", [
            "cli", "full", "-t",
            "John went to the store. He bought milk because he was hungry.",
        ]):
            result = main()
            self.assertEqual(result, 0)

    def test_score_from_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8",
        ) as f:
            f.write("John went. He bought milk. John paid. He left.")
            tmp = f.name
        try:
            with patch("sys.argv", ["cli", "score", "-f", tmp]):
                result = main()
                self.assertEqual(result, 0)
        finally:
            os.unlink(tmp)

    def test_score_missing_input(self):
        with patch("sys.argv", ["cli", "score"]):
            with self.assertRaises(ValueError):
                main()

    def test_score_file_not_found(self):
        with patch("sys.argv", ["cli", "score", "-f", "nonexistent_file.txt"]):
            with self.assertRaises(FileNotFoundError):
                main()

    def test_analyze_empty_text(self):
        with patch("sys.argv", ["cli", "analyze", "-t", ""]):
            result = main()
            self.assertEqual(result, 1)

    def test_score_verbose(self):
        with patch("sys.argv", [
            "cli", "analyze", "-v", "-t",
            "John went to the store. He bought milk.",
        ]):
            result = main()
            self.assertEqual(result, 0)

    def test_clauses_simple_sentence(self):
        with patch("sys.argv", [
            "cli", "clauses", "-t", "John walked home.",
        ]):
            result = main()
            self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
