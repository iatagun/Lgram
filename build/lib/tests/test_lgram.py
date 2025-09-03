"""
Test suite for the Lgram package.
"""

import unittest
import sys
import os

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestLgram(unittest.TestCase):
    """Basic tests for Lgram functionality."""
    
    def test_import(self):
        """Test that the package can be imported."""
        try:
            import lgram
            self.assertTrue(hasattr(lgram, '__version__'))
            self.assertTrue(hasattr(lgram, 'EnhancedLanguageModel'))
            self.assertTrue(hasattr(lgram, 'create_language_model'))
        except ImportError as e:
            self.fail(f"Could not import lgram: {e}")
    
    def test_config(self):
        """Test configuration class."""
        try:
            from lgram import Config
            self.assertTrue(hasattr(Config, 'DEFAULT_NUM_SENTENCES'))
            self.assertTrue(hasattr(Config, 'DEFAULT_SENTENCE_LENGTH'))
            self.assertTrue(hasattr(Config, 'SEMANTIC_THRESHOLD'))
        except ImportError as e:
            self.fail(f"Could not import Config: {e}")
    
    def test_utils(self):
        """Test utility functions."""
        try:
            from lgram.utils import validate_model_config, format_generated_text
            
            # Test configuration validation
            valid_config = {
                'DEFAULT_NUM_SENTENCES': 5,
                'DEFAULT_SENTENCE_LENGTH': 13,
                'MIN_SENTENCE_LENGTH': 5,
                'SEMANTIC_THRESHOLD': 0.65
            }
            self.assertTrue(validate_model_config(valid_config))
            
            # Test text formatting
            text = "This is a test sentence. This is another test sentence."
            formatted = format_generated_text(text, max_line_length=50)
            self.assertIsInstance(formatted, str)
            self.assertIn('\n', formatted)
            
        except ImportError as e:
            self.fail(f"Could not import utils: {e}")

if __name__ == '__main__':
    unittest.main()
