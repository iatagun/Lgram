#!/usr/bin/env python3
"""
Debug contamination detection
"""

from lgram.models.simple_language_model import create_language_model

def debug_contamination():
    """Debug contamination detection"""
    print("🔍 Debugging contamination detection...")
    
    try:
        model = create_language_model()
        
        # Test contamination detection
        test_sentences = [
            "Tell me the story.",
            "Improve grammar, sentence structure and word choice.",
            "Tell me the story. Improve grammar, sentence structure and word choice."
        ]
        
        for sentence in test_sentences:
            is_contaminated = model._has_prompt_contamination(sentence)
            print(f"'{sentence}' -> Contaminated: {is_contaminated}")
        
        print("\nTesting correction with debug:")
        contaminated = "Improve grammar, sentence structure and word choice."
        print(f"Input: {contaminated}")
        print(f"Contamination check: {model._has_prompt_contamination(contaminated)}")
        
        # Test rule-based correction
        rule_based = model.correct_grammar(contaminated)
        print(f"Rule-based result: {rule_based}")
        
        # Test T5 correction
        t5_result = model.correct_grammar_t5_preserve_sentences(contaminated, "comprehensive")
        print(f"T5 preserve result: {t5_result}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_contamination()