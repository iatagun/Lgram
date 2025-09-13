#!/usr/bin/env python3
"""
Test improved T5 cleaning to remove prompt contamination
"""

from lgram.models.simple_language_model import create_language_model

def test_prompt_contamination_fix():
    """Test T5 cleaning to prevent prompt contamination"""
    print("🧪 Testing prompt contamination fix...")
    
    try:
        model = create_language_model()
        
        # Test a contaminated input similar to what user saw
        contaminated_text = "Tell me the true story that. he 'll go the country be at a boat have done you. Improve grammar, sentence structure, word choice, clarity, and flow while preserving."
        
        print(f"📝 Contaminated input: {contaminated_text}")
        print("=" * 60)
        
        # Test cleaning with both styles
        print("Testing Simple Style:")
        corrected_simple = model.correct_grammar_t5(contaminated_text, prompt_style="simple")
        print(f"Result: {corrected_simple}")
        
        print("\nTesting Comprehensive Style:")
        corrected_comp = model.correct_grammar_t5(contaminated_text, prompt_style="comprehensive")
        print(f"Result: {corrected_comp}")
        
        # Check if prompt words were removed
        prompt_indicators = ["improve", "grammar", "sentence structure", "word choice", "clarity", "flow while"]
        
        def has_prompt_contamination(text):
            text_lower = text.lower()
            return any(indicator in text_lower for indicator in prompt_indicators)
        
        print("\n" + "=" * 60)
        print("Contamination Analysis:")
        
        simple_clean = not has_prompt_contamination(corrected_simple)
        comp_clean = not has_prompt_contamination(corrected_comp)
        
        if simple_clean:
            print("✅ Simple style: No prompt contamination detected")
        else:
            print("⚠️  Simple style: Still has prompt contamination")
            
        if comp_clean:
            print("✅ Comprehensive style: No prompt contamination detected")
        else:
            print("⚠️  Comprehensive style: Still has prompt contamination")
        
        # Test with generation
        print("\n" + "=" * 60)
        print("Testing full generation pipeline:")
        
        result = model.generate_text_with_centering(
            num_sentences=2,
            input_words=["Tell", "me", "the", "story"],
            t5_prompt_style="comprehensive"
        )
        
        print(f"Generated result: {result}")
        
        if not has_prompt_contamination(result):
            print("✅ Full pipeline: No prompt contamination!")
        else:
            print("⚠️  Full pipeline: Still has some contamination")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_prompt_contamination_fix()