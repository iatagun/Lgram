#!/usr/bin/env python3
"""
Test T5 generation warning fix
"""

from lgram.models.simple_language_model import create_language_model

def test_t5_warning_fix():
    """Test if T5 generation warnings are fixed"""
    print("🧪 Testing T5 generation warning fix...")
    
    try:
        model = create_language_model()
        
        # Test T5 correction with a simple sentence
        test_text = "Tell me the story of book."
        
        print(f"📝 Testing T5 correction on: {test_text}")
        print("=" * 50)
        
        # Test both styles
        print("Testing comprehensive style:")
        result_comp = model.correct_grammar_t5(test_text, prompt_style="comprehensive")
        print(f"Result: {result_comp}")
        
        print("\nTesting simple style:")
        result_simple = model.correct_grammar_t5(test_text, prompt_style="simple")
        print(f"Result: {result_simple}")
        
        # Test full generation
        print("\n" + "=" * 50)
        print("Testing full text generation:")
        
        result = model.generate_text_with_centering(
            num_sentences=2,
            input_words=["Tell", "me"],
            t5_prompt_style="comprehensive"
        )
        
        print(f"Generated: {result}")
        print("\n✅ T5 generation completed - check for any length warnings above")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_t5_warning_fix()