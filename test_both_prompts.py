#!/usr/bin/env python3
"""
Test both prompt styles for T5 correction
"""

from lgram.models.simple_language_model import create_language_model

def test_both_prompt_styles():
    """Test both comprehensive and simple prompt styles"""
    print("🧪 Testing both T5 prompt styles...")
    
    try:
        model = create_language_model()
        
        # Test cases
        test_cases = [
            "Tell me the true story of it on the other hand. the story for them also that I. they really have to she clasp her room.",
            "This are a test sentence with grammar errors.",
            "She don't know what to do about this."
        ]
        
        print("\n📝 Comparing Comprehensive vs Simple prompt styles:\n")
        
        for i, test_text in enumerate(test_cases, 1):
            print(f"Test {i}:")
            print(f"Original:      {test_text}")
            
            # Test comprehensive prompt
            comprehensive = model.correct_grammar_t5(test_text, prompt_style="comprehensive")
            print(f"Comprehensive: {comprehensive}")
            
            # Test simple prompt  
            simple = model.correct_grammar_t5(test_text, prompt_style="simple")
            print(f"Simple:        {simple}")
            
            print("-" * 100)
        
        print("\n✅ Both prompt styles tested!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_both_prompt_styles()