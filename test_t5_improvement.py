#!/usr/bin/env python3
"""
Test improved T5 grammar correction
"""

from lgram.models.simple_language_model import create_language_model

def test_t5_improvements():
    """Test T5 correction with improved settings"""
    print("🧪 Testing improved T5 grammar correction...")
    
    try:
        model = create_language_model()
        
        # Test cases with various grammar issues
        test_cases = [
            "Tell me the true story of it on the other hand. the story for them also that I. they really have to she clasp her room.",
            "This are a test sentence with grammar errors.",
            "The cat was sat on the mat yesterday evening.",
            "She don't know what to do about this.",
            "Me and him went to the store together."
        ]
        
        print("\n📝 Testing T5 correction on various grammar issues:\n")
        
        for i, test_text in enumerate(test_cases, 1):
            print(f"Test {i}:")
            print(f"Original:  {test_text}")
            
            # Test T5 correction
            corrected = model.correct_grammar_t5(test_text)
            print(f"Corrected: {corrected}")
            print("-" * 80)
        
        print("\n✅ T5 correction testing completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_t5_improvements()