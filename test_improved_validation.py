#!/usr/bin/env python3
"""
Test improved T5 validation with sentence preservation
"""

from lgram.models.simple_language_model import create_language_model

def test_improved_validation():
    """Test T5 correction with improved validation"""
    print("🧪 Testing improved T5 validation...")
    
    try:
        model = create_language_model()
        
        # Test input
        input_sentence = "Tell me the true story"
        input_words = input_sentence.split()
        
        print(f"📝 Input: {input_sentence}")
        print("Requesting 3 sentences...")
        print("=" * 50)
        
        # Test generation with improved system
        result = model.generate_text_with_centering(
            num_sentences=3,
            input_words=input_words,
            t5_prompt_style="simple"
        )
        
        print(f"Generated result: {result}")
        
        # Count sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', result.strip())
        sentences = [s for s in sentences if s.strip()]
        print(f"Sentence count: {len(sentences)}")
        
        # Check individual sentences
        for i, sent in enumerate(sentences, 1):
            print(f"{i}. {sent}")
        
        if len(sentences) == 3:
            print("\n✅ Success! Generated exactly 3 sentences with T5 correction")
        else:
            print(f"\n⚠️  Generated {len(sentences)} sentences instead of 3")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_improved_validation()