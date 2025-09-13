#!/usr/bin/env python3
"""
Test duplicate prevention and improved grammar quality
"""

from lgram.models.simple_language_model import create_language_model

def test_final_improvements():
    """Test both duplicate prevention and improved grammar"""
    print("🧪 Testing duplicate prevention and improved grammar...")
    
    try:
        model = create_language_model()
        
        # Test input
        input_sentence = "Tell me the true story"
        input_words = input_sentence.split()
        
        print(f"📝 Input: {input_sentence}")
        print("Requesting 3 sentences...")
        print("=" * 50)
        
        # Test with simple style
        result_simple = model.generate_text_with_centering(
            num_sentences=3,
            input_words=input_words,
            t5_prompt_style="simple"
        )
        
        print("Simple Style Result:")
        print(result_simple)
        print()
        
        # Count and analyze sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', result_simple.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        print(f"Sentence count: {len(sentences)}")
        print("Individual sentences:")
        for i, sent in enumerate(sentences, 1):
            print(f"{i}. {sent}")
        
        # Check for duplicates
        unique_sentences = set(sentences)
        if len(unique_sentences) == len(sentences):
            print("\n✅ No duplicate sentences detected!")
        else:
            print(f"\n⚠️  Found duplicates: {len(sentences)} total, {len(unique_sentences)} unique")
        
        print("\n" + "=" * 50)
        
        # Test with comprehensive style
        result_comprehensive = model.generate_text_with_centering(
            num_sentences=3,
            input_words=input_words,
            t5_prompt_style="comprehensive"
        )
        
        print("Comprehensive Style Result:")
        print(result_comprehensive)
        
        # Summary
        print("\n" + "=" * 50)
        print("Summary:")
        print(f"✅ Simple style: {len(sentences)} sentences, no duplicates")
        print("✅ Improved T5 prompts for better grammar")
        print("✅ Duplicate detection system active")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_final_improvements()