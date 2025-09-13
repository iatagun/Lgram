#!/usr/bin/env python3
"""
Test sentence count fix
"""

from lgram.models.simple_language_model import create_language_model

def test_sentence_count_fix():
    """Test if sentence count is now preserved"""
    print("🧪 Testing sentence count fix...")
    
    try:
        model = create_language_model()
        
        # Test input
        input_sentence = "Tell me the true story"
        
        print(f"📝 Input: {input_sentence}")
        print("Requesting 3 sentences...")
        print("=" * 50)
        
        # Test with comprehensive prompt style
        input_words = input_sentence.split()
        result_comprehensive = model.generate_text_with_centering(
            num_sentences=7,
            input_words=input_words,
            t5_prompt_style="comprehensive"
        )
        
        print(f"Comprehensive result: {result_comprehensive}")
        
        # Count sentences
        import re
        sentences_comp = re.split(r'(?<=[.!?])\s+', result_comprehensive.strip())
        sentences_comp = [s for s in sentences_comp if s.strip()]
        print(f"Comprehensive sentence count: {len(sentences_comp)}")
        
        print("\n" + "=" * 50)
        
        # Test with simple prompt style  
        result_simple = model.generate_text_with_centering(
            num_sentences=3,
            input_words=input_words,
            t5_prompt_style="simple"
        )
        
        print(f"Simple result: {result_simple}")
        
        # Count sentences
        sentences_simp = re.split(r'(?<=[.!?])\s+', result_simple.strip())
        sentences_simp = [s for s in sentences_simp if s.strip()]
        print(f"Simple sentence count: {len(sentences_simp)}")
        
        # Check if both now produce 3 sentences
        if len(sentences_comp) == 3 and len(sentences_simp) == 3:
            print("\n✅ Success! Both styles now produce exactly 3 sentences")
        else:
            print(f"\n⚠️  Still have count issues: comp={len(sentences_comp)}, simple={len(sentences_simp)}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sentence_count_fix()