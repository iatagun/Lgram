#!/usr/bin/env python3
"""
Debug T5 correction behavior for different prompt styles
"""

from lgram.models.simple_language_model import create_language_model

def debug_t5_styles():
    """Debug T5 correction for different styles"""
    print("🔍 Debugging T5 correction styles...")
    
    try:
        model = create_language_model()
        
        # Create a test text with 3 clear sentences
        test_text = "Tell me the true story about her. The story have been content to conceal in reason to its high road and there. Story have to be contrived for all reason to conceal it's high road and there."
        
        print(f"Original text: {test_text}")
        print("\nOriginal sentence count:", len(test_text.split('. ')))
        
        print("\n" + "=" * 60)
        
        # Test comprehensive T5 correction
        corrected_comp = model.correct_grammar_t5(test_text, prompt_style="comprehensive")
        print(f"Comprehensive corrected: {corrected_comp}")
        
        import re
        sentences_comp = re.split(r'(?<=[.!?])\s+', corrected_comp.strip())
        sentences_comp = [s for s in sentences_comp if s.strip()]
        print(f"Comprehensive sentence count: {len(sentences_comp)}")
        
        print("\n" + "=" * 60)
        
        # Test simple T5 correction
        corrected_simple = model.correct_grammar_t5(test_text, prompt_style="simple")
        print(f"Simple corrected: {corrected_simple}")
        
        sentences_simple = re.split(r'(?<=[.!?])\s+', corrected_simple.strip())
        sentences_simple = [s for s in sentences_simple if s.strip()]
        print(f"Simple sentence count: {len(sentences_simple)}")
        
        # Analysis
        print("\n" + "=" * 60)
        print("Analysis:")
        print(f"Original: 3 sentences")
        print(f"Comprehensive T5: {len(sentences_comp)} sentences")
        print(f"Simple T5: {len(sentences_simple)} sentences")
        
        if len(sentences_comp) < 3:
            print("⚠️  Comprehensive T5 is merging sentences!")
        if len(sentences_simple) >= 3:
            print("✅ Simple T5 preserves sentence count")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_t5_styles()