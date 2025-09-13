#!/usr/bin/env python3
"""
Test centering generation with both T5 prompt styles
"""

from lgram.models.simple_language_model import create_language_model

def test_centering_with_both_styles():
    """Test centering generation with both T5 prompt styles"""
    print("🎯 Testing centering generation with both T5 prompt styles...")
    
    try:
        model = create_language_model()
        
        # Test input
        input_sentence = "Tell me the true story"
        input_words = input_sentence.split()
        
        print(f"\n📝 Input: {input_sentence}")
        print("=" * 80)
        
        # Test with comprehensive prompt
        print("\n🔍 COMPREHENSIVE PROMPT (Original):")
        comprehensive_text = model.generate_text_with_centering(
            num_sentences=3,
            input_words=input_words,
            length=12,
            use_progress_bar=True,
            t5_prompt_style="comprehensive"
        )
        print(f"Result: {comprehensive_text}")
        
        print("\n" + "="*80)
        
        # Test with simple prompt  
        print("\n⚡ SIMPLE PROMPT (More Aggressive):")
        simple_text = model.generate_text_with_centering(
            num_sentences=3,
            input_words=input_words,
            length=12,
            use_progress_bar=True,
            t5_prompt_style="simple"
        )
        print(f"Result: {simple_text}")
        
        print("\n" + "="*80)
        print("\n✅ Both T5 prompt styles tested in centering generation!")
        
        # Cache performance
        cache_stats = model.get_cache_stats()
        print(f"\n📊 Cache Performance: {cache_stats['overall_performance']['overall_hit_rate']:.1%}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_centering_with_both_styles()