#!/usr/bin/env python3
"""
Test centering generation with improved T5 correction
"""

from lgram.models.simple_language_model import create_language_model

def test_centering_with_improved_t5():
    """Test full centering generation with improved T5"""
    print("🎯 Testing centering generation with improved T5 correction...")
    
    try:
        model = create_language_model()
        
        # Test the original sentence that had issues
        input_sentence = "Tell me the true story"
        input_words = input_sentence.split()
        
        print(f"\n📝 Input: {input_sentence}")
        print("=" * 60)
        
        # Generate text with centering
        generated_text = model.generate_text_with_centering(
            num_sentences=3,
            input_words=input_words,
            length=12,
            use_progress_bar=True
        )
        
        print(f"\n✨ Generated with Improved T5:")
        print(generated_text)
        
        # Test cache stats
        cache_stats = model.get_cache_stats()
        print(f"\n📊 Cache Performance:")
        print(f"Overall hit rate: {cache_stats['overall_performance']['overall_hit_rate']:.1%}")
        
        print("\n✅ Centering generation with improved T5 completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_centering_with_improved_t5()