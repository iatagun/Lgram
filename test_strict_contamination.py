#!/usr/bin/env python3
"""
Test strict contamination validation and fallback system
"""

from lgram.models.simple_language_model import create_language_model

def test_strict_contamination():
    """Test strict contamination detection and fallback"""
    print("🧪 Testing strict contamination validation...")
    
    try:
        model = create_language_model()
        
        # Test input
        input_sentence = "Tell me the true story"
        input_words = input_sentence.split()
        
        print(f"📝 Input: {input_sentence}")
        print("Requesting 3 sentences with strict validation...")
        print("=" * 60)
        
        # Test comprehensive style (should be clean)
        result = model.generate_text_with_centering(
            num_sentences=3,
            input_words=input_words,
            t5_prompt_style="comprehensive"
        )
        
        print(f"Generated result: {result}")
        print()
        
        # Check contamination
        if model._has_prompt_contamination(result):
            print("⚠️  Still has contamination")
        else:
            print("✅ No contamination detected!")
        
        # Count sentences
        import re
        sentences = re.split(r'(?<=[.!?])\s+', result.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        print(f"Sentence count: {len(sentences)}")
        print("Individual sentences:")
        for i, sent in enumerate(sentences, 1):
            print(f"{i}. {sent}")
            if model._has_prompt_contamination(sent):
                print(f"   ⚠️  Sentence {i} has contamination")
            else:
                print(f"   ✅ Sentence {i} is clean")
        
        print("\n" + "=" * 60)
        print("Testing individual contaminated sentence:")
        
        # Test a known contaminated sentence
        contaminated = "Tell me the story. Improve grammar, sentence structure and word choice."
        corrected = model.correct_grammar_t5_preserve_sentences(contaminated, "comprehensive")
        
        print(f"Input: {contaminated}")
        print(f"Corrected: {corrected}")
        
        if model._has_prompt_contamination(corrected):
            print("⚠️  Correction still contaminated (should use rule-based fallback)")
        else:
            print("✅ Contamination successfully removed/avoided")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_strict_contamination()