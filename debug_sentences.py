#!/usr/bin/env python3
"""
Debug sentence count issue
"""

from lgram.models.simple_language_model import create_language_model

def debug_sentence_count():
    """Debug why 3 sentences become 2"""
    print("🔍 Debugging sentence count issue...")
    
    try:
        model = create_language_model()
        
        # Test input
        input_sentence = "Tell me the true story"
        input_words = input_sentence.split()
        
        print(f"\n📝 Input: {input_sentence}")
        print("Requesting 3 sentences...")
        print("=" * 60)
        
        # Manual step-by-step generation to debug
        generated_sentences = []
        
        for i in range(3):
            if i == 0:
                # First sentence with input words
                sentence = model.generate_sentence(input_words, 12)
            else:
                # Use centering theory for subsequent sentences
                if model.centering and generated_sentences:
                    prev_sentence = generated_sentences[-1]
                    center = model._extract_center_from_sentence(prev_sentence)
                    start_words = [center] if center else None
                else:
                    start_words = None
                    
                sentence = model.generate_sentence(start_words, 12)
            
            # Ensure sentence ends properly
            if sentence and not sentence.endswith(('.', '!', '?')):
                sentence += '.'
            
            print(f"Generated sentence {i+1}: {sentence}")
            generated_sentences.append(sentence)
        
        print(f"\nTotal generated sentences: {len(generated_sentences)}")
        print("Before T5 correction:")
        for i, sent in enumerate(generated_sentences, 1):
            print(f"{i}. {sent}")
        
        # Join and apply T5 correction
        final_text = " ".join(generated_sentences)
        print(f"\nJoined text: {final_text}")
        
        # Test T5 correction
        corrected_simple = model.correct_grammar_t5(final_text, prompt_style="simple")
        print(f"\nAfter T5 simple: {corrected_simple}")
        
        # Count sentences in corrected text
        import re
        corrected_sentences = re.split(r'(?<=[.!?])\s+', corrected_simple.strip())
        corrected_sentences = [s for s in corrected_sentences if s.strip()]
        
        print(f"\nSentences after T5 correction: {len(corrected_sentences)}")
        for i, sent in enumerate(corrected_sentences, 1):
            print(f"{i}. {sent}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_sentence_count()