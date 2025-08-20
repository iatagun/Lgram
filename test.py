import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from lgram.models.chunk import create_language_model
model = create_language_model()
        
# Generate text
input_sentence = "The truth "
input_words = input_sentence.strip().rstrip('.').split()

generated_text = model.generate_text(
    num_sentences=5,
    input_words=input_words,
    length=13,
    use_progress_bar=True
)

print("Generated Text:")
print(generated_text)

# Apply T5 correction
corrected_text = model.correct_grammar_t5(generated_text)
print("\nCorrected Text:")
print(corrected_text)