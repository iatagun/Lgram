import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Use the updated models directly from models folder
from models.simple_language_model import create_language_model
model = create_language_model()
        
# Generate text using centering theory
input_sentence = "The truth "
input_words = input_sentence.strip().rstrip('.').split()

print("Testing generate_text_with_centering function...")
print(f"Input: {input_sentence}")
print("="*50)

generated_text = model.generate_text_with_centering(
    num_sentences=5,
    input_words=input_words,
    length=13,
    use_progress_bar=True
)

print("\nGenerated Text with Centering:")
print(generated_text)
print("="*50)