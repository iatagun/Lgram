from transformers import BartTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
import torch
import re

# Load the BART model and tokenizer, and the sentence embedding model for similarity
model_name = "facebook/bart-large"
tokenizer = BartTokenizer.from_pretrained(model_name)
bart_model = BartForConditionalGeneration.from_pretrained(model_name)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient for sentence similarity

# Function to generate paraphrased text
def paraphrase_text(text, max_length=60, num_return_sequences=3):
    input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True, max_length=512)
    
    # Generate paraphrases
    outputs = bart_model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        num_beams=8,
        early_stopping=True,
        do_sample=True,
        temperature=0.7,          # Adjusted for diversity
        top_k=100,                # Limits the pool of next words for variety
        top_p=0.90                # Nucleus sampling
    )
    
    paraphrases = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return paraphrases

# Select the best paraphrase based on cosine similarity
def best_paraphrase_by_similarity(original_text, paraphrase_variants):
    original_embedding = embedding_model.encode(original_text, convert_to_tensor=True)
    paraphrase_embeddings = embedding_model.encode(paraphrase_variants, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(original_embedding, paraphrase_embeddings)
    best_idx = torch.argmax(similarities).item()
    return paraphrase_variants[best_idx]

# Function to check for repetitive patterns
def is_repetitive(text):
    return bool(re.search(r'\b(\w+)\b(?:.*\b\1\b){2,}', text))

# Original Text
text = """
    There was as ill at upper part struck was likely that the key has a more stately than upon the two or else of resistance. 
    Twitch of immorality which were stationary. Coping had sprung out the soil, there is hastened to one of his visits that of life to determine whether with members of the working. 
    Ramble in an exclamation in the presence of the other individual has withheld from all the other hand on accounts and the night, though. 
    Pitiful arrears; he determined that he, his face, i was his duel with a time we had enough to those miserable. 
    Individual than equivocal life -- was bonded for the letter in the other piece of his departure, it would have been interrupted. 
    Honorable fellow, glancing at the personage who bore with the housekeeper's secretary came by saying, and his behalf of the opinion that. 
    Prayed for yourself. but it is possible that we not so much that i do not good god. 
    India in ninety thousand when he had the right to the truth was the internal, but the more than the assertions have. 
    Turfed ground, with a very unusual gaping hole had small boat to the fort at the slopes of the man of the embodiment of this.
"""

# Split text into sentences
sentences = text.split('.')
sentences = [sentence.strip() for sentence in sentences if sentence]

# Paraphrased sentences list
paraphrased_sentences = []

for sentence in sentences:
    # Generate paraphrases
    paraphrase_variants = paraphrase_text(sentence)
    
    # Choose the best paraphrase based on cosine similarity
    best_paraphrase = best_paraphrase_by_similarity(sentence, paraphrase_variants)
    
    # Filter out repetitive paraphrases
    if not is_repetitive(best_paraphrase) and best_paraphrase not in paraphrased_sentences:
        paraphrased_sentences.append(best_paraphrase)

# Combine paraphrased sentences into coherent text
paraphrased_text = '. '.join(paraphrased_sentences) + '.'

# Print the result
print("Paraphrased Text:\n", paraphrased_text)
