import random
import pickle
from collections import defaultdict
import spacy
import numpy as np
from tqdm import tqdm  # Import tqdm for progress bar

# Load the SpaCy English model with word vectors
nlp = spacy.load("en_core_web_md")  # Use medium model for better embeddings

class EnhancedLanguageModel:
    def __init__(self, text, n=2):
        self.n = n  # N-gram size
        self.model, self.total_counts = self.build_model(text)  # Combine model and total counts
        
    def build_model(self, text):
        model = defaultdict(lambda: defaultdict(int))
        doc = nlp(text.lower())
        
        # Create n-grams
        tokens = [token.text for token in doc if token.is_alpha]  # Filter out non-alpha tokens
        n_grams = [tuple(tokens[i:i+self.n]) for i in range(len(tokens)-self.n+1)]
        
        for n_gram in n_grams:
            prefix = n_gram[:-1]  # All but last word
            next_word = n_gram[-1]
            model[prefix][next_word] += 1
        
        # Normalize probabilities (add Kneser-Ney smoothing)
        total_counts = defaultdict(int)
        for prefix, next_words in model.items():
            total_count = sum(next_words.values())
            total_counts[prefix] = total_count
            
            for word in next_words:
                # Kneser-Ney smoothing (basic implementation)
                next_words[word] = (next_words[word] + 1) / (total_count + len(next_words))

        return dict(model), dict(total_counts)  # Convert to regular dicts

    def generate_sentence(self, start_words=None, length=10):
        if start_words is None:
            start_words = random.choice(list(self.model.keys()))
        else:
            start_words = tuple(start_words)

        current_words = list(start_words)
        sentence = current_words.copy()

        for _ in tqdm(range(length), desc="Generating words"):
            prefix = tuple(current_words[-(self.n-1):])  # Get the last n-1 words
            next_words = self.model.get(prefix, {})

            if not next_words:
                break  # Stop if no next words

            next_word = self.choose_word_with_context(next_words)  # Use the new method
            sentence.append(next_word)
            current_words.append(next_word)

        sentence_text = ' '.join(sentence).strip()
        sentence_text = self.clean_text(sentence_text)

        return sentence_text

    def choose_word_with_context(self, next_words):
        if not next_words:
            return None  # No next words available

        word_choices = list(next_words.keys())
        probabilities = np.array(list(next_words.values()))

        # Ensure probabilities are non-negative
        probabilities = np.maximum(probabilities, 0)

        # Normalize probabilities
        total = probabilities.sum()
        if total > 0:
            probabilities /= total  # Normalize to sum to 1
        else:
            probabilities = np.ones_like(probabilities) / len(probabilities)  # Equal distribution if all probabilities are zero

        # Bağlam kelimesinin son kelime olduğunu varsayalım (veya başka bir bağlam belirleme yöntemi)
        context_word = word_choices[-1] if word_choices else None

        if context_word:
            # Bağlam kelimesinin vektörünü al
            context_vector = nlp(context_word).vector
            similarity_scores = np.array([np.dot(context_vector, nlp(word).vector) for word in word_choices])  # Kosinüs benzerliği
            
            # Benzerlikler negatifse sıfırlama
            similarity_scores = np.maximum(similarity_scores, 0)

            # Benzerlikleri olasılıklara ekleyin
            adjusted_probabilities = probabilities * (similarity_scores + 1)  # +1 eklenmesi, negatif benzerliklerden kaçınmak için
            adjusted_probabilities /= adjusted_probabilities.sum()  # Normalizasyon
            
            # Olasılıkların pozitif olduğundan emin olun
            adjusted_probabilities = np.maximum(adjusted_probabilities, 0)  # Pozitif olmasını sağla

            # Seçim yap
            chosen_word = np.random.choice(word_choices, p=adjusted_probabilities)
        else:
            chosen_word = np.random.choice(word_choices, p=probabilities)

        return chosen_word

    def clean_text(self, text):
        # Remove unwanted spaces before punctuation
        text = text.replace(" .", ".").replace(" ,", ",").replace(" ;", ";").replace(" :", ":").replace(" ?", "?").replace(" !", "!")
        
        # Fix double punctuation issues
        text = text.replace("..", ".").replace(",,", ",").replace("!!", "!").replace("??", "?")
        
        # Remove leading and trailing quotes if they are misplaced
        text = text.strip("'").strip('"')
        
        # Add a period at the end of the sentence if not already there
        if text and text[-1] not in ['.', '!', '?']:
            text += '.'
        
        # Normalize spaces: ensure only one space between words
        text = ' '.join(text.split())
        
        # Capitalize the first letter of the sentence
        if text:
            text = text[0].capitalize() + text[1:]
        
        return text

    def post_process_sentences(self, sentences):
        # Combine sentences into a single text for analysis
        full_text = ' '.join(sentences)

        # Use SpaCy to process the text
        doc = nlp(full_text)

        # Example: Check for coherence based on named entities or noun phrases
        entities = set(ent.text for ent in doc.ents)  # Extract named entities
        noun_phrases = set(chunk.text for chunk in doc.noun_chunks)  # Extract noun phrases

        # Implement logic to ensure thematic consistency
        # This is a simple heuristic based on the presence of common entities
        if len(entities) < 2:  # If not enough entities, encourage rephrasing
            print("Low entity diversity detected. Rephrasing might be needed.")

        # Optional: Additional logic to adjust sentence structure or rephrase
        adjusted_sentences = []
        for sentence in sentences:
            # Example rephrasing logic could be placed here
            adjusted_sentences.append(self.clean_text(sentence))  # Clean each sentence

        return adjusted_sentences

    def generate_and_post_process(self, num_sentences=10, input_words=None, length=20):
        generated_sentences = []

        for i in tqdm(range(num_sentences), desc="Generating sentences"):
            if i == 0:
                generated_sentence = self.generate_sentence(start_words=input_words, length=length)
            else:
                generated_sentence = self.generate_sentence(length=length)

            generated_sentences.append(generated_sentence)

        # Post-process generated sentences for coherence and flow
        processed_sentences = self.post_process_sentences(generated_sentences)
        final_text = ' '.join(processed_sentences)
        return final_text

    def save_model(self, filename):
        # Convert defaultdicts to regular dicts for pickling
        with open(filename, 'wb') as f:
            pickle.dump((self.model, self.total_counts), f)
        
    @classmethod
    def load_model(cls, filename):
        with open(filename, 'rb') as f:
            model, total_counts = pickle.load(f)
        instance = cls("dummy text")  # We need to create an instance
        instance.model = defaultdict(lambda: defaultdict(int), model)  # Load as defaultdict
        instance.total_counts = total_counts
        return instance


def load_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Load the text from the file
file_path = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_data.txt'
text = load_text_from_file(file_path)

# Check if the model file exists
model_file = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\language_model.pkl'

try:
    # Attempt to load the existing model
    language_model = EnhancedLanguageModel.load_model(model_file)
    print("Loaded existing model.")
except (FileNotFoundError, EOFError):
    # If the model does not exist, create a new one
    language_model = EnhancedLanguageModel(text, n=20)
    language_model.save_model(model_file)  # Save the newly created model
    print("Created and saved new model.")

# Generate the specified number of sentences
num_sentences = 10 # Number of sentences to generate
input_words = ["we", "know", "of", "them"]  # Words to be used

# Use the integrated generate and post-process method
generated_text = language_model.generate_and_post_process(num_sentences=num_sentences, input_words=input_words, length=10)
print("Generated Text:\n", generated_text)
