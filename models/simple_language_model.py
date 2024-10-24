import random
import pickle
from collections import defaultdict, Counter
import spacy
import tensorflow as tf

# Load the SpaCy English model
nlp = spacy.load("en_core_web_sm")

class SimpleLanguageModel:
    def __init__(self, text, n=2, smoothing='laplace'):
        self.n = n  # N-gram size
        self.smoothing = smoothing  # Smoothing type
        self.model, self.total_counts = self.build_model(text)  # Combine model and total counts
        
    def build_model(self, text):
        model = defaultdict(lambda: defaultdict(int))
        doc = nlp(text.lower())
        
        # Create n-grams
        tokens = [token.text for token in doc]
        n_grams = [tuple(tokens[i:i+self.n]) for i in range(len(tokens)-self.n+1)]
        
        for n_gram in n_grams:
            prefix = n_gram[:-1]  # All but last word
            next_word = n_gram[-1]
            model[prefix][next_word] += 1
        
        # Normalize probabilities (add smoothing)
        total_counts = defaultdict(int)
        for prefix, next_words in model.items():
            total_count = sum(next_words.values())
            total_counts[prefix] = total_count
            
            for word in next_words:
                # Laplace smoothing
                next_words[word] = (next_words[word] + 1) / (total_count + len(next_words))

        return dict(model), dict(total_counts)  # Convert to regular dicts

    def generate_sentence(self, start_words=None, length=10):
        if start_words is None:
            start_words = random.choice(list(self.model.keys()))
        else:
            start_words = tuple(start_words)
        
        current_words = list(start_words)
        sentence = current_words.copy()

        for _ in range(length):
            prefix = tuple(current_words[-(self.n-1):])  # Get the last n-1 words
            next_words = self.model.get(prefix, {})
            
            if not next_words:
                break  # Stop if no next words

            # Choose a word with more diversity
            next_word = random.choices(list(next_words.keys()), weights=next_words.values())[0]
            sentence.append(next_word)
            current_words.append(next_word)

        # Clean up the sentence structure
        sentence_text = ' '.join(sentence).strip()
        sentence_text = self.clean_text(sentence_text)  # Clean the text

        # Add a period at the end of the sentence
        if sentence_text:
            return sentence_text.capitalize() + '.'  # Capitalize the sentence
        return ''
    
    def clean_text(self, text):
        # Normalize spaces: replace multiple spaces with a single space
        text = ' '.join(text.split())
        
        # Fix common punctuation issues
        text = text.replace("..", ".")  # Replace double periods
        text = text.replace(" .", ".")  # Remove space before period
        text = text.replace(" ,", ",")  # Remove space before comma
        text = text.replace(" ;", ";")  # Remove space before semicolon
        text = text.replace(" :", ":")   # Remove space before colon
        text = text.replace("  '", "'")   # Remove space before single quote
        text = text.replace("  ''", "''")  # Remove space before double quote
        text = text.replace('""', '"')  # Fix double quotes
        text = text.replace("''", "'")   # Fix single quotes
        
        # Remove leading and trailing spaces from punctuation
        text = text.replace(" .", ".").replace(" ,", ",").replace(" ;", ";").replace(" :", ":")
        
        # Optionally remove unwanted leading spaces before punctuation
        text = text.strip()
        
        # Handle unwanted punctuation at the start or end of the text
        if text and text[-1] not in ['.', '?', '!']:
            text += '.'
        
        return text

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

def load_transition_model(model_path):
    # Load the transition model
    model = tf.keras.models.load_model(model_path)
    return model

def get_strongest_noun_phrases(input_text):
    doc = nlp(input_text)
    noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks]
    return noun_phrases

def get_transition_words(model, text):
    noun_phrases = get_strongest_noun_phrases(text)
    if noun_phrases:
        return random.choice(noun_phrases)  # Choose a random noun phrase
    return None

# Load the text from the file
file_path = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_data.txt'
text = load_text_from_file(file_path)

# Check if the model file exists
model_file = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\language_model.pkl'

try:
    # Attempt to load the existing model
    language_model = SimpleLanguageModel.load_model(model_file)
    print("Loaded existing model.")
except (FileNotFoundError, EOFError):
    # If the model does not exist, create a new one
    language_model = SimpleLanguageModel(text, n=2)
    language_model.save_model(model_file)  # Save the newly created model
    print("Created and saved new model.")

# Load the transition model
transition_model_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\best_transition_model.keras"
transition_model = load_transition_model(transition_model_path)

# Generate the specified number of sentences
num_sentences = 10  # Number of sentences to generate
generated_sentences = []

# Specify input words
input_words = ["i", "saw"]  # Words to be used

for i in range(num_sentences):
    if i == 0:
        # Use input_words for the first sentence
        generated_sentence = language_model.generate_sentence(start_words=input_words, length=10)
    else:
        # Get the strongest noun phrase from the transition model for subsequent sentences
        topic_word = get_transition_words(transition_model, text)
        if topic_word:
            generated_sentence = language_model.generate_sentence(start_words=[topic_word], length=10)
        else:
            generated_sentence = language_model.generate_sentence(length=10)  # Fallback if no noun phrase found

    generated_sentences.append(generated_sentence)  # Add the sentence to the list

# Join sentences and form the final text
final_text = ' '.join(generated_sentences)
print("Generated Text:\n", final_text)
