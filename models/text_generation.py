import numpy as np
import tensorflow as tf
from difflib import SequenceMatcher
import random
from collections import defaultdict
import nltk
from nltk import word_tokenize
from nltk import trigrams
from nltk import bigrams

class DynamicNGram:
    def __init__(self, text, n=2):
        self.model = defaultdict(lambda: defaultdict(lambda: 0))
        self.n = n
        self._build_ngram_model(text)

    def _build_ngram_model(self, text):
        tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in nltk.sent_tokenize(text)]
        for sentence in tokenized_text:
            ngrams = [tuple(sentence[i:i+self.n]) for i in range(len(sentence) - self.n + 1)]
            for ngram in ngrams:
                self.model[ngram[:-1]][ngram[-1]] += 1
        
        for w1 in self.model:
            total_count = float(sum(self.model[w1].values()))
            for w2 in self.model[w1]:
                self.model[w1][w2] /= total_count

    def predict_next_word(self, context):
        context_tuple = tuple(context[-(self.n-1):])
        return self.model.get(context_tuple, {})


class SentenceGenerator:
    def __init__(self, text, transition_model_path, seq_length=15):
        self.sentences = self._split_into_sentences(text)
        self.transition_model = tf.keras.models.load_model(transition_model_path)
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(self.sentences)
        self.seq_length = seq_length
        
        # Initialize models for bigram and trigram
        self.model_bigram = defaultdict(lambda: defaultdict(lambda: 0))
        self.model_trigram = defaultdict(lambda: defaultdict(lambda: 0))
        
        # Build the n-gram models
        self._build_ngram_models(text)
        
        # Set min and max n-gram values based on statistics
        self.min_ngram_length, self.max_ngram_length = self.set_ngram_bounds()

    def _split_into_sentences(self, text):
        return [sentence.strip() for sentence in text.split('.') if sentence]

    def are_sentences_similar(self, sentence1, sentence2, threshold=0.8):
        similarity = SequenceMatcher(None, sentence1, sentence2).ratio()
        return similarity > threshold

    def predict_transition(self, current_sentence, new_sentence):
        current_sequence = self.tokenize(current_sentence)
        new_sequence = self.tokenize(new_sentence)

        current_sequence_padded = tf.keras.preprocessing.sequence.pad_sequences([current_sequence], maxlen=self.seq_length, padding='post')
        new_sequence_padded = tf.keras.preprocessing.sequence.pad_sequences([new_sequence], maxlen=self.seq_length, padding='post')

        input_data = np.concatenate((current_sequence_padded, new_sequence_padded), axis=1)
        prediction = self.transition_model.predict(input_data)

        return np.argmax(prediction, axis=1)

    def tokenize(self, sentence):
        return [self.tokenizer.word_index[word] for word in sentence.split() if word in self.tokenizer.word_index]

    def _build_ngram_models(self, text):
        # Tokenize the text
        tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in nltk.sent_tokenize(text)]

        # Build bigram model
        for sentence in tokenized_text:
            for w1, w2 in bigrams(sentence, pad_right=True, pad_left=True):
                self.model_bigram[w1][w2] += 1

        # Transform counts to probabilities for bigrams
        for w1 in self.model_bigram:
            total_count = float(sum(self.model_bigram[w1].values()))
            for w2 in self.model_bigram[w1]:
                self.model_bigram[w1][w2] /= total_count

        # Build trigram model
        for sentence in tokenized_text:
            for w1, w2, w3 in trigrams(sentence, pad_right=True, pad_left=True):
                self.model_trigram[(w1, w2)][w3] += 1

        # Transform counts to probabilities for trigrams
        for w1_w2 in self.model_trigram:
            total_count = float(sum(self.model_trigram[w1_w2].values()))
            for w3 in self.model_trigram[w1_w2]:
                self.model_trigram[w1_w2][w3] /= total_count

    def set_ngram_bounds(self):
        # Calculate n-gram frequencies
        ngram_counts = defaultdict(int)

        for n in range(2, 6):  # Example: checking for n-grams of length 2 to 5
            dynamic_ngram_model = DynamicNGram(' '.join(self.sentences), n=n)
            total_ngrams = sum(len(v) for v in dynamic_ngram_model.model.values())
            ngram_counts[n] = total_ngrams

        # Set min and max n-gram lengths based on frequency analysis
        min_length = min(ngram_counts, key=ngram_counts.get)
        max_length = max(ngram_counts, key=ngram_counts.get)

        return min_length, max_length

    def generate_sentence(self, min_length=5, max_length=10, prob_threshold=0.02):
        # Use dynamic n-gram length
        ngram_length = random.randint(self.min_ngram_length, self.max_ngram_length)

        # Create dynamic n-gram model with the chosen length
        dynamic_ngram_model = DynamicNGram(' '.join(self.sentences), n=ngram_length)
        
        # Select start words
        start_words = random.choice(list(dynamic_ngram_model.model.keys()))
        sentence = list(start_words)

        while len(sentence) < max_length:
            next_word_probs = dynamic_ngram_model.predict_next_word(sentence)
            if next_word_probs:
                filtered_probs = {word: prob for word, prob in next_word_probs.items() if prob > prob_threshold}
                if filtered_probs:
                    next_word = random.choices(list(filtered_probs.keys()), weights=list(filtered_probs.values()))[0]
                    if next_word:
                        sentence.append(next_word)
                    else:
                        break  # Stop if next word is None
                else:
                    break  # Stop if no words meet the probability threshold
            else:
                break  # Stop if no prediction for the next word

        # End the sentence with punctuation
        if len(sentence) >= min_length:
            sentence[-1] += '.'  # Add a period to the last word

        # Ensure the first word is capitalized
        if len(sentence) > 1:
            sentence[0] = sentence[0].capitalize()
    
        return ' '.join(sentence)

    def generate_text(self, initial_sentence, num_sentences):
        generated_text = initial_sentence
        current_sentence = initial_sentence

        for _ in range(num_sentences):
            new_sentence = self.generate_sentence()

            # Predict transition type
            transition_type = self.predict_transition(current_sentence, new_sentence)

            # If transition type is not suitable, select a new sentence
            while self.are_sentences_similar(new_sentence, current_sentence) or transition_type[0] == 0:
                new_sentence = self.generate_sentence()
                transition_type = self.predict_transition(current_sentence, new_sentence)

            generated_text += " " + new_sentence
            current_sentence = new_sentence

        # Ensure the generated text ends with a period
        if not generated_text.strip().endswith('.'):
            generated_text += '.'

        return generated_text.strip()

# Load text data
def load_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Example usage
if __name__ == "__main__":
    text_file_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_gen_data.txt"
    transition_model_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\best_transition_model.keras"
    
    # Load text data
    text = load_text_data(text_file_path)

    # Create the sentence generator
    sentence_generator = SentenceGenerator(text, transition_model_path)

    # Initial sentence
    initial_sentence = "Mira whispered, tears of joy glistening in her eyes."
    
    # Generate text
    generated_text = sentence_generator.generate_text(initial_sentence, num_sentences=30)
    
    print("Generated Text:")
    print(generated_text)
