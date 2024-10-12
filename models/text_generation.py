import numpy as np
import tensorflow as tf
from difflib import SequenceMatcher
import random
from collections import defaultdict
import nltk
from nltk import word_tokenize
from nltk import trigrams
from nltk import bigrams
import string
import re

class DynamicNGram:
    def __init__(self, text, n=3):
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

    def get_word_vector(self, word):
        """Placeholder for retrieving a word vector; should be replaced with actual embeddings."""
        return np.random.rand(1000)

    def is_semantically_relevant(self, last_word, next_word):
        """Check if the next word is semantically relevant to the last word."""
        last_word_vec = self.get_word_vector(last_word)
        next_word_vec = self.get_word_vector(next_word)

        # Handle cases where a word vector is not found
        if last_word_vec is None or next_word_vec is None:
            return False

        # Calculate cosine similarity with epsilon to avoid division by zero
        epsilon = 1e-8  # Small value to prevent zero division
        similarity = np.dot(last_word_vec, next_word_vec) / (np.linalg.norm(last_word_vec) * np.linalg.norm(next_word_vec) + epsilon)

        return similarity > 0.3


class SentenceGenerator:
    def __init__(self, text, transition_model_path, seq_length=20):
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

    def remove_punctuation(self, text):
        """Remove punctuation."""
        return text.translate(str.maketrans('', '', string.punctuation.replace('.', '')))

    def are_sentences_similar(self, sentence1, sentence2, threshold=0.3):
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
        # Remove punctuation and tokenize
        cleaned_sentence = self.remove_punctuation(sentence)
        return [self.tokenizer.word_index[word] for word in cleaned_sentence.split() if word in self.tokenizer.word_index]

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

        for n in range(3, 5):  # Example: checking for n-grams of length 3 to 5
            dynamic_ngram_model = DynamicNGram(' '.join(self.sentences), n=n)
            total_ngrams = sum(len(v) for v in dynamic_ngram_model.model.values())
            ngram_counts[n] = total_ngrams

        # Set min and max n-gram lengths based on frequency analysis
        min_length = min(ngram_counts, key=ngram_counts.get)
        max_length = max(ngram_counts, key=ngram_counts.get)

        return min_length, max_length

    def is_coherent(self, last_word, next_word):
        """Check if the next word maintains coherence with the last word."""
        
        # Handle 'is', 'are', 'was', 'were' - next word should be lowercase
        if last_word in ['is', 'are', 'was', 'were']:
            return next_word[0].islower()

        # Handle punctuation - next word should be uppercase after punctuation
        elif last_word[-1] in ['.', '!', '?']:
            # Ensure conjunctions don't follow punctuation unless capitalized
            if next_word.lower() in ['and', 'but', 'or']:
                return False
            # Check for valid sentence starters
            sentence_starters = ['the', 'a', 'an', 'he', 'she', 'it', 'they', 'we']
            if next_word.lower() not in sentence_starters:
                return False
            return next_word[0].isupper()
        
        # Preposition checks - ensure coherence with following word
        prepositions = ['of', 'in', 'on', 'at', 'by', 'with', 'about', 'for', 'to', 'from', 'under']
        if last_word in prepositions and next_word[0].isupper():
            return False  # Prepositions should not be followed by capitalized words
        
        # Subject-verb agreement
        subject_pronouns = ['he', 'she', 'it', 'they', 'we', 'I', 'you']
        if last_word.lower() in subject_pronouns and not next_word.endswith('ing'):
            return False  # The next word after a subject should likely be a verb
        
        # Article check
        articles = ['the', 'a', 'an']
        if last_word.lower() in articles and not next_word[0].islower():
            return False  # Articles should not be followed by capitalized words

        return True  # Default to true if no specific checks apply


    def sample_with_temperature(self, probabilities, temperature=0.3):
        """Sample from the probability distribution with temperature scaling."""
        probabilities = np.asarray(probabilities).astype(float)
        probabilities = np.exp(probabilities / temperature)  # Apply temperature
        probabilities /= np.sum(probabilities)  # Normalize
        return np.random.choice(len(probabilities), p=probabilities)

    def generate_sentence(self, min_length=2, max_length=20, prob_threshold=0.0001, temperature=0.3):
        """Generate a sentence using a dynamic n-gram model with robust word selection."""
        # Create dynamic n-gram model with varying n-gram lengths
        ngram_lengths = list(range(self.min_ngram_length, self.max_ngram_length + 1))
        length_probs = {length: 0 for length in ngram_lengths}

        # Evaluate the probability for each n-gram length
        for n in ngram_lengths:
            dynamic_ngram_model = DynamicNGram(' '.join(self.sentences), n=n)
            start_word_probs = dynamic_ngram_model.predict_next_word([])  # Assume empty sentence for start words
            length_probs[n] = sum(start_word_probs.values())  # Sum probabilities to evaluate strength of n

        # Select the n-gram length with the highest probability
        ngram_length = max(length_probs, key=length_probs.get)

        # Create dynamic n-gram model with the chosen length
        dynamic_ngram_model = DynamicNGram(' '.join(self.sentences), n=ngram_length)

        # Select start words from the model
        start_word_probs = dynamic_ngram_model.predict_next_word([])  # Assume empty sentence for start words
        if start_word_probs:
            # Sort words by probability in descending order and select the top one
            start_words = max(start_word_probs, key=start_word_probs.get)
        else:
            start_words = random.choice(list(dynamic_ngram_model.model.keys()))  # Fallback if no probabilities

        sentence = list(start_words)

        attempts = 0
        max_attempts = 50  # Limit attempts to avoid infinite loops

        while len(sentence) < max_length and attempts < max_attempts:
            next_word_probs = dynamic_ngram_model.predict_next_word(sentence)

            if next_word_probs:
                # Filter probabilities based on the threshold
                filtered_probs = {word: prob for word, prob in next_word_probs.items() if prob > prob_threshold}
                if filtered_probs:
                    # Use temperature sampling for the next word
                    words = list(filtered_probs.keys())
                    probs = list(filtered_probs.values())
                    selected_index = self.sample_with_temperature(probs, temperature)
                    next_word = words[selected_index]

                    # Check if the chosen word maintains coherence and relevance
                    if self.is_coherent(sentence[-1], next_word) and dynamic_ngram_model.is_semantically_relevant(sentence[-1], next_word):
                        sentence.append(next_word)
                        attempts = 0  # Reset attempts if a valid word is added
                    else:
                        attempts += 1  # Increment attempts if coherence or relevance is not maintained
                else:
                    break  # Stop if no words meet the probability threshold
            else:
                break  # Stop if no prediction for the next word

        # Ensure the generated sentence meets the minimum length requirement
        if len(sentence) < min_length:
            return "Sentence is too short to meet the minimum length."  # Or handle it as you see fit

        return ' '.join(sentence)



    def generate_text(self, initial_sentence=None, num_sentences=5):
        """Generate a specified number of coherent sentences."""
        text = []
        
        if initial_sentence:
            text.append(initial_sentence)
            current_sentence = initial_sentence
        else:
            # Create a dynamic n-gram model to select the strongest starting sentence
            ngram_length = random.randint(self.min_ngram_length, self.max_ngram_length)
            dynamic_ngram_model = DynamicNGram(' '.join(self.sentences), n=ngram_length)

            # Get probabilities for starting sentences and select the one with the highest probability
            start_sentence_probs = dynamic_ngram_model.predict_next_word([])  # Pass empty list to get starting sentences
            if start_sentence_probs:
                current_sentence = max(start_sentence_probs, key=start_sentence_probs.get)
            else:
                current_sentence = random.choice(self.sentences)  # Fallback if no predictions
            text.append(current_sentence)  # Include it in the output

        for _ in range(num_sentences - 1):  # Generate the remaining sentences
            generated_sentence = self.generate_sentence()
            if generated_sentence:
                text.append(generated_sentence)
                current_sentence = generated_sentence  # Update current sentence for next generation
            else:
                break

        return ' '.join(text) if text else "No coherent text generated."




# Load text data and preprocess
def load_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    
    # Lowercase the text
    text = text.lower()
    
    # Remove non-alphanumeric characters (keeping spaces)
    text = re.sub(r'[^a-z0-9\s.]+', '', text)  # Allow dots for sentence splitting

    return text  # Return as string instead of list


# Example usage
if __name__ == "__main__":
    text_file_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_gen_data.txt"
    transition_model_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\best_transition_model.keras"
    input_text = load_text_data(text_file_path)
    
    sentence_generator = SentenceGenerator(input_text, transition_model_path)
    initial_sentence = "As the spirit began to fade, it gifted each of them a glowing seed."
    num_sentences_to_generate = 5

    generated_text = sentence_generator.generate_text(initial_sentence, num_sentences_to_generate)
    print("Generated Text:")
    print(generated_text)
