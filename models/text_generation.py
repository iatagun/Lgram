import numpy as np
import tensorflow as tf
from difflib import SequenceMatcher
import random
from collections import defaultdict
import nltk
from nltk import word_tokenize
from nltk import trigrams
from nltk import bigrams

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
        
        # Build the bigram and trigram models
        self._build_ngram_models(text)

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

    def generate_sentence(self, length=10, prob_threshold=0.05):
        start_words = random.choice(list(self.model_trigram.keys()))
        sentence = list(start_words)

        for _ in range(length - 2):
            next_word_probs = self.model_trigram.get((sentence[-2], sentence[-1]), {})
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
        
        # Ensure no None values in sentence
        sentence = [word for word in sentence if word is not None]

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

# Metin dosyasını yükleme
def load_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Örnek kullanım
if __name__ == "__main__":
    text_file_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_gen_data.txt"
    transition_model_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\best_transition_model.keras"
    
    # Metin verisini yükle
    text = load_text_data(text_file_path)

    # Cümle üreteciyi oluştur
    sentence_generator = SentenceGenerator(text, transition_model_path)

    # Başlangıç cümlesi
    initial_sentence = "Deep within the enchanted woods of Eldoria, a long-forgotten prophecy began to awaken, hinting at a destiny intertwined with ancient magic and untold adventures."
    
    # Metin üret
    generated_text = sentence_generator.generate_text(initial_sentence, num_sentences=30)
    
    print("Generated Text:")
    print(generated_text)
