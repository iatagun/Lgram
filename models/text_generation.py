import numpy as np
import tensorflow as tf
from difflib import SequenceMatcher
import random

class SentenceGenerator:
    def __init__(self, text, transition_model_path, seq_length=15):
        self.sentences = self._split_into_sentences(text)
        self.transition_model = tf.keras.models.load_model(transition_model_path)
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts(self.sentences)
        self.seq_length = seq_length

    def _split_into_sentences(self, text):
        # Metni cümlelere ayırır
        return [sentence.strip() for sentence in text.split('.') if sentence]

    def are_sentences_similar(self, sentence1, sentence2, threshold=0.8):
        # İki cümlenin benzerliğini kontrol eder
        similarity = SequenceMatcher(None, sentence1, sentence2).ratio()
        return similarity > threshold

    def predict_transition(self, current_sentence, new_sentence):
        # Cümlelerin geçiş türünü tahmin eder
        current_sequence = self.tokenize(current_sentence)
        new_sequence = self.tokenize(new_sentence)

        current_sequence_padded = tf.keras.preprocessing.sequence.pad_sequences([current_sequence], maxlen=self.seq_length, padding='post')
        new_sequence_padded = tf.keras.preprocessing.sequence.pad_sequences([new_sequence], maxlen=self.seq_length, padding='post')

        input_data = np.concatenate((current_sequence_padded, new_sequence_padded), axis=1)
        prediction = self.transition_model.predict(input_data)

        return np.argmax(prediction, axis=1)

    def tokenize(self, sentence):
        # Cümleyi tokenize eder
        return [self.tokenizer.word_index[word] for word in sentence.split() if word in self.tokenizer.word_index]

    def generate_sentence(self):
        # Rastgele bir cümle seçer
        return random.choice(self.sentences)

    def generate_text(self, initial_sentence, num_sentences):
        generated_text = initial_sentence
        current_sentence = initial_sentence

        for _ in range(num_sentences):
            new_sentence = self.generate_sentence()

            # Geçiş türünü tahmin et
            transition_type = self.predict_transition(current_sentence, new_sentence)

            # Eğer geçiş türü uygun değilse yeni cümle seç
            while self.are_sentences_similar(new_sentence, current_sentence) or transition_type[0] == 0:
                new_sentence = self.generate_sentence()
                transition_type = self.predict_transition(current_sentence, new_sentence)

            generated_text += " " + new_sentence
            current_sentence = new_sentence

        return generated_text.strip()

# Metin dosyasını yükleme
def load_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Örnek kullanım
if __name__ == "__main__":
    # Metin dosyasının yolunu belirtin
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
