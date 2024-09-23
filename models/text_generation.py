import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from difflib import SequenceMatcher

# Text file path
text_file_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_gen_data.txt"

# Function to load text data from a file
def load_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.readlines()

# Load the text data
text_data = load_text_data(text_file_path)
text_data = [line.strip() for line in text_data if line.strip()]
text = " ".join(text_data)

# Check if two sentences are too similar
def are_sentences_similar(sentence1, sentence2, threshold=0.8):
    similarity = SequenceMatcher(None, sentence1, sentence2).ratio()
    return similarity > threshold

# Step 1: Build a language model to generate the first sentence
class TextGeneratorModel:
    def __init__(self, text, seq_length=15):
        self.seq_length = seq_length
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts([text])
        self.total_words = len(self.tokenizer.word_index) + 1
        self.sequences = self._generate_sequences(text)
        self.model = self._build_model()

    def _generate_sequences(self, text):
        input_sequences = []
        token_list = self.tokenizer.texts_to_sequences([text])[0]
        for i in range(self.seq_length, len(token_list)):
            ngram_sequence = token_list[i - self.seq_length:i + 1]
            input_sequences.append(ngram_sequence)
        return np.array(input_sequences)

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.total_words, 100, input_length=self.seq_length),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.LSTM(100),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(self.total_words, activation='softmax')
        ])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def train(self, epochs=1000):
        input_sequences = np.array(self.sequences)
        X, y = input_sequences[:, :-1], input_sequences[:, -1]
        y = tf.keras.utils.to_categorical(y, num_classes=self.total_words)

        early_stopping = EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)

        self.model.fit(X, y, epochs=epochs, verbose=1, callbacks=[early_stopping])
        self.model.save("C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_generator_model.h5")
        print("Text generator model saved successfully.")

    def generate_text(self, seed_text, next_words=10, temperature=0.7):
        generated_text = seed_text
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([generated_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.seq_length, padding='pre')
            predictions = self.model.predict(token_list, verbose=0)[0]

            # Apply temperature
            predictions = predictions ** (1 / temperature)
            predictions = predictions / np.sum(predictions)  # Normalize

            # Sample from predictions
            predicted = np.random.choice(range(len(predictions)), p=predictions)
            output_word = self.tokenizer.index_word.get(predicted, '')

            generated_text += " " + output_word

        return generated_text.strip()


# Step 2: Use the transition_model.h5 for topic prediction
class TopicPredictor:
    def __init__(self, model_path, max_length):
        self.model = load_model(model_path)
        self.word_index = {}  # Define your word index here
        self.max_length = max_length  # Define maximum sentence length here

    def predict_transition(self, current_sentence, new_sentence):
        current_sequence = self.tokenize(current_sentence)
        new_sequence = self.tokenize(new_sentence)

        current_sequence_padded = pad_sequences([current_sequence], maxlen=self.max_length, padding='post')
        new_sequence_padded = pad_sequences([new_sequence], maxlen=self.max_length, padding='post')

        input_data = np.concatenate((current_sequence_padded, new_sequence_padded), axis=1)
        prediction = self.model.predict(input_data)

        return np.argmax(prediction, axis=1)

    def tokenize(self, sentence):
        return [self.word_index[word] for word in sentence.split() if word in self.word_index]


# Step 3: Create a combined text generator class
class AITextGenerator:
    def __init__(self, text_data, transition_model_path):
        self.text_model = TextGeneratorModel(text=" ".join(text_data))
        self.topic_predictor = TopicPredictor(transition_model_path, max_length=20)

    def generate_text_with_topic_prediction(self, initial_sentence, num_sentences=5, temperature=1.2):
        generated_text = initial_sentence
        current_sentence = initial_sentence
        previous_sentences = [initial_sentence]

        transition_type_dict = {
            'CON': 3,
            'RET': 2,
            'SSH': 2,
            'RSH': 1,
            'NTT': 0,
            'EST': 1
        }
        reverse_transition_type_dict = {v: k for k, v in transition_type_dict.items()}

        for _ in range(num_sentences):
            new_sentence = self.text_model.generate_text(current_sentence, next_words=10, temperature=temperature)

            # Benzersizlik kontrolü
            if any(are_sentences_similar(new_sentence, prev_sentence, threshold=0.85) for prev_sentence in previous_sentences):
                continue  # Çok benzer bir cümle ise tekrar dene

            # Geçiş türünü tahmin et
            transition_type = self.topic_predictor.predict_transition(current_sentence, new_sentence)
            predicted_class = np.argmax(transition_type)
            predicted_transition_label = reverse_transition_type_dict.get(predicted_class, "Unknown")

            print(f"Predicted Transition Type: {predicted_transition_label}")

            # Üretilen cümleyi güncelle
            generated_text += " " + new_sentence
            previous_sentences.append(new_sentence)
            current_sentence = new_sentence

            # Başlangıç cümlesini değiştirme stratejisi
            if len(previous_sentences) > 3:  # Önceki cümle sayısına bağlı olarak
                current_sentence = np.random.choice(previous_sentences[:-1])  # Önceki cümlelerden rastgele birini seç

        return generated_text





# Train the model and generate text
if __name__ == "__main__":
    # Train the language model
    text_gen_model = TextGeneratorModel(text)
    text_gen_model.train(epochs=1000)  # Adjust the number of epochs for training
    
    # Initialize the AI text generation model using the trained transition model
    ai_generator = AITextGenerator(text_data, transition_model_path="C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\transition_model.h5")
    
    # Generate text with the initial sentence
    initial_sentence = "Kaelan revealed that a dark force was awakening, threatening to consume Eldoria. The spirit faltered for a moment, but then its eyes glinted with malice."
    
    # Keep generating sentences, predicting the transition after each sentence
    generated_text = ai_generator.generate_text_with_topic_prediction(initial_sentence, num_sentences=5, temperature=1.2)

    print("Generated Text:")
    print(generated_text)
