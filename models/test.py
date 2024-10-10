import random
from collections import defaultdict
import nltk
from nltk import word_tokenize
from nltk import bigrams

class StatisticalModel:
    def __init__(self, text, n=2):
        self.n = n
        self.model = defaultdict(lambda: defaultdict(lambda: 0))
        self._build_ngram_model(text)

    def _build_ngram_model(self, text):
        # Tokenize the text
        tokenized_text = [list(map(str.lower, word_tokenize(sent))) for sent in nltk.sent_tokenize(text)]
        for sentence in tokenized_text:
            for ngram in bigrams(sentence, pad_right=True, pad_left=True):
                self.model[ngram[0]][ngram[1]] += 1

        # Transform counts to probabilities
        for w1 in self.model:
            total_count = float(sum(self.model[w1].values()))
            for w2 in self.model[w1]:
                self.model[w1][w2] /= total_count

    def generate_sentence(self, max_length=10):
        # Start with a random word
        current_word = random.choice(list(self.model.keys()))
        sentence = [current_word]

        while len(sentence) < max_length:
            next_word_probs = self.model[current_word]
            if not next_word_probs:
                break  # Stop if no next word probabilities
            
            # Randomly choose the next word based on probabilities
            next_word = random.choices(list(next_word_probs.keys()), weights=list(next_word_probs.values()), k=1)[0]
            if next_word:  # Check if next_word is valid
                sentence.append(next_word)  # Append the chosen next word
                current_word = next_word
            else:
                break  # Stop if no words are found

        # End the sentence with a period
        if sentence:  # Check if sentence is not empty
            sentence[-1] += '.'  # Add a period to the last word
        return ' '.join(sentence).capitalize() if sentence else ''

# Load text data
def load_text_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

if __name__ == "__main__":
    # Örnek metin dosyası yolu
    text_file_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_gen_data.txt"
    
    # Metin verisini yükle
    text = load_text_data(text_file_path)

    # İstatistiksel modeli oluştur
    statistical_model = StatisticalModel(text)

    # Cümleler oluştur
    for _ in range(5):  # 5 cümle üret
        print(statistical_model.generate_sentence())
