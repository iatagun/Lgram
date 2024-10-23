import random
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from collections import defaultdict, Counter
import tensorflow as tf  # TensorFlow'u içe aktarın

# NLTK'den dil modeli için gerekli kaynakları indir
nltk.download('punkt')

class SimpleLanguageModel:
    def __init__(self, text, n=2):
        self.n = n  # N-gram boyutu
        self.model = self.build_model(text)
    
    def build_model(self, text):
        model = defaultdict(lambda: defaultdict(int))
        tokens = word_tokenize(text.lower())
        n_grams = ngrams(tokens, self.n)
        
        for n_gram in n_grams:
            prefix = n_gram[:-1]  # Son kelime hariç
            next_word = n_gram[-1]
            model[prefix][next_word] += 1
        
        # Olasılıkları normalize et
        for prefix, next_words in model.items():
            total_count = sum(next_words.values())
            for word in next_words:
                next_words[word] /= total_count
        
        return model
    
    def generate_sentence(self, start_words=None, length=10):
        if start_words is None:
            start_words = random.choice(list(self.model.keys()))
        else:
            start_words = tuple(start_words)
        
        current_words = list(start_words)
        sentence = current_words.copy()

        for _ in range(length):
            prefix = tuple(current_words[-(self.n-1):])  # Son n-1 kelimeyi al
            next_words = self.model[prefix]
            
            if not next_words:
                break  # Hiçbir kelime yoksa dur

            next_word = random.choices(list(next_words.keys()), weights=next_words.values())[0]
            sentence.append(next_word)
            current_words.append(next_word)

        # Cümle yapısını düzeltin
        sentence_text = ' '.join(sentence).strip()
        sentence_text = self.clean_text(sentence_text)  # Metni temizle

        # Cümle sonuna nokta ekleyin
        if sentence_text:
            return sentence_text.capitalize() + '.'  # Cümleyi büyük harfle başlat
        return ''
    
    def clean_text(self, text):
        text = text.replace("..", ".")  # İki noktayı tek noktaya çevir
        text = text.replace(" .", ".")  # Noktaları düzelt
        text = text.replace(" ,", ",")  # Virgülleri düzelt
        text = text.replace("  ", " ")  # Fazla boşlukları kaldır
        return text

def load_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def load_transition_model(model_path):
    # Geçiş modelini yükle
    model = tf.keras.models.load_model(model_path)
    return model

def get_strongest_topic_words(model, input_text, context=None):
    words = word_tokenize(input_text.lower())
    word_frequencies = Counter(words)
    most_common_words = word_frequencies.most_common(5)  # İlk 5 kelimeyi al
    strongest_words = [word for word, freq in most_common_words]

    if context:
        strongest_words = [word for word in strongest_words if word in context]

    return random.choices(strongest_words, k=min(len(strongest_words), 2))  # 2 kelime seçiyoruz

def get_transition_words(model, text):
    words = get_strongest_topic_words(model, text)
    return random.choice(words)

# Dosyadan metni yükle
file_path = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_data.txt'
text = load_text_from_file(file_path)

# Modeli oluştur
language_model = SimpleLanguageModel(text, n=2)

# Geçiş modelini yükle
transition_model_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\best_transition_model.keras"
transition_model = load_transition_model(transition_model_path)

# Belirtilen sayıda cümle oluştur
num_sentences = 15  # Oluşturulacak cümle sayısı
generated_sentences = []

# Giriş kelimlerini belirleyin
input_words = ["i", "have"]  # Kullanılacak giriş kelimleri

for i in range(num_sentences):
    if i == 0:
        # İlk cümle için input_words'u kullan
        generated_sentence = language_model.generate_sentence(start_words=input_words, length=10)
    else:
        # Sonraki cümleler için sadece geçiş modelinden kelime al
        topic_words = [get_transition_words(transition_model, text)]
        generated_sentence = language_model.generate_sentence(start_words=topic_words, length=10)

    generated_sentences.append(generated_sentence)  # Cümleyi listeye ekle

# Cümleleri birleştir ve metin haline getir
final_text = ' '.join(generated_sentences)
print("Oluşturulan Metin:\n", final_text)
