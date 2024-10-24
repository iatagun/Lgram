from gensim.models import Word2Vec
import spacy

# SpaCy İngilizce modelini yükle
nlp = spacy.load("en_core_web_sm")

def load_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

file_path = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\models\\text_data.txt'

# Metni yükleyin
text = load_text_from_file(file_path)

# Metni cümlelere ayırın
sentences = [sentence.text.lower().split() for sentence in nlp(text).sents]

# Word2Vec modelini eğitin
word2vec_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, sg=0)

# Modeli kaydedin
word2vec_model.save("word2vec.model")
