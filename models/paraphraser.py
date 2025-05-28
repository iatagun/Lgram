import torch
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import re

class WordEmbeddingModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embed_dim)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, inputs):
        return self.dropout(self.embeddings(inputs))

# Ayarlar
embedding_dim = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# MODELLERİ YÜKLE
with open("C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\word_to_idx.pkl", "rb") as f:
    word_to_idx = pickle.load(f)
with open("C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\idx_to_word.pkl", "rb") as f:
    idx_to_word = pickle.load(f)

model = WordEmbeddingModel(len(word_to_idx), embedding_dim).to(device)
model.load_state_dict(torch.load("C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\word_embedding_model.pt", map_location=device))
model.eval()

# EMBEDDING MATRİSİ
with torch.no_grad():
    embeddings = model.embeddings.weight.cpu().numpy()
similarity_matrix = cosine_similarity(embeddings)

# EN BENZER KELİMEYİ BUL
def get_similar_word(word, topn=3):
    word = word.lower()
    if word not in word_to_idx:
        return word
    idx = word_to_idx[word]
    sim_scores = similarity_matrix[idx]
    top_indices = np.argsort(sim_scores)[::-1][1:topn+1]
    for new_idx in top_indices:
        new_word = idx_to_word[new_idx]
        if new_word != word:
            return new_word
    return word

# PARAFRAZLAYICI
def paraphrase_sentence(sentence):
    tokens = word_tokenize(sentence)
    new_tokens = []
    for token in tokens:
        clean_token = re.sub(r'\W+', '', token)
        punctuation = token[len(clean_token):]
        new_token = get_similar_word(clean_token)
        new_tokens.append(new_token + punctuation)
    return " ".join(new_tokens)

# ÖRNEK KULLANIM
if __name__ == "__main__":
    original = "The study of language is fascinating and complex."
    paraphrased = paraphrase_sentence(original)
    print("Original:   ", original)
    print("Paraphrased:", paraphrased)
