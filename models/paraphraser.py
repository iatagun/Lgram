import nltk
import torch
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
import re
import string

class WordEmbeddingModel(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = torch.nn.Embedding(vocab_size, embed_dim)
        self.dropout = torch.nn.Dropout(0.3)

    def forward(self, inputs):
        return self.dropout(self.embeddings(inputs))

# Ayarlar
embedding_dim = 500
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
def get_similar_word_contextual(word, context_tokens, pos=None):
    """
    Girdi kelimeye semantik olarak benzer ve bağlamsal olarak uyumlu
    bir alternatif döndür. İstersen cosine similarity + POS filtrelemesi ekleyebilirsin.
    """
    if word not in word_to_idx:
        return None
    
    word_vec = embeddings[word_to_idx[word]]
    context_vec = sum([embeddings[word_to_idx[w]] for w in context_tokens if w in word_to_idx])
    combined_vec = (word_vec + context_vec) / 2

    similarities = cosine_similarity([combined_vec], embeddings)[0]
    top_indices = similarities.argsort()[::-1]

    for i in top_indices[1:10]:  # İlk sırada kendisi olabilir
        candidate = idx_to_word[i]
        if candidate == word:
            continue
        if pos and nltk.pos_tag([candidate])[0][1] != pos:
            continue
        return candidate
    return None

# PARAFRAZLAYICI
def paraphrase_sentence(sentence):
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)  # bağlam için POS bilgisi
    new_tokens = []

    for idx, (token, pos) in enumerate(pos_tags):
        if token in string.punctuation:
            new_tokens.append(token)
            continue

        match = re.match(r"^([\w\-']+)(\W*)$", token)
        if match:
            word, punctuation = match.groups()
        else:
            word, punctuation = token, ""

        is_capitalized = word[0].isupper()

        if len(word) < 3 or word.lower() in {"the", "and", "is", "are"}:
            new_token = word
        else:
            context = tokens[max(0, idx - 3):idx] + tokens[idx + 1:idx + 4]
            replacement = get_similar_word_contextual(word.lower(), context, pos) or word
            new_token = replacement.capitalize() if is_capitalized else replacement

        new_tokens.append(new_token + punctuation)

    return " ".join(new_tokens)

# ÖRNEK KULLANIM
if __name__ == "__main__":
    original = "He could see in preparation and cut he does not a red and think."
    paraphrased = paraphrase_sentence(original)
    print("Original:   ", original)
    print("Paraphrased:", paraphrased)
