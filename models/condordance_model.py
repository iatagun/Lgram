import nltk
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from collections import defaultdict, Counter
from nltk.corpus import stopwords
import pickle
import random

nltk.download('punkt')
nltk.download('stopwords')

# Parametreler
window_size = 5
embedding_dim = 1000
chunk_size = 200_000
file_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\more.txt"

# Token ve temizlik
def preprocess_tokens(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalnum() and t not in stop_words]
    counter = Counter(tokens)
    return [t for t in tokens if counter[t] > 1]

# Chunk fonksiyonu
def read_in_chunks(file_path, chunk_size=200_000):
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

# Embedding modeli
class WordEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        return self.dropout(self.embeddings(inputs))

# Loader fonksiyonu
def get_loader(pairs, batch_size=1024):
    x = torch.tensor([p[0] for p in pairs], dtype=torch.long)
    y = torch.tensor([p[1] for p in pairs], dtype=torch.long)
    labels = torch.tensor([p[2] for p in pairs], dtype=torch.float)
    dataset = torch.utils.data.TensorDataset(x, y, labels)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Vocab oluştur
all_tokens = []
for chunk in read_in_chunks(file_path):
    all_tokens.extend(preprocess_tokens(chunk))
unique_words = list(set(all_tokens))
word_to_idx = {word: idx for idx, word in enumerate(unique_words)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# Model ve optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = WordEmbeddingModel(len(unique_words), embedding_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
loss_fn = nn.CosineEmbeddingLoss()

# Chunk başına eğitim
for chunk in read_in_chunks(file_path, chunk_size=chunk_size):
    tokens = preprocess_tokens(chunk)
    concordance_dict = defaultdict(list)
    for i in range(len(tokens)):
        word = tokens[i]
        left = tokens[max(i - window_size, 0):i]
        right = tokens[i+1:i+1+window_size]
        context = left + right
        concordance_dict[word].append(context)

    current_words = list(set(tokens))
    pairs = []
    for word, contexts in concordance_dict.items():
        for context in contexts:
            for ctx_word in context:
                if word in word_to_idx and ctx_word in word_to_idx:
                    pairs.append((word_to_idx[word], word_to_idx[ctx_word], 1.0))
                    neg = random.choice(current_words)
                    if neg != ctx_word and neg in word_to_idx:
                        pairs.append((word_to_idx[word], word_to_idx[neg], -1.0))

    if not pairs:
        continue

    train_pairs, val_pairs = train_test_split(pairs, test_size=0.1, random_state=42)
    train_loader = get_loader(train_pairs)
    val_loader = get_loader(val_pairs)

    for epoch in range(20):
        model.train()
        total_loss = 0.0
        for bx, by, bl in train_loader:
            bx, by, bl = bx.to(device), by.to(device), bl.to(device)
            optimizer.zero_grad()
            ex, ey = model(bx), model(by)
            loss = loss_fn(ex, ey, bl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for vx, vy, vl in val_loader:
                vx, vy, vl = vx.to(device), vy.to(device), vl.to(device)
                ex, ey = model(vx), model(vy)
                loss = loss_fn(ex, ey, vl)
                val_loss += loss.item()

        print(f"Chunk Epoch {epoch:02d} — Train: {total_loss/len(train_loader):.4f} — Val: {val_loss/len(val_loader):.4f}")

# Model ve sözlükleri kaydet
torch.save(model.state_dict(), "word_embedding_model.pt")
with open("word_to_idx.pkl", "wb") as f:
    pickle.dump(word_to_idx, f)
with open("idx_to_word.pkl", "wb") as f:
    pickle.dump(idx_to_word, f)


# Anlamsal benzerliği test et (sadece ilk 10 kelime için)
with torch.no_grad():
    embeddings = model.embeddings.weight.cpu().numpy()
    similarity_matrix = cosine_similarity(embeddings)
    for i, word in enumerate(unique_words[:10]):  # Sadece ilk 10 kelimeyi göster
        top_idx = np.argsort(similarity_matrix[i])[::-1][1:4]  # En benzer 3 kelime
        print(f"{word}: {[unique_words[j] for j in top_idx]}")
