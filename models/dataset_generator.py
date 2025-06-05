from datasets import load_dataset
import nltk
from nltk.tokenize import sent_tokenize
import random

nltk.download("punkt")

# 1. Dataseti yükle ve karıştır
raw_ds = load_dataset("wikipedia", "20220301.en", split="train")
raw_ds = raw_ds.shuffle(seed=42)

# 2. Tematik keyword listesi
keywords = [
    "physics", "biology", "philosophy", "psychology", "history", "language",
    "mathematics", "technology", "art", "music", "literature", "neuroscience",
    "computer", "chemistry", "engineering"
]

# 3. Belirli başlıklara göre örnek seç
selected = []
for example in raw_ds:
    title = example.get("title", "").lower()
    if any(k in title for k in keywords):
        selected.append(example)
    if len(selected) >= 100:
        break

print(f"Toplam seçilen madde: {len(selected)}")

# 4. Cümlelere böl ve sınırlı sayıda al
sentences = []
for ex in selected:
    sents = sent_tokenize(ex["text"])
    sentences.extend(sents)

# 5. 5000 cümle ile sınırla
random.shuffle(sentences)
sentences = sentences[:5000]

# 6. Dosyaya yaz
with open("thematic_wiki.txt", "w", encoding="utf-8") as f:
    for sent in sentences:
        f.write(sent.strip() + "\n")

print("Cümleler 'thematic_wiki.txt' dosyasına yazıldı.")
