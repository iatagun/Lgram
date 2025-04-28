import pickle
import pandas as pd
import os

# Dosya yolları
model_paths = {
    "bigram_model": "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\bigram_model.pkl",
    "trigram_model": "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\trigram_model.pkl",
    "fourgram_model": "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\fourgram_model.pkl",
    "fivegram_model": "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\fivegram_model.pkl",
    "sixgram_model": "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\sixgram_model.pkl",
}

# Parametre ve boyut bilgilerini toplayalım
summary_data = []

for model_name, path in model_paths.items():
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
            if isinstance(model, dict):
                num_prefixes = len(model)
                num_total_next_words = sum(len(next_words) for next_words in model.values())
            else:
                num_prefixes = "Not dict"
                num_total_next_words = "Not dict"
    except Exception as e:
        num_prefixes = "Error"
        num_total_next_words = "Error"

    try:
        size_bytes = os.path.getsize(path)
        if size_bytes < 1024:
            readable_size = f"{size_bytes} B"
        elif size_bytes < 1024**2:
            readable_size = f"{size_bytes/1024:.2f} KB"
        else:
            readable_size = f"{size_bytes/1024**2:.2f} MB"
    except Exception as e:
        readable_size = "Error"

    summary_data.append({
        "Model Name": model_name,
        "Number of Prefixes": num_prefixes,
        "Total Next Word Links": num_total_next_words,
        "File Size": readable_size
    })

# Verileri bir DataFrame'de toplayalım
summary_df = pd.DataFrame(summary_data)
print("\n=== N-gram Model Summary ===\n")
print(summary_df)

# chunk.py parametre önerileri
print("\n=== chunk.py Parametre Önerileri ===\n")

# n parametresi önerisi:
bigram_prefixes = summary_df.loc[summary_df['Model Name'] == 'bigram_model', 'Number of Prefixes'].values[0]
trigram_prefixes = summary_df.loc[summary_df['Model Name'] == 'trigram_model', 'Number of Prefixes'].values[0]

if trigram_prefixes != "Not dict" and trigram_prefixes != "Error" and trigram_prefixes > bigram_prefixes:
    print("Öneri: chunk.py içinde EnhancedLanguageModel oluştururken n=3 veya daha yüksek kullanabilirsin.")
else:
    print("Öneri: chunk.py içinde EnhancedLanguageModel oluştururken n=2 (bigram) daha mantıklı olabilir.")

# length parametresi önerisi
total_links = summary_df['Total Next Word Links'].apply(lambda x: x if isinstance(x, int) else 0).sum()

if total_links > 50000:
    print("Öneri: generate_sentence fonksiyonunda length=15-20 gibi uzun cümleler üretilebilir.")
elif total_links > 20000:
    print("Öneri: generate_sentence fonksiyonunda length=10-15 arasında tutabilirsin.")
else:
    print("Öneri: generate_sentence fonksiyonunda length=8-12 gibi daha kısa cümleler üretmek daha iyi olabilir.")

# num_sentences parametresi önerisi
print("Öneri: generate_and_post_process fonksiyonunda num_sentences=10-20 arası güvenli bir değer olur.")
