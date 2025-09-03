import pickle
import pandas as pd
import os

# Dosya yolları\ 
model_paths = {
    "bigram_model": "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\bigram_model.pkl",
    "trigram_model": "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\trigram_model.pkl",
    "fourgram_model": "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\fourgram_model.pkl",
    "fivegram_model": "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\fivegram_model.pkl",
    "sixgram_model": "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\sixgram_model.pkl",
}
colloc_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\collocations.pkl"

# 1) N-gram model özetini oluştur
summary_data = []
for model_name, path in model_paths.items():
    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
            num_prefixes = len(model) if isinstance(model, dict) else None
            num_links = sum(len(v) for v in model.values()) if isinstance(model, dict) else None
    except Exception:
        num_prefixes, num_links = None, None
    try:
        size = os.path.getsize(path)
        readable_size = (f"{size}" + " B") if size < 1024 else (f"{size/1024:.2f} KB" if size < 1024**2 else f"{size/1024**2:.2f} MB")
    except Exception:
        readable_size = None
    summary_data.append({
        "Model": model_name,
        "# Prefixes": num_prefixes,
        "# Next-Word Links": num_links,
        "Size": readable_size
    })
ngram_df = pd.DataFrame(summary_data)
print("\n=== N-gram Model Summary ===\n")
print(ngram_df)

# 2) Collocations özetini oluştur
try:
    with open(colloc_path, 'rb') as f:
        collocs = pickle.load(f)
    num_heads = len(collocs)
    counts = [len(v) for v in collocs.values()]
    total_collocs = sum(counts)
    avg_per_head = total_collocs / num_heads if num_heads else 0
    # PMI dağılımını düzleştir
    all_pmi = [pmi for v in collocs.values() for pmi in v.values()]
    avg_pmi = sum(all_pmi)/len(all_pmi) if all_pmi else 0
    median_pmi = pd.Series(all_pmi).median() if all_pmi else 0
    coll_data = [{"# Heads": num_heads,
                  "Total Collocations": total_collocs,
                  "Avg per Head": avg_per_head,
                  "Avg PMI": avg_pmi,
                  "Median PMI": median_pmi}]
    coll_df = pd.DataFrame(coll_data)
    print("\n=== Collocations Summary ===\n")
    print(coll_df)
except Exception as e:
    print(f"Collocations yüklenirken hata oluştu: {e}")

# 3) chunk.py için parametre önerileri
print("\n=== chunk.py Parametre Önerileri ===\n")
# n parametresi
bi = ngram_df.loc[ngram_df['Model']=='bigram_model', '# Prefixes'].iloc[0]
tri = ngram_df.loc[ngram_df['Model']=='trigram_model', '# Prefixes'].iloc[0]
if all(isinstance(x, int) for x in (bi, tri)) and tri > bi:
    print("- n değeri için öneri: EnhancedLanguageModel(text, n=3) veya daha yüksek kullanabilirsiniz.")
else:
    print("- n değeri için öneri: EnhancedLanguageModel(text, n=2) (bigram) daha uygun görünüyor.")

# length parametresi
total_links = sum(v for v in ngram_df['# Next-Word Links'] if isinstance(v, int))
if total_links > 50000:
    print("- generate_sentence(): length=15-20 arasında tutabilirsiniz.")
elif total_links > 20000:
    print("- generate_sentence(): length=10-15 arası uygun olur.")
else:
    print("- generate_sentence(): length=8-12 arasında tercih edilebilir.")

# num_sentences
print("- generate_and_post_process(): num_sentences=10-20 arası güvenli bir aralıktır.")

# collocation parametreleri
if 'collocs' in locals():
    # window_size önerisi
    if avg_per_head > 5:
        print(f"- build_collocation(): window_size=5 makul (ortalama {avg_per_head:.1f} kolokasyon başına).")
    else:
        print(f"- build_collocation(): window_size=3-5 arası deneyebilirsiniz (ortalama {avg_per_head:.1f}).")
    # pmi_threshold önerisi
    suggested_threshold = median_pmi
    print(f"- build_collocation(): pmi_threshold için öneri: yaklaşık {suggested_threshold:.2f} (medyan PMI).")
