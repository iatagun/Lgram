import os
import pickle
import statistics

def summarize_collocations(colloc_path, top_n=10):
    # Dosya var mı kontrolü
    if not os.path.exists(colloc_path):
        print(f"⚠️ Dosya bulunamadı: {colloc_path}")
        return

    # 1) Pickle’dan yükle
    with open(colloc_path, "rb") as f:
        collocations = pickle.load(f)

    # 2) Tüm PMI değerlerini topla
    all_pmis = [
        pmi
        for neighbors in collocations.values()
        for pmi in neighbors.values()
    ]

    # 3) Boş kontrolü
    if not all_pmis:
        print("⚠️ Collocations sözlüğü boş: PMI değeri bulunamadı.")
        return

    # 4) Temel metrikleri hesapla
    max_pmi = max(all_pmis)
    min_pmi = min(all_pmis)
    avg_pmi = statistics.mean(all_pmis)
    median_pmi = statistics.median(all_pmis)
    stdev_pmi = statistics.pstdev(all_pmis)

    # 5) En yüksek PMI’lı kelime çiftlerini listele
    top_pairs = []
    for word, nbrs in collocations.items():
        for nbr, pmi in nbrs.items():
            if pmi == max_pmi:
                top_pairs.append((word, nbr))
    top_pairs = list(dict.fromkeys(top_pairs))  # tekrarı eler

    # 6) Çıktıyı yazdır
    print(f"📈 PMI Özetleri:")
    print(f"   • Max PMI    : {max_pmi:.4f}")
    print(f"   • Min PMI    : {min_pmi:.4f}")
    print(f"   • Avg PMI    : {avg_pmi:.4f}")
    print(f"   • Median PMI : {median_pmi:.4f}")
    print(f"   • Std Dev    : {stdev_pmi:.4f}\n")

    print(f"🔝 En yüksek PMI değeri ({max_pmi:.4f}) gösteren ilk {top_n} çift:")
    for i, (w, n) in enumerate(top_pairs[:top_n], 1):
        print(f"   {i:2d}. {w} ↔ {n}")

# Kullanım örneği:
colloc_path = "C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\collocations.pkl"

summarize_collocations(colloc_path, top_n=20)
