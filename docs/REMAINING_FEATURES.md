# Analysis Layer — Remaining Features Plan

## Phase Map

```
P1: Benchmark Suite        [■■■■□□□□]  4 adım
P2: Sentence Transformers  [■■□□□□□□]  2 adım  
P3: Cohesion Trend         [■■■□□□□□]  3 adım
P4: Heatmap                [■■□□□□□□]  2 adım
P5: Readability            [■■□□□□□□]  2 adım
P6: Suggestions            [■■■■□□□□]  4 adım
P7: Diff Analysis          [■■■□□□□□]  3 adım
```

---

## P1: Benchmark Suite

**Amaç:** Metotların doğruluğunu standart testlerle kanıtlamak.

**Test 1 — Permutation Test (en basit, en güvenilir)**
- 5 farklı türde metin al (haber, akademik, edebi, blog, diyalog)
- Her metnin cümlelerini rastgele sırala (3 farklı permütasyon)
- Orijinal > permütasyon skoru olmalı
- Başarı = tüm orijinaller permütasyonlardan yüksek

**Test 2 — Cohesion Degradation**
- Metinden her seferinde 1 cümle çıkar
- Skorun düşüş hızını ölç
- İyi metinlerde düşüş yavaş, kötülerde hızlı olmalı

**Test 3 — Cross-Method Agreement**
- Aynı metin için 4 skoru karşılaştır:
  - Centering cohesion
  - Entity grid score
  - Lexical chain score
  - Graph density
- Beklenen: hepsi aynı yönde (yüksek/düşük) hareket etmeli

**Test 4 — GCDC-style Binary Classification**
- 10 coherent + 10 incoherent metin
- Her metot için ROC eğrisi çıkar
- AUC > 0.80 olmalı

**Implementasyon:** `lgram/benchmark.py` yeni dosya

---

## P2: Sentence Transformers

**Amaç:** `en_core_web_sm` vektörleri yerine gerçek sentence embedding'ler.

**Adım 1 — Lazy loader**
```python
# TextAnalyzer.__init__ parametresi
use_sentence_transformers: bool = False
# İlk kullanımda yükle, yoksa spaCy vektörlerine düş
```

**Adım 2 — Etkilenen metotları güncelle**
- `build_cohesion_graph()`: vektör benzerliği kısmı
- `texttile_segments()`: cümle vektörleri
- `lexical_chain_score()`: kelime benzerliği

**Implementasyon:** `lgram/analyzer.py` içinde mevcut metotlara ek

---

## P3: Cohesion Trend

**Amaç:** Metin boyunca bağdaşıklığın anlık değişimini görmek.

**Adım 1 — `cohesion_trend()` metodu**
- Kayan pencere (window=3 cümle) ile her pozisyonda cohesion hesapla
- [(window_start, window_end, score), ...] listesi döndür

**Adım 2 — Trend analizi**
- Ortalama, std sapma, min/max
- Eğim (baştan sona artıyor mu azalıyor mu)
- En zayıf pencere indeksi

**Adım 3 — ASCII trend grafiği**
```
Cohesion trend:
  1.00 |██
  0.80 |████████
  0.60 |████████████
  0.40 |████████
  0.20 |
       +--+--+--+--+--+-->
        1  2  3  4  5  6
```

---

## P4: Heatmap

**Amaç:** Hangi cümleler arası bağın zayıf olduğunu görselleştirmek.

**Adım 1 — `cohesion_heatmap()` metodu**
- N×N simetri matrisi: her (i,j) için cümleler arası benzerlik
- Benzerlik = entity overlap + vector similarity

**Adım 2 — ASCII/HTML render**
- ASCII: `▓` = yüksek, `░` = orta, ` ` = düşük
- HTML: `<table>` renkli hücreler (kırmızı=sorunlu)
- Zayıf bağlantıları `< 0.3` olanları işaretle

---

## P5: Readability Integration

**Amaç:** Bağdaşıklık + okunabilirlik = birleşik metin kalitesi.

**Adım 1 — Basit okunabilirlik metrikleri (pure Python)**
- Flesch Reading Ease: hece/cümle/kelime sayısına dayalı
- Average sentence length
- Average word length

**Adım 2 — `combined_score()` metodu**
- cohesion_score * 0.6 + readability_score * 0.4
- Tek bir "metin kalitesi" sayısı

---

## P6: Düzeltme Önerileri

**Amaç:** "Şu cümleler arası kopuk — şunu düzelt" diyebilmek.

**Adım 1 — Zayıf geçiş tespiti**
- ROUGH_SHIFT olan cümle çiftlerini bul
- Cb = None olanları işaretle (en zayıf)

**Adım 2 — Öneri üretme kuralları**
- Cb None → "Bu cümlede önceki konuyla bağlantı yok. Bir zamir veya tekrar ekle."
- ROUGH_SHIFT → "Konu değişimi çok keskin. Geçiş ifadesi kullan (however, therefore, in addition)."
- Art arda 3+ ROUGH_SHIFT → "Bu bölüm tamamen kopuk. Yeniden yazılmalı."

**Adım 3 — `suggest_improvements()` metodu**
- Her sorunlu nokta için: (indeks, sorun_tipi, öneri_metni)

**Adım 4 — Zayıf noktaları işaretle**
- `annotate_weak_points()` → metni al, `<WEAK>` etiketleriyle döndür

---

## P7: Diff Analysis

**Amaç:** Metnin iki versiyonu arasındaki bağdaşıklık farkını ölçmek.

**Adım 1 — `diff_cohesion()` metodu**
- İki metni al
- Her ikisini de analiz et
- Skor farkını, geçiş dağılım farkını, segment farkını göster

**Adım 2 — İyileşme/kötüleşme raporu**
- Hangi geçişler düzeldi, hangileri kötüleşti
- CONTINUE sayısı arttı mı, ROUGH_SHIFT azaldı mı

**Adım 3 — `improvement_score()` metodu**
- v2 skoru - v1 skoru = iyileşme miktarı
- Yüzde olarak göster

---

## Tahmini Süre

| Faz | Süre |
|---|---|
| P1: Benchmark | 45 dk |
| P2: Sentence Transformers | 20 dk |
| P3: Cohesion Trend | 25 dk |
| P4: Heatmap | 20 dk |
| P5: Readability | 15 dk |
| P6: Suggestions | 30 dk |
| P7: Diff Analysis | 20 dk |
| **Toplam** | **~3 saat** |
