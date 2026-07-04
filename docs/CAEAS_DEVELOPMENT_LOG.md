# CAEAS — Geliştirme Günlüğü ve Kalibrasyon Araştırma Protokolü

**Tarih:** 4 Temmuz 2025
**Branch:** `product/essay-grader`
**Versiyon:** v0.2.0
**Kapsam:** EFL (Yabancı Dil Olarak İngilizce) yazma ödevleri — geri bildirim aracı
**Hedef L1:** Türkçe
**Hedef CEFR:** B1, B2, C1

---

# Bölüm 1: Bugün İnşa Edilenler

## v0.1 — Temel Mimari (CAEASGrader)

5 katmanlı kanıt-tabanlı essay analiz sistemi:

| Katman | Modül | İşlev |
|--------|-------|-------|
| L1 | `layer1_content.py` | İçerik analizi (MockContentAnalyzer + pluggable LLM arayüzü) |
| L2 | `layer2_cohesion.py` | Segment-duyarlı Lgram kohezyon analizi (intro/body/conclusion) |
| L3 | `layer3_surface.py` | Yüzey kalitesi (Flesch, TTR, cümle çeşitliliği) |
| L4 | `layer4_calibration.py` | Popülasyon kalibrasyonu (QWK, ICC, isotonic regression) |
| L5 | `layer5_confidence.py` | Güven aralığı, sınır tespiti, öğretmen inceleme tetikleyicileri |

## v0.1 — EFL Kapsamına Daraltma

| Eklenen | Modül |
|---------|-------|
| 5 boyutlu EFL rubriği (Grammar, Content, Organization, Style, Mechanics) | `efl.py` |
| CEFR profil veritabanı (B1, B2, C1) | `efl.py` |
| L1 transfer analizi (Türkçe pro-drop, gender, article, SOV) | `efl.py` |
| CEFR seviye tahmini (heuristic) | `efl.py` |
| EFL modu varsayılan, CEFR-aware verdict'ler | `grader.py` |

## v0.2 — Production Hardening (6 risk mitigasyonu)

### 1. PreFilter — Grammar/Cohesion Ayırıcı Katman
**Dosya:** `prefilter.py`
**Risk:** spaCy, hatalı EFL metninde kırılgan → yanlış Cb/Cf → sahte kohezyon hataları
**Çözüm:** Centering analizinden ÖNCE çalışan grammar check katmanı
- LanguageTool entegrasyonu (opsiyonel, `pip install language-tool-python`)
- Heuristic fallback: subject-drop, article eksikliği, SOV transfer tespiti
- Hataları GRAMMAR vs POSSIBLE_COHESION olarak etiketler
- Grammar hatası yüksekse parse confidence'ı düşürür
- "Bu bir kohezyon hatası mı yoksa dilbilgisi hatası mı?" sorusunu yanıtlar

### 2. CEFR Seviye-Bazlı Kalibrasyon Pipeline'ı
**Dosya:** `cefr_calibration.py`
**Risk:** Tek eşik tüm seviyeler için → "daha iyi yazan öğrenci daha düşük kohezyon skoru alır" paradoksu
**Çözüm:**
- `CEFRCalibrator`: seviye başına ayrı QWK/ICC/recalibration
- `ComplexityProfile`: sözdizimsel karmaşıklık ölçümü (cümle uzunluğu, clause oranı, kelime çeşitliliği)
- Karmaşıklık-ayarlı skorlama: karmaşık metinlere tolerans bonusu
- `LevelCalibration` dataclass: seviye-bazlı eşikler ve yeniden kalibrasyon eğrileri

### 3. Terminoloji Denetimi
**Etkilenen dosyalar:** 7 dosya
**Değişiklikler:**
- `coherence` → `cohesion` (her yerde)
- `score` → `indicator` (kullanıcıya dönük metinlerde)
- `verdict` → `suggestion`
- `judge` → `analyzer`
- `assessment system` → `feedback tool`
- `EXCELLENT/GOOD/ADEQUATE` → betimleyici geri bildirim

### 4. Feedback-Mode Konumlandırması
**Risk:** Sistem "not verici" olarak algılanırsa etik/hukuki sorumluluk doğar
**Çözüm:**
- `grader.grade()` → `grader.analyze()` (eski isim backward-compat)
- `CAEASReport.verdict` → `CAEASReport.suggestion`
- `CAEASReport.overall_score` → `CAEASReport.overall_cohesion_indicator`
- Tüm çıktılarda "Bu otomatik bir geri bildirimdir, öğretmen kararı nihaidir" notu
- `human_review_recommended` → `teacher_review_recommended`
- `ContentJudge` → `ContentAnalyzer`

### 5. Veri Toplama / Export Modülü
**Dosya:** `export.py`
- `DataExporter`: araştırma-kalitesinde yapılandırılmış JSON/CSV export
- `anonymize()`: öğrenci bilgilerini temizleme
- `compare_institutions()`: kurumlar arası karşılaştırma
- ICLE/TOEFL11 formatlarıyla uyumlu

### 6. Hata Tipolojisi Çerçevesi
**Dosya:** `typology.py`
- 8 hata kategorisi (PRONOUN_REFERENCE, MISSING_TRANSITION, ABRUPT_TOPIC_SHIFT, OVERUSE_REPETITION, GENDER_MISMATCH, ARTICLE_COHESION, SUBJECT_DROP, SOV_TRANSFER)
- Her kategori için: sıklık, şiddet, CEFR dağılımı, L1 korelasyonu, öğretmen rehberi
- `ErrorTypology.build()` → yayınlanabilir tipoloji raporu
- Bağımsız araştırma katkısı olarak kullanılabilir

---

# Bölüm 2: Mevcut Dosya Yapısı

```
lgram/essay/
├── __init__.py              — Paket export'ları (tüm modüller)
├── models.py                — Essay, CAEASReport, ContentAnalyzer, LayerResult
├── layer1_content.py        — İçerik analizi (MockContentAnalyzer)
├── layer2_cohesion.py       — Segment-duyarlı kohezyon (Lgram)
├── layer3_surface.py        — Yüzey kalitesi (Flesch, TTR, vb.)
├── layer4_calibration.py    — Popülasyon kalibrasyonu (QWK, ICC)
├── layer5_confidence.py     — Güven aralığı + öğretmen tetikleyicileri
├── efl.py                   — EFL modülü (CEFR profilleri, L1 transfer)
├── prefilter.py             — Grammar/cohesion ayırıcı (v0.2)
├── cefr_calibration.py      — CEFR seviye kalibrasyonu (v0.2)
├── export.py                — Veri export modülü (v0.2)
├── typology.py              — Hata tipolojisi (v0.2)
└── grader.py                — CAEASGrader ana sınıfı

tests/
├── test_essay.py            — 34 test (temel CAEAS)
├── test_efl.py              — 26 test (EFL özel)
└── test_lgram.py, ...       — 99 test (Lgram core)

docs/
├── CAEAS_V02_PLAN.md        — v0.2 implementasyon planı
├── RESEARCH.md              — Literatür taraması
└── ...
```

---

# Bölüm 3: Kalibrasyon Araştırma Protokolü

## Amaç

CAEAS sisteminin gerçek EFL öğrenci yazıları üzerinde, gerçek öğretmen değerlendirmeleriyle karşılaştırmalı olarak doğrulanması ve kalibrasyonu.

## Kapsam ve Sınırlamalar

**Ne ölçüyoruz:** Organization/Cohesion alt-boyutu (EFL rubriğinin 5 boyutundan biri).
**Ne ölçmüyoruz:** Grammar, Content, Style, Mechanics — bunlar sistemin kapsamı dışında.
**Konumlandırma:** Bu bir "not verme sistemi" değil, "geri bildirim aracı"dır. Nihai not her zaman öğretmen tarafından verilir.

## Adım 1: Kurumsal İzin ve Veri Gizliliği

| Gereklilik | Detay |
|------------|-------|
| Kurum onayı | Okul yönetiminden yazılı izin |
| KVKK uyumu | Öğrenci verileri (ödev + not) kişisel veri kapsamında |
| Anonimleştirme | İsim, numara, tanımlayıcı bilgiler toplanmaz / toplananlar silinir |
| Veli/öğrenci onayı | Mümkünse bilgilendirilmiş onam (özellikle yayın planlanıyorsa) |
| Veri saklama | Sadece anonimleştirilmiş metin + sayısal not saklanır |

## Adım 2: Veri Toplama

### Minimum Örneklem

| Amaç | Minimum n | İdeal n |
|------|-----------|---------|
| Pilot / fizibilite | 30 | 50 |
| Kalibrasyon (istatistiksel anlamlı) | 50 | 100 |
| Yayın kalitesinde | 100 | 200+ |

### Veri Formatı

Her ödev için toplanacaklar:

```
{
  "essay_id": "anonim_id",
  "text": "öğrencinin yazdığı tam metin",
  "cefr_level": "B1 | B2 | C1",
  "grade_level": "9 | 10 | 11 | 12",
  "teacher_scores": {
    "grammar": 75,
    "content": 80,
    "organization": 70,    // ← kritik: Lgram'ın karşılaştırılacağı boyut
    "style": 72,
    "mechanics": 85,
    "overall": 76
  }
}
```

**Kritik:** Öğretmenden mutlaka **organization/cohesion alt-notu** ayrı istenmeli. Sadece genel not ("76/100") alınırsa, grammar+content+mechanics karıştığı için cohesion skoruyla karşılaştırma anlamsız olur — elma ile armut karşılaştırması.

## Adım 3: Inter-Rater Reliability (Öğretmenler Arası Uyum)

**Neden:** Tek öğretmenin verdiği not, "doğru not" değil "bir kişinin öznel görüşü"dür. Öğretmenler arası uyumu bilmeden "sistemim öğretmenle %X uyumlu" demek anlamsız.

**Yöntem:**
- Ödevlerin bir alt-kümesi (en az 30, ideal 40-50) **2 farklı öğretmene bağımsız** verilir
- Öğretmenler birbirinin notunu görmez
- ICC(2,k) hesaplanır → bu "insan tavanı"dır (human ceiling)
- Beklenti: Sistem bu tavanı aşamaz, ona yaklaşması başarıdır

**Yorumlama:**
```
ICC > 0.90  →  Mükemmel öğretmen uyumu (nadir)
ICC > 0.80  →  İyi uyum (kabul edilebilir)
ICC > 0.70  →  Orta uyum
ICC < 0.70  →  Zayıf uyum — referans notun kendisi güvenilir değil
```

## Adım 4: Kör (Blind) Karşılaştırma

**Neden:** Veri kirlenmesini önlemek. Öğretmen sistem çıktısını görürse notu etkilenir; sen öğretmen notunu görüp sistemi ayarlarsan overfitting olur.

**Protokol:**
1. Öğretmen(ler) ödevleri sistem sonucunu GÖRMEDEN notlar
2. Sen öğretmen notlarını GÖRMEDEN sistemi çalıştırırsın
3. Karşılaştırma en son yapılır
4. Kalibrasyon aşamasında (Adım 6) veri kullanılır, ama bu aşamada HENÜZ kullanılmaz

## Adım 5: CEFR / Seviye Stratifikasyonu

**Neden:** Farklı seviyelerdeki öğrencilerin kohezyon profilleri farklıdır. Tüm seviyeleri karıştırıp tek bir ortalama vermek, seviye-bazlı performans farklarını gizler.

**Yöntem:**
- Veri CEFR seviyesine göre gruplanır (B1, B2, C1)
- Her seviye için ayrı QWK/ICC hesaplanır
- "Sistem B1'de iyi, C1'de yetersiz" gibi bir gerçek varsa, bu bilinmeli ve raporlanmalı
- Seviye başına minimum 15-20 ödev (tercihen 30+) gerekir

## Adım 6: Kalibrasyon vs Doğrulama Ayrımı

**Neden:** Aynı veriyle hem ayarlayıp hem test edersen, sonuç yapay olarak iyi çıkar (overfitting). Gerçek dünya performansını yansıtmaz.

**Train/Test Split:**
- **Kalibrasyon seti (%70):** Sistem eşikleri ve ağırlıkları bu veriyle ayarlanır
- **Doğrulama seti (%30, held-out):** Bu veriye HİÇ dokunulmaz, sadece en son test için kullanılır
- Bölme rastgele yapılır, seviye dağılımı korunur (stratified split)

**Overfitting kontrolü:**
- Kalibrasyon setindeki QWK ile doğrulama setindeki QWK arasındaki fark
- Fark > 0.10 ise overfitting var demektir → model sadeleştirilmeli

## Adım 7: İstatistiksel Analiz

### Metrikler

| Metrik | Formül | Yorum |
|--------|--------|-------|
| **QWK** (Quadratic Weighted Kappa) | 1 - O/E | 0 = şans, 1 = mükemmel uyum |
| **ICC(2,k)** | (MSB-MSW)/(MSB+(k-1)MSW) | Öğretmenler arası güvenilirlik |
| **MAE** (Mean Absolute Error) | mean(\|sistem - öğretmen\|) | Ortalama sapma (puan cinsinden) |
| **Within-10** | %(\|sistem - öğretmen\| <= 10) | 10 puan içinde tutturma oranı |
| **Pearson r** | kovaryans / (std_s * std_t) | Doğrusal korelasyon |
| **Calibration curve** | sistem vs öğretmen scatter + loess | Sistematik sapma var mı? |

### Raporlanacak Bulgular

1. **İnsan tavanı:** Öğretmenler arası ICC = X
2. **Sistem uyumu:** Sistem vs öğretmen QWK = Y
3. **Göreceli başarı:** Sistem, insan tavanının %Z'sine ulaştı (Y / X)
4. **Seviye bazlı:** B1'de QWK=X1, B2'de QWK=X2, C1'de QWK=X3
5. **Kalibrasyon sonrası:** Recalibration sonrası doğrulama setinde QWK = X'

**Örnek rapor cümlesi:**
> "Öğretmenler arası uyum ICC = 0.82 olarak hesaplanmıştır. Sistem-öğretmen uyumu QWK = 0.68'dir. Bu, sistemin insan puanlayıcılar arası uyumun %83'üne ulaştığı anlamına gelir. B2 seviyesinde performans (QWK=0.74) B1 seviyesinden (QWK=0.61) daha yüksektir."

## Adım 8: Sonuçların Yayınlanması

### Savunulabilir İddialar (✓)
- "Sistem, organization/cohesion alt-boyutunda öğretmen puanlarıyla orta-yüksek uyum göstermektedir (QWK = X)"
- "Sistem, insan puanlayıcılar arası uyumun %Z'sine ulaşmıştır"
- "B2 seviyesinde performans B1'den anlamlı derecede yüksektir"
- "Türk EFL öğrencilerinde en sık görülen kohezyon hataları: [tipoloji sonuçları]"

### Kaçınılması Gereken İddialar (✗)
- "Sistem not vermede insan kadar iyi" → Hayır, sadece organization alt-boyutunda
- "Sistem %X doğru" → "Doğru" diye bir şey yok, referans not zaten öznel
- "Tüm seviyelerde aynı performans" → Muhtemelen değil, raporla
- "Sistem tek başına essay değerlendirebilir" → Hayır, geri bildirim aracı, not verici değil

---

# Bölüm 4: Beştepe İçin Özet Protokol

| Adım | Eylem | Süre |
|------|-------|------|
| 1 | Okul yönetiminden yazılı izin al | 1-2 hafta |
| 2 | KVKK protokolü belirle, anonimleştirme yöntemini netleştir | 1 hafta |
| 3 | 50-100 ödev + 5-boyutlu rubrik notu topla | 2-4 hafta |
| 4 | Alt-kümede (30-40) 2. öğretmenden bağımsız not al | Paralel |
| 5 | Veriyi CEFR/seviye bazında ayır | 1 gün |
| 6 | Train/test split (%70/%30) | 1 gün |
| 7 | Kalibrasyon setiyle eşikleri ayarla | 1-2 gün |
| 8 | Doğrulama setiyle QWK/ICC hesapla | 1 gün |
| 9 | Sonuçları raporla — insan tavanına göreceli başarı | 1 hafta |

**Toplam tahmini süre:** 6-10 hafta (bürokratik süreçler dahil)

---

# Bölüm 5: Teknik Referans

## Backward-Compatible API

v0.2'de kaldırılan isimler için alias'lar mevcut:

```python
# Yeni isim (tercih edilen)
grader = CAEASGrader()
report = grader.analyze(essay)
print(report.suggestion)
print(report.overall_cohesion_indicator)

# Eski isim (çalışır ama deprecated)
report = grader.grade(essay)
print(report.verdict)
print(report.overall_score)
```

## Hızlı Başlangıç

```python
from lgram.essay import CAEASGrader, Essay

# Varsayılan: EFL modu, CEFR otomatik
grader = CAEASGrader()

# Opsiyonel: PreFilter + L1 analizi + belirli CEFR
grader = CAEASGrader(
    cefr_level="B2",
    l1_language="tr",
    use_prefilter=True,      # LanguageTool opsiyonel
)

essay = Essay(
    title="Social Media Effects",
    text="Social media has changed how people communicate...",
)

report = grader.analyze(essay)

print(f"Kohezyon göstergesi: {report.overall_cohesion_indicator:.0f}/100")
print(f"Güven aralığı: {report.confidence_interval}")
print(f"Öneri: {report.suggestion}")
print(f"Öğretmen incelemesi önerilir: {report.teacher_review_recommended}")

# Kalibrasyon için export
from lgram.essay import DataExporter
exporter = DataExporter()
bundle = exporter.create_bundle([essay], [report])
exporter.to_json(bundle, "calibration_data.json")

# Hata tipolojisi
from lgram.essay import ErrorTypology
typology = ErrorTypology()
typology.feed(report, cefr_level="B2")
typo_report = typology.build()
print(typo_report.summary)
```

## Test Durumu

```
159 tests passed, 0 failures
├── 99  Lgram core tests
├── 34  CAEAS base tests
└── 26  EFL-specific tests
```

---

# Bölüm 6: Sınırlamalar ve Bilinen Riskler

| Risk | Durum | Çözüm |
|------|-------|-------|
| spaCy hatalı EFL'de kırılgan | Azaltıldı | PreFilter grammar/cohesion ayırımı, parse confidence downgrade |
| CEFR profilleri heuristic | Veri bekliyor | Gerçek öğrenci verisiyle CEFRCalibrator devreye girecek |
| Tek öğretmen notu yetersiz | Protokolde | Adım 3: inter-rater reliability (2+ öğretmen) |
| Overfitting riski | Protokolde | Adım 6: train/test split (%70/%30) |
| KVKK/etik risk | Protokolde | Adım 1: kurumsal izin + anonimleştirme |
| Sadece organization ölçüyor | Bilinçli seçim | Diğer 4 boyut ölçülmüyor, bu açıkça belirtiliyor |
| MockContentAnalyzer zayıf | Bilinçli seçim | Gerçek LLM analyzer için pluggable arayüz hazır |
| LanguageTool opsiyonel | Bilinçli seçim | `pip install language-tool-python` ile aktifleşir |
