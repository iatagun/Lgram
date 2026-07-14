# CAEAS Geliştirme Planı — Validasyon-Öncelikli Yol Haritası

**Tarih:** 2026-07-14
**İlke:** Önce ölç, sonra sat. Her faz bir sonrakine geçmek için sayısal bir eşik
(gate) tanımlar; eşik tutmazsa plan orada durur ve pivot kararı verilir.
**Mevcut durum:** v2.3.1 — kod tabanı temiz (CI yeşil, 171 test), ama ölçüm
çekirdeğinin dış validasyonu sıfır.

---

## Neden bu sıra?

Ürün değerlendirmesindeki teşhis: CAEAS'ın savunulabilir kimliği "feedback tool"
değil, **denetlenebilir/tekrarlanabilir yazma değerlendirme altyapısı**. Bu iddianın
tek dayanağı ölçüm kalitesi olabilir. Ölçüm kalitesi kanıtlanmadan dashboard, SaaS,
fiyatlandırma konuşmak sıralamayı ters çevirmek olur. Bu yüzden plan dört fazlı:

```
Faz 0 (2-3 hafta)  → Ölçüm çekirdeğini sağlamlaştır (kod işi, veri gerektirmez)
Faz 1 (3-4 hafta)  → Halka açık korpuslarla dış validasyon      [GATE 1]
Faz 2 (8-12 hafta) → Kurumsal pilot + akademik yayın            [GATE 2]
Faz 3 (paralel)    → Validasyonun gösterdiği zayıf katmanları güçlendir
Faz 4 (karar)      → Ürünleşme kararı: API-first MVP veya araştırma aracı olarak kal
```

---

## FAZ 0 — Ölçüm Çekirdeğini Sağlamlaştırma (2-3 hafta, sadece kod)

Denetimde bulunan zayıflıkların ürün öncesi kapatılması. Hiçbiri dış veri
gerektirmez; hepsi mevcut repo içinde yapılır.

### 0.1 Regex sezgisellerini spaCy dependency kalıplarına taşı
`efl.py`'daki L1 transfer dedektörleri (pro-drop, SOV, gender) kelime-listesi
regex'leri. spaCy parse zaten elimizde (`en_core_web_md`) — kullanılmıyor.

- Pro-drop: cümle kökünün (`ROOT`) `nsubj`/`nsubjpass` çocuğu yoksa + emir kipi
  değilse işaretle (POS/morph ile). Soru, emir, devrik yapı false-positive'leri
  yapısal olarak elenir.
- SOV: `ROOT` fiilinin pozisyonunu bağımlılık ağacından al; "son kelime fiil listesinde
  mi" bakma.
- Gender: mevcut switch-sayacı kalsın; ek olarak coreference zincirindeki referans
  değişimini `centering_theory`'nin entity map'inden besle (aynı entity'ye önce "he"
  sonra "she" — gerçek sinyal bu).

**Çıktı:** Her dedektör için precision ölçülebilir hale gelir (0.4'te ölçeceğiz).

### 0.2 Content layer'ı ciddileştir
En zayıf katman (heuristic ya da stok LLM). İki paralel iyileştirme:

- **Prompt mühendisliği:** Mevcut prompt tek satır ("Rate this essay 0-100").
  Rubrik tanımlı, few-shot örnekli (B1/B2/C1'den birer kalibre örnek), gerekçe
  isteyen yapılandırılmış prompt'a geçir. Structured output şeması zaten var.
- **Tutarlılık testi:** Aynı essay'i aynı modelle 5 kez skorla, standart sapmayı
  raporla (`raw_details["score_stability"]`). "LLM tutarsızlığı" bizim satış
  argümanımız — kendi LLM katmanımızın tutarsızlığını ölçüp raporlamak hem dürüstlük
  hem ürün özelliği (temperature=0.1 zaten var, yeterli mi göreceğiz).

### 0.3 Kanıt çıktısını cümle seviyesine indir
Öğretmenin güveneceği şey skor değil, **gösterilebilir kanıt**. `TextAnalyzer` zaten
cümle başına transition üretiyor; `CohesionLayer` bunu özetleyip atıyor.
`CAEASReport`'a ekle: `sentence_evidence: [{sentence, transition, issue?}]` — 
"3. cümlede Rough-Shift: 'the economy' konusundan 'my family'ye geçiş bağlantısız"
formatında. Bu, Faz 2'de öğretmen arayüzünün ham maddesi.

### 0.4 Skor sürüm damgası
Her rapora `scoring_version` alanı (semver). Kalibrasyon ve validasyon sonuçları
skor fonksiyonuna bağlı; formül her değiştiğinde sürüm artmalı, eski kalibrasyonlar
geçersiz sayılmalı. (Bugünkü composite değişikliği tam da bu yüzden 2.3.0'dı.)

---

## FAZ 1 — Halka Açık Korpuslarla Dış Validasyon (3-4 hafta) → GATE 1

Kurum pilotundan **önce**, sıfır maliyetle yapılabilecek en değerli iş. İnsan
puanlı, halka açık learner korpusları zaten var:

### 1.1 ELLIPSE korpusu — birincil hedef
Feedback Prize / ELLIPSE (~6.500 ELL essay), **cohesion dahil** 6 analitik boyutta
insan puanlı. CAEAS'ın ana iddiasını doğrudan test eder:

```
lgram/ellipse_benchmark.py:
  - CAEAS cohesion_score  ↔  ELLIPSE human cohesion  (Pearson r, Spearman ρ, QWK)
  - composite_indicator   ↔  ELLIPSE overall
  - Karşılaştırma tabanı: kelime sayısı baseline'ı (r_baseline) —
    metriğimiz uzunluk proxy'sinden anlamlı derecede iyi olmak zorunda
```

### 1.2 İkincil setler
- **ASAP-AES** (Kaggle, insan puanlı): composite için QWK; literatürde bol
  karşılaştırma noktası var.
- **ICLE/TICLE** (Türkçe L1 alt-korpusu, lisans gerekir): L1 transfer
  dedektörlerinin gerçek Türk öğrenci metinlerinde tetiklenme oranı — Faz 0.1'in
  precision ölçümü.
- **GCDC tam seti** (Grammarly'den istenir, ücretsiz araştırma lisansı):
  gömülü 16 örnek yerine gerçek benchmark; enron inversiyonunun gerçek veride
  olup olmadığı görülür.

### 1.3 GATE 1 — geçiş kriteri

| Metrik | Eşik | Anlamı |
|---|---|---|
| ELLIPSE cohesion korelasyonu | r ≥ 0.5 ve r > r_baseline + 0.1 | Metrik gerçek sinyal taşıyor |
| ASAP composite QWK | ≥ 0.55 | Literatür alt bandında |
| L1 dedektör precision (TICLE örneklemi, elle 100 işaret) | ≥ 0.6 | Uyarılar güvenilir |

- **Geçerse:** Sonuçlar blog + arXiv preprint olarak yayınlanır ("CAEAS validated
  against ELLIPSE, r=..."), Faz 2 pilot görüşmelerinde bu rapor kapı açar.
- **Geçmezse:** Kurum pilotu İPTAL. Metrik iyileştirmeye dön (Faz 3'ü öne çek)
  veya kütüphaneyi "araştırma aracı" olarak konumlandırıp ürün iddiasını bırak.
  Bu da meşru bir sonuç — 3 haftada öğrenilmiş olur, 6 ayda değil.

---

## FAZ 2 — Kurumsal Pilot + Yayın (8-12 hafta) → GATE 2

GATE 1 geçildiyse: bir üniversite hazırlık programı veya dil okulu ile protokol.

### 2.1 Pilot tasarımı
- **Hedef:** 300-500 essay, **en az 2 bağımsız öğretmen puanı** (ICC için şart —
  `PopulationCalibrator.icc_multirater` zaten bunu bekliyor).
- Rubrik: mevcut 5 boyutlu EFL rubriği; öğretmenler aynı rubrikle puanlar.
- Etik kurul/veri izni: anonimleştirme `DataExporter.anonymize` ile; KVKK metni.
- Karşılaştırma kolu: aynı essay'lerin bir alt kümesini (n=100) ChatGPT/Claude'a
  da puanlatıp tutarlılık (test-retest) ve QWK karşılaştırması — "LLM'e karşı
  ölçüm tutarlılığı" iddiamızın ampirik kanıtı. Yavuz (2025) ile aynı desen.

### 2.2 Veri toplama aracı (minimal, 1-2 hafta)
Dashboard DEĞİL — pilot için yeterli en küçük şey:
- CSV/xlsx toplu yükleme → CAEAS toplu analiz → öğretmen puanlarıyla birleştirme
- `PopulationCalibrator.calibrate()` çıktısı + kalibrasyon eğrisi grafiği
- Streamlit tek dosya yeterli (`tools/pilot_app.py`); auth, deploy, DB yok.

### 2.3 Analiz ve yayın
- QWK, ICC, hata dağılımı (`within_5/10/15` zaten raporlanıyor)
- CEFR seviyesine göre kırılım; L1 transfer bulgularının niteliksel analizi
- Hedef mecra: *Assessing Writing*, *Language Testing*, ya da CALICO/EUROCALL
  bildirisi. Ortak yazar olarak pilot kurumdan bir akademisyen — hem emek
  karşılığı hem güvenilirlik.

### 2.4 GATE 2 — ürünleşme kriteri

| Metrik | Eşik |
|---|---|
| Cohesion ↔ öğretmen Organization notu | QWK ≥ 0.6 |
| Composite ↔ öğretmen toplam | QWK ≥ 0.65, ICC(insanlar arası) ≥ 0.7 raporlanmış |
| Test-retest tutarlılığı | CAEAS σ ≈ 0 (deterministik çekirdek) vs LLM σ raporu |
| Öğretmen geri bildirimi | "Kanıt çıktısı işime yaradı" ≥ %60 (anket) |

---

## FAZ 3 — Teknik Güçlendirme (Faz 2 ile paralel, validasyonun gösterdiği yere)

Öncelik sırası Faz 1-2 bulgularına göre belirlenir; aday iş listesi:

1. **Content layer fine-tune:** Pilot verisi (300+ essay + puan) LoRA eğitimi için
   yeterli başlangıç. Alternatif: Claude/GPT API'si opsiyonu (`LLM_BASE_URL` zaten
   destekliyor) — maliyet/doğruluk karşılaştırmasını pilotta ölç.
2. **Kalibrasyon motoru iyileştirme:** isotonic regression docstring'de var ama
   implementasyon lineer bin interpolasyonu; gerçek isotonic'e geçir (sklearn'süz,
   PAVA algoritması ~50 satır).
3. **CEFR tahmincisi:** Faz 2 verisiyle süpervizyonlu hale getir (şu an sezgisel).
4. **Genre farkındalığı:** Brown/Modern kalibrasyon tabloları ana kütüphanede var;
   essay grader'a bağlanmadı. Argumentative/narrative ödev tipine göre eşik seçimi.
5. **Çoklu L1:** SADECE Türkçe L1 pilotu başarılıysa; sıradaki aday pazar
   araştırmasıyla (Arapça L1 — Körfez EFL pazarı büyük ve yakın).

---

## FAZ 4 — Ürünleşme Kararı

GATE 2 geçildiyse üç seçenek, önerilen sıra:

1. **API-first (önerilen):** REST servis (`POST /analyze` → CAEASReport JSON).
   Satış hedefi EdTech platformları ve LMS eklentileri; dashboard/destek yükü yok.
   Fiyat: istek başına veya kurum lisansı. İlk müşteri adayı: pilot kurumun kendisi.
2. **Self-hosted kurum paketi:** Docker imajı + kalibrasyon CLI'ı. KVKK/GDPR
   hassas kurumlar için; "veri dışarı çıkmaz" hikayesinin doğal ürünü.
3. **Hosted SaaS dashboard:** En pahalı yol; ancak 1-2'den gelen taleple, Year 2'de.

Her durumda **açık çekirdek korunur** (kütüphane MIT kalır); para hosted servis,
kalibrasyon hizmeti ve destekten gelir. PRODUCTION_READINESS_ASSESSMENT.md'deki
Year-1 gelir projeksiyonları bu plan doğrulanana kadar geçersiz sayılmalı.

---

## Zaman ve Efor Özeti (tek geliştirici varsayımı)

```
Faz 0   2-3 hafta   kod (bağımsız çalışılabilir)
Faz 1   3-4 hafta   kod + analiz (ELLIPSE/ASAP indirilebilir; ICLE lisans yazışması paralel başlasın)
GATE 1  ─────────── GEÇEMEZSE DUR / PİVOT ───────────
Faz 2   8-12 hafta  kurum bulma (2-4 hafta) + toplama (4-6 hafta) + analiz/yazım (3-4 hafta)
Faz 3   paralel     Faz 2'nin veri toplama beklemelerinde yapılır
GATE 2  ─────────── GEÇEMEZSE araştırma aracı olarak kal ───────────
Faz 4   karar + 4-6 hafta MVP (API-first ise)

Toplam: iyimser ~4 ay, gerçekçi ~6 ay (kurum bürokrasisi dahil)
```

## İlk hafta için somut başlangıç listesi

- [ ] ELLIPSE veri setini indir, `lgram/ellipse_benchmark.py` iskeletini yaz
- [ ] Kelime-sayısı baseline korelasyonunu hesapla (yenilmesi gereken çıta)
- [ ] Pro-drop dedektörünü dependency-parse tabanlı yaz + eski/yeni precision karşılaştır
- [ ] Content prompt'unu rubrik+few-shot'a genişlet, 5-koşu stabilite testi ekle
- [ ] GCDC tam set erişim e-postasını at (bekleme süresi uzun, erken başla)
- [ ] ICLE/TICLE lisans başvurusu
```
