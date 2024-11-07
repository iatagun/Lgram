# TransitionAnalyzer Sınıfı

TransitionAnalyzer, bir metindeki cümleler arasındaki geçişleri analiz etmek için tasarlanmış bir sınıftır. Bu sınıf, metni cümlelere ayırarak cümle çiftleri oluşturur ve ardından her cümle çifti arasındaki isim tamlamalarını ve geçiş türlerini belirler.

### Özellikler:
* Önişleme: Metni cümlelere ayırarak cümle çiftleri oluşturur.
* İsim Tamlamaları Çıkarma: Her cümledeki isim tamlamalarını çıkarır.
* Geçiş Sınıflandırma: Cümleler arasındaki geçişleri sınıflandırarak daha iyi anlamayı sağlar (örneğin, merkez kurma, devam etme, yeni konu geçişi).
* Anaforik İlişkileri Etiketleme: Cümlelerdeki anaforik ilişkileri tanımlayarak bağlamı güçlendirir.

### Kullanım:
Sınıfın analyze metodu, metni analiz eder ve her cümle çifti için geçiş bilgilerini içeren bir liste döndürür. Her bir sonuç, cümlelerin içeriği, geçiş türü, isim tamlamaları ve anaforik ilişkileri hakkında detaylı bilgi sunar.

# TransitionScorer Sınıfı
TransitionScorer, geçiş türlerinin belirli ağırlıklarına dayanarak bir metindeki geçişleri puanlamak için kullanılan bir sınıftır. Sınıf, varsayılan ağırlıkları veya kullanıcıdan alınan özel ağırlıkları kullanarak geçiş türlerine puan atar.

### Özellikler:
* Ağırlıklar: Geçiş türlerine göre belirlenen puan ağırlıklarıyla çalışır. Varsayılan olarak belirli değerler ile başlar.
* Geçiş Puanlama: Her geçiş türü için bir puan döndürerek geçişlerin kalitesini ölçer.

### Kullanım:
score_transition metodu, verilen geçiş türü için puanı döndürür. Eğer geçiş türü tanınmıyorsa, varsayılan olarak 0 döner.

# CenteringModel Sınıfı
CenteringModel, bir metindeki geçişleri analiz edip puanlamak için TransitionAnalyzer ve TransitionScorer sınıflarını kullanan bir yapıdadır. Bu model, metnin genel geçiş kalitesini değerlendirmeyi amaçlar.

### Özellikler:
* Analiz ve Puanlama: Metni analiz eder ve her bir cümle çifti arasındaki geçişleri puanlar.
* Toplam Skor: Analiz sonuçları üzerinden toplam geçiş puanını hesaplar.

### Kullanım:
score_transitions metodu, metni analiz eder ve her cümle çifti için geçiş türü ile puanını içeren bir liste döndürür. Ayrıca, toplam puanı da sağlar. calculate_transition_score metodu, geçiş türünü haritalayarak doğru puanı alır ve böylece geçişlerin kalitesini daha iyi değerlendirir.

# Metin Verilerini Yükleme ve Geçiş Analizi

Bu kod, metin dosyalarını parça parça yükleyerek geçiş analizleri yapar.

## Fonksiyon: `load_text_data_in_chunks`
- Belirtilen boyutta metin dosyasını okur ve parça parça döner.

## Geçiş Analizi
- Her parça için `analyze_transitions` fonksiyonu ile geçiş türleri ve özellikler çıkartılır. Sonuçlar `full_transition_df` adında bir DataFrame'de birleştirilir.

## Verilerin Hazırlanması
- Cümleler, tokenizer ile sayısal dizilere dönüştürülüp sıfırlarla doldurulur (`X_padded`).

## Modelin Oluşturulması
- Derin öğrenme modeli, iki yönlü LSTM katmanları ve dropout kullanarak inşa edilir. Model, 'sparse_categorical_crossentropy' kaybı ile derlenir.

## Eğitim ve Değerlendirme
- Model, eğitim verileriyle eğitilir. Erken durdurma ve kontrol noktası ile en iyi hali kaydedilir. Eğitim ve test verilerinin şekilleri yazdırılır.

# Gelişmiş Dil Modeli

Bu Python kodu, metin verilerini kullanarak gelişmiş bir dil modeli oluşturur. Model, n-gram temelli kelime tahminleri yaparak cümleler üretir.

## Gerekli Kütüphaneler
- `random`, `pickle`: Rastgele seçim ve model kaydetme işlemleri için.
- `collections.defaultdict`: Varsayılan bir değer ile sözlük oluşturmak için.
- `spacy`: Doğal dil işleme için.
- `numpy`: Sayısal işlemler için.
- `tqdm`: İlerleme çubuğu için.

## Sınıf: `EnhancedLanguageModel`
### Yapıcı: `__init__`
- **Parametreler**: `text` (kullanılacak metin), `n` (n-gram boyutu).
- **İşlev**: Model ve toplam sayıları oluşturur.

### Metodlar
1. **`build_model(text)`**: 
   - Metin verisinden n-gram modeli oluşturur.
   - Kneser-Ney düzeltmesi ile olasılıkları normalleştirir.

2. **`generate_sentence(start_words=None, length=10)`**:
   - Belirtilen başlangıç kelimleri ile belirtilen uzunlukta cümle üretir.

3. **`choose_word_with_context(next_words)`**:
   - Verilen olasılıklara göre bağlama uygun bir kelime seçer.

4. **`clean_text(text)`**:
   - Cümleleri temizler ve biçimlendirir (gereksiz boşlukları ve yazım hatalarını düzeltir).

5. **`post_process_sentences(sentences)`**:
   - Oluşturulan cümleleri birleştirir ve tematik tutarlılık için kontrol eder.

6. **`generate_and_post_process(num_sentences=10, input_words=None, length=20)`**:
   - Belirtilen sayıda cümle üretir ve bunları işlemden geçirir.

7. **`save_model(filename)`**:
   - Modeli belirtilen dosyaya kaydeder.

8. **`load_model(cls, filename)`**:
   - Kaydedilmiş bir modeli yükler.

## Metin Yükleme Fonksiyonu: `load_text_from_file`
- Belirtilen dosyadan metni okur.

## Kullanım
1. Metni belirtilen dosyadan yükler.
2. Mevcut model dosyasını kontrol eder. Dosya mevcutsa, modeli yükler; değilse, yeni bir model oluşturur ve kaydeder.
3. Belirtilen sayıda cümle üretir ve sonucu ekrana yazdırır.

## Örnek Kullanım
- 5 cümle üretmek için başlangıç kelimeleri olarak `The next morning, Mia sent the entire ledger to the press, a digital bomb waiting to explode.` kullanılır.

## Üretilen Metin

```
Generated Text:
   The next morning, Mia sent the entire ledger to the press, a digital bomb waiting to explode. 
   Duncan to feel my gaze, he exclaim try a butcher 's terror, Inspector he alive there, we urge Most certainly which communicate with reluctance, the ubiquitous reporter anonymously share. 
   Be fix upon the receipt in that day we’ll regret the email it shine through Carlyle Beauvais prevail that is cause we number and restless pick an exciting mystery attend he. 
   Difficulty and I seek Holmes unfold the cry Jones put the impertinence of expose he write she still to trap I pant somewhere near troop of Overland here you arelieve they. 
   Morstan say cheerily look his pocket the quinine bottle would prevent any problem and peril let loose method have one evening sunshine Wiggins be preconcerted management here the facilis descensus.
```
### Değerlendirme

Üretilen metin, dilbilimsel ve yazılımsal bir çerçevede incelendiğinde, geçiş analizi açısından ilginç veriler sunmaktadır. Cümle yapılarının karmaşıklığı, metin içindeki referansların ve geçişlerin belirsizliğini artırırken, aynı zamanda metnin zenginliğini de vurgular. Örneğin, soyut ifadeler ve imgesel dil, sözdizimsel yapıların incelenmesine olanak tanır ve dilin işleyişine dair önemli ipuçları sunar.

Yazılım geliştirme bağlamında, bu tür metinlerin analizi, doğal dil işleme (NLP) uygulamalarında geçişlerin ve bağlamın modellemesi açısından kritik bir rol oynar. Geçişlerin belirlenmesi ve analiz edilmesi, metinler arası ilişkilerin ortaya konmasına yardımcı olur, böylece metnin anlamı daha net bir şekilde anlaşılabilir. Sonuç olarak, bu metin, hem dilbilimsel inceleme hem de yazılımsal modelleme açısından zengin bir kaynak oluşturarak, geçiş analizi çalışmalarına değerli katkılarda bulunabilir.

#### 07.11.2024 Değerlendirme

Metin, dil çeşitliliği ve atmosferik, gerilimli ton açısından güçlü bir potansiyele sahip. Dil bilgisi yapısı, tutarlılık ve mantıksal akışta iyileştirmelerle daha okunabilir ve etkileyici hale gelebilir.


