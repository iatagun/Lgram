# Transition Analyzer

Transition Analyzer, metin analizi için geliştirilmiş bir Python kütüphanesidir. Bu kütüphane, merkezleme kuramını temel alarak cümleler arasındaki bağdaşıklık ilişkilerini analiz eder. Merkezleme kuramı, bir metin içerisindeki belirli unsurların (özellikle isimler ve zamirler) birbirleriyle nasıl ilişkilendiğini inceleyerek metinlerin akışını ve anlamını daha iyi kavrayabilmemize yardımcı olur.

Bu araç, bilgisayarlı metin üretiminde bağdaşıklığı sağlamak için kritik öneme sahiptir. Cümleler arasındaki geçişlerin doğru bir şekilde sınıflandırılması, metinlerin daha tutarlı ve anlaşılır olmasını sağlar. Kullanıcıların metinlerini analiz ederek, geçiş türlerini belirler ve anaforik ilişkileri etiketler. Bu sayede, üretilecek metinlerin bağdaşıklığı artırılır, okuyucunun dikkatini çeker ve metinlerin akışkanlığını sağlar.

Transition Analyzer, doğal dil işleme (NLP) tekniklerini kullanarak, cümle çiftleri arasındaki ilişkiyi analiz eder ve bu ilişkileri farklı geçiş türlerine ayırır. Böylece, kullanıcılar metin üretiminde daha etkili sonuçlar elde edebilirler.

## Özellikler

- **Geçiş Sınıflandırması**: Cümle çiftleri arasındaki geçiş türlerini belirler.
  - Center Establishment (EST)
  - Center Continuation (CON)
  - Center Retaining (RET)
  - Smooth Shift (SSH)
  - Rough Shift (RSH)
  - New Topic Transition
- **Anaforik İlişki Belirleme**: Metin içindeki anaforik ilişkileri tanımlar.

## Gereksinimler

- Python 3.x
- spaCy kütüphanesi
- `en_core_web_sm` dil modeli

## Kurulum

1. **Kütüphaneyi Yükleyin**:

   ```bash
   pip install spacy
2. **Dil Modelini Yükleyin**:

   ```bash
   python -m spacy download en_core_web_sm


## Kullanım
   ```python
   from transition_analyzer import TransitionAnalyzer

   text = (
      "Alice was excited about her upcoming vacation. She had been planning it for months. "
      "Her friend Bob decided to join her. They both agreed that visiting Paris would be the highlight of the trip. "
      "Bob had always dreamed of seeing the Eiffel Tower. Later, they discussed what to do in the city."
   )

   analyzer = TransitionAnalyzer(text)
   results = analyzer.analyze()

   for idx, result in enumerate(results):
      print(f"Pair {idx + 1}:")
      print("Current Sentences:", result['current_sentences'])
      print("Next Sentences:", result['next_sentences'])
      print("Transition Type:", result['transition'])
      print("Current NPs (Possible Centers):", result['current_nps'])
      print("Next NPs (Possible Centers):", result['next_nps'])
      print("Anaphoric Relations:", result['anaphoric_relations'])
      print()
   ```
# Geçiş Analizi Sonuçları
   ```
   --- Transition 1 ---
   Current Sentence: Alice was excited about her upcoming vacation.
   Next Sentence: She had been planning it for months.
   Transition Type: Center Continuation (CON)
   Score: 3

   --- Transition 2 ---
   Current Sentence: She had been planning it for months.
   Next Sentence: Her friend Bob decided to join her.
   Transition Type: Rough Shift (RSH)
   Score: 1

   --- Transition 3 ---
   Current Sentence: Her friend Bob decided to join her.
   Next Sentence: They both agreed that visiting Paris would be the highlight of the trip.
   Transition Type: Center Continuation (CON)
   Score: 3

   Total Transition Score: 7
   
# Sentence Generator

## Süreç Detayları

1. **N-gram Kullanımı**:
- İlk aşamada, `SentenceGenerator` sınıfı içinde cümleler mevcut metinden alınarak n-gram tabanlı bir yapı oluşturulabilir. N-gram, belirli bir kelime dizisi uzunluğunu temsil eder; örneğin, `n=1` (unigram), `n=2` (bigram) gibi.
- Bu yapı, kelimeler arasındaki geçişleri modellemek için kullanılabilir, ancak burada esas olarak mevcut cümlelerden rastgele seçim yapılmaktadır.

2. **Merkezleme Modelinin Kullanımı**:
- Geçiş türü tahmininde `best_transition_model.keras` modelini kullanarak, mevcut cümle ile yeni cümle arasındaki ilişkiyi değerlendiriyoruz. Bu model, belirli bir bağlamda uygun cümle geçişlerini tahmin etmek üzere eğitilmiştir.
- Cümle geçişinin uygun olup olmadığını belirlemek için model, cümlelerin belirli bir özellikler setini (örneğin, benzerlik oranı) değerlendirir.

3. **Döngü ve Metin Üretimi**:
- Başlangıç cümlesi ile başlayan bir döngü kurulur. Her döngü adımında, yeni bir cümle üretilir ve mevcut cümle ile kıyaslanır.
- Eğer geçiş modeli yeni cümleyi uygun bulmazsa (örneğin, cümleler çok benzer veya geçiş türü istenilen biçimde değilse), yeni bir cümle seçilir.
- Uygun bir cümle bulunduğunda, bu cümle `generated_text` değişkenine eklenir ve mevcut cümle güncellenir.

4. **Devamlılık**:
- Bu döngü, belirtilen `num_sentences` kadar devam eder ve her seferinde cümle geçişlerini değerlendirir. Sonuç olarak, mantıklı ve bağlamla uyumlu bir metin oluşturur.

## Özetle
Metin üretimi sürecinde hem n-gram tabanlı cümle seçimleri hem de merkezleme modeli (transition model) birlikte çalışarak, yeni ve uyumlu cümleler oluşturur. Bu iki bileşen, bir döngü içinde birbirini besler ve bağdaşık bir metin oluşturulmasını sağlar.

### Kullanım Örneği

Aşağıda, `SentenceGenerator` sınıfının nasıl kullanılacağına dair kısa bir örnek verilmiştir:

```python
# Gerekli kütüphaneleri içe aktarın
from your_module import SentenceGenerator, load_text_data

# Metin dosyasının yolunu belirtin
text_file_path = "path/to/text_gen_data.txt"
transition_model_path = "path/to/best_transition_model.keras"

# Metin verisini yükleyin
text = load_text_data(text_file_path)

# Cümle üreteciyi oluşturun
sentence_generator = SentenceGenerator(text, transition_model_path)

# Başlangıç cümlesi belirleyin
initial_sentence = "Deep within the enchanted woods of Eldoria, a long-forgotten prophecy began to awaken."

# Metin üretin
generated_text = sentence_generator.generate_text(initial_sentence, num_sentences=5)

print("Generated Text:")
print(generated_text)
