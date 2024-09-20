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
