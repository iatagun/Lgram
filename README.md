# Transition Analyzer

Transition Analyzer, metin analizi için geliştirilmiş bir Python kütüphanesidir. Cümleler arasındaki geçişleri sınıflandırarak ve anaforik ilişkileri etiketleyerek metinlerin anlamını daha iyi anlamaya yardımcı olur.

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
   pip install spacy

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
