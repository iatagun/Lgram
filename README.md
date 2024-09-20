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
