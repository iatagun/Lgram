# 🚀 Centering-LGram Başka Projede Kullanım

## 📦 Kurulum

```bash
pip install centering-lgram==1.1.2
```

## ⚡ Hızlı Başlangıç

```python
# Temel kullanım
from lgram.models.simple_language_model import create_language_model

model = create_language_model()
text = model.generate_text_with_centering(
    input_words=["The", "mystery"],
    num_sentences=3,
    length=12
)
print(text)
```

## 🎯 Ana Özellikler

### 1. **Centering Theory ile Metin Üretimi**
```python
from lgram.models.simple_language_model import EnhancedLanguageModel

model = EnhancedLanguageModel()
result = model.generate_text_with_centering(
    input_words=["Ancient", "castle"],
    num_sentences=5,
    length=15,
    use_progress_bar=True
)
```

### 2. **Centering Theory Analizi**
```python
from lgram.models.centering_theory import CenteringTheory

ct = CenteringTheory()
analysis = ct.analyze_text("John went home. He was tired.")
for sentence in analysis:
    print(f"Transition: {sentence['transition_type']}")
```

### 3. **Pattern Learning**
```python
from lgram.models.transition_pattern_learner import TransitionPatternLearner

learner = TransitionPatternLearner()
patterns = learner.extract_patterns("Your text here...")
```

## 🎛️ Gelişmiş Kullanım

### Command Line Interface
```bash
# Metin üretimi
lgram generate "The old man" --sentences 3 --length 10

# Analiz
lgram analyze "Your text" --show-transitions

# Batch işleme
lgram batch input.txt output.txt --mode generate
```

### Özelleştirme
```python
# Kendi verini yükle
model = create_model(data_path='my_data.txt')

# Cache optimizasyonu
model.enable_smart_cache(max_size=10000)

# Neural ranking
ct = CenteringTheory(use_neural_ranking=True)
```

## 📊 Transition Types

- **CONTINUE**: Aynı entity'ye odaklanma devam ediyor
- **RETAIN**: Ana entity değişiyor ama önceki entity kalıyor  
- **SMOOTH_SHIFT**: Yumuşak geçiş yeni entity'ye
- **ROUGH_SHIFT**: Ani geçiş tamamen yeni entity'ye

## 🌟 Gerçek Dünya Örnekleri

### Blog Yazısı Tamamlama
```python
blog_continuation = model.generate_text_with_centering(
    input_words=['Technology', 'changes', 'society'],
    num_sentences=4,
    length=20
)
```

### Hikaye Yazımı
```python
story = model.generate_creative_text(
    prompt="Dark mysterious forest",
    genre="fantasy",
    num_sentences=6
)
```

### Chatbot Yanıtları
```python
response = model.continue_conversation(
    context="User asked about weather",
    style="friendly"
)
```

## 🔧 Import Yapısı

```python
# Ana modüller
from lgram.models.simple_language_model import (
    EnhancedLanguageModel,
    create_language_model
)

from lgram.models.centering_theory import CenteringTheory

from lgram.models.transition_pattern_learner import (
    TransitionPatternLearner
)

# Utilities
from lgram.core import create_model
from lgram.utils import text_preprocessing
```

## 📈 Performance

- **SmartCache**: 482.9x hızlanma
- **Memory Efficient**: Optimize edilmiş memory kullanımı
- **Neural Ranking**: Gelişmiş entity ranking
- **Pre-trained Models**: 4.5MB Edgar Allan Poe corpus

## 🐛 Troubleshooting

### Import Hataları
```bash
# Güncelleme
pip install centering-lgram==1.1.2 --upgrade

# Cache temizleme
pip cache purge
```

### Memory Sorunları
```python
# Cache boyutunu azalt
model.enable_smart_cache(max_size=1000)

# Batch size ayarla
model.set_batch_size(16)
```

## 📚 Daha Fazla Örnek

Detaylı örnekler için `usage_examples.py` dosyasına bakın!

## 🔗 Links

- **PyPI**: https://pypi.org/project/centering-lgram/
- **GitHub**: https://github.com/iatagun/Lgram
- **Documentation**: Package içinde CHANGELOG.md
