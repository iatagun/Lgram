#!/usr/bin/env python3
"""
Centering-LGram Kullanım Örnekleri
PyPI Package: centering-lgram==1.1.2
"""

# =============================================================================
# 🎯 1. TEMEL METIN ÜRETİMİ (Centering Theory ile)
# =============================================================================
def basic_text_generation():
    """Centering Theory kullanarak temel metin üretimi"""
    print("🎯 Temel Metin Üretimi (Centering Theory)")
    print("="*50)
    
    from lgram.models.simple_language_model import create_language_model
    
    # Model oluştur (otomatik öğrenme ile)
    model = create_language_model()
    
    # Giriş metni
    input_text = "The mysterious shadow"
    input_words = input_text.split()
    
    # Centering theory ile metin üret
    generated = model.generate_text_with_centering(
        num_sentences=3,
        input_words=input_words,
        length=12,
        use_progress_bar=True
    )
    
    print(f"Giriş: {input_text}")
    print(f"Üretilen: {generated}")
    print()


# =============================================================================
# 🧠 2. CENTERING THEORY ANALİZİ
# =============================================================================
def centering_analysis():
    """Metnin centering theory analizi"""
    print("🧠 Centering Theory Analizi")
    print("="*50)
    
    from lgram.models.centering_theory import CenteringTheory
    
    # Centering Theory analyzer
    ct = CenteringTheory()
    
    # Test metni
    text = """John went to the store. He bought some milk. The cashier was friendly. 
              She smiled at John. He thanked her and left."""
    
    # Analiz yap
    analysis = ct.analyze_text(text)
    
    print("📝 Test Metni:")
    print(text)
    print("\n🔍 Centering Analizi:")
    
    for i, sentence_info in enumerate(analysis):
        entities = sentence_info.get('entities', [])
        transition = sentence_info.get('transition_type', 'N/A')
        print(f"Cümle {i+1}: {transition} - Entities: {entities}")
    print()


# =============================================================================
# 🎛️ 3. GELIŞMIŞ PARAMETRELİ KULLANIM
# =============================================================================
def advanced_generation():
    """Gelişmiş parametreli metin üretimi"""
    print("🎛️ Gelişmiş Parametreli Kullanım")
    print("="*50)
    
    from lgram.models.simple_language_model import EnhancedLanguageModel
    
    # Enhanced model (direct access)
    model = EnhancedLanguageModel()
    
    # Çeşitli parametrelerle üretim
    scenarios = [
        {"input": "The ancient castle", "sentences": 2, "length": 10},
        {"input": "Technology advances", "sentences": 3, "length": 8},
        {"input": "Dreams and reality", "sentences": 2, "length": 15}
    ]
    
    for scenario in scenarios:
        input_words = scenario["input"].split()
        result = model.generate_text_with_centering(
            num_sentences=scenario["sentences"],
            input_words=input_words,
            length=scenario["length"],
            use_progress_bar=False
        )
        print(f"'{scenario['input']}' → {result}")
    print()


# =============================================================================
# 📊 4. TRANSİTİON PATTERN ANALİZİ
# =============================================================================
def transition_analysis():
    """Geçiş pattern analizi"""
    print("📊 Transition Pattern Analizi")
    print("="*50)
    
    from lgram.models.transition_pattern_learner import TransitionPatternLearner
    
    # Pattern learner
    learner = TransitionPatternLearner()
    
    # Test metinleri
    test_texts = [
        "The cat sat on the mat. It was comfortable.",
        "John loves music. He plays guitar every day. The instrument is his passion.",
        "Rain started falling. The streets became wet. People opened umbrellas."
    ]
    
    print("🔍 Pattern Analizi:")
    for i, text in enumerate(test_texts, 1):
        patterns = learner.extract_patterns(text)
        print(f"\nMetin {i}: '{text[:30]}...'")
        for pattern in patterns[:3]:  # İlk 3 pattern
            print(f"  • {pattern}")
    print()


# =============================================================================
# 🎨 5. COMMANd LINE INTERFACE
# =============================================================================
def cli_example():
    """CLI kullanım örneği"""
    print("🎨 Command Line Interface")
    print("="*50)
    
    print("Terminal'de şu komutları kullanabilirsiniz:")
    print()
    print("# Temel metin üretimi")
    print('lgram generate "The old man" --sentences 3 --length 10')
    print()
    print("# Centering analizi")
    print('lgram analyze "John went home. He was tired." --show-transitions')
    print()
    print("# Batch işleme")
    print('lgram batch input.txt output.txt --mode generate')
    print()


# =============================================================================
# 🔧 6. ÖZELLEŞTİRİLMİŞ KULLANIM
# =============================================================================
def custom_usage():
    """Özelleştirilmiş kullanım"""
    print("🔧 Özelleştirilmiş Kullanım")
    print("="*50)
    
    from lgram.core import create_model
    from lgram.models.centering_theory import CenteringTheory
    
    # Kendi veri setinle model oluştur
    print("📚 Kendi verini yükle:")
    print("model = create_model(data_path='my_texts.txt')")
    print()
    
    # Cache optimizasyonu
    print("⚡ Cache optimizasyonu:")
    print("model.enable_smart_cache(max_size=10000)")
    print()
    
    # Custom centering theory
    print("🎯 Özel centering ayarları:")
    print("ct = CenteringTheory(use_neural_ranking=True)")
    print()


# =============================================================================
# 🌟 7. GERÇEK DÜNYA ÖRNEKLERİ
# =============================================================================
def real_world_examples():
    """Gerçek dünya kullanım örnekleri"""
    print("🌟 Gerçek Dünya Örnekleri")
    print("="*50)
    
    print("📖 Blog yazısı tamamlama:")
    print("blog_text = model.generate_text_with_centering(")
    print("    input_words=['Artificial', 'intelligence', 'future'],")
    print("    num_sentences=5, length=15)")
    print()
    
    print("📰 Haber özetleme:")
    print("summary = model.summarize_with_centering(long_article)")
    print()
    
    print("💬 Chatbot yanıtları:")
    print("response = model.continue_conversation(")
    print("    context=user_message, style='friendly')")
    print()
    
    print("📝 Yaratıcı yazım:")
    print("story = model.generate_creative_text(")
    print("    prompt='Dark forest', genre='mystery')")
    print()


# =============================================================================
# 🚀 MAIN - TÜM ÖRNEKLERİ ÇALIŞTIR
# =============================================================================
if __name__ == "__main__":
    print("🎉 Centering-LGram v1.1.2 Kullanım Örnekleri")
    print("=" * 60)
    print()
    
    try:
        # 1. Temel kullanım
        basic_text_generation()
        
        # 2. Centering analizi  
        centering_analysis()
        
        # 3. Gelişmiş kullanım
        advanced_generation()
        
        # 4. Pattern analizi
        transition_analysis()
        
        # 5. CLI örnekleri
        cli_example()
        
        # 6. Özelleştirilmiş kullanım
        custom_usage()
        
        # 7. Gerçek dünya örnekleri
        real_world_examples()
        
        print("✅ Tüm örnekler tamamlandı!")
        
    except ImportError as e:
        print(f"❌ Import hatası: {e}")
        print("💡 Çözüm: pip install centering-lgram==1.1.2")
        
    except Exception as e:
        print(f"⚠️ Hata: {e}")


# =============================================================================
# 📋 HIZLI BAŞLANGIÇ TEMPLATE
# =============================================================================
"""
HIZLI BAŞLANGIÇ TEMPLATE:

# 1. Kurulum
pip install centering-lgram==1.1.2

# 2. Temel kod
from lgram.models.simple_language_model import create_language_model

model = create_language_model()
result = model.generate_text_with_centering(
    input_words=["Your", "input"],
    num_sentences=3,
    length=10
)
print(result)
"""
