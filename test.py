"""
Centering-LGram v1.1.2 Test & Demo
Başka projede kullanım örnekleri
"""

print("🎉 Centering-LGram v1.1.2 Test & Demo")
print("="*60)

# =============================================================================
# 📦 1. TEMEL İMPORT TEST
# =============================================================================
try:
    from lgram.models.simple_language_model import create_language_model
    print("✅ Model import başarılı")
except ImportError as e:
    print(f"❌ Import hatası: {e}")
    print("💡 Çözüm: pip install centering-lgram==1.1.2")
    exit(1)

# =============================================================================
# 🎯 2. MODEL OLUŞTURMA VE TEST
# =============================================================================
print("\n🎯 Model Oluşturuluyor...")
model = create_language_model()
print("✅ Model başarıyla oluşturuldu")

# =============================================================================
# 🚀 3. CENTERING THEORY İLE METİN ÜRETİMİ
# =============================================================================
print("\n🚀 Centering Theory ile Metin Üretimi")
print("-" * 40)

test_cases = [
    {"input": "The truth", "sentences": 3, "length": 12},
    {"input": "Ancient mysteries", "sentences": 2, "length": 15},
    {"input": "Technology evolves", "sentences": 4, "length": 10}
]

for i, test in enumerate(test_cases, 1):
    print(f"\n📝 Test {i}: '{test['input']}'")
    input_words = test["input"].split()
    
    generated_text = model.generate_text_with_centering(
        num_sentences=test["sentences"],
        input_words=input_words,
        length=test["length"],
        use_progress_bar=True
    )
    
    print(f"🎯 Üretilen: {generated_text}")

print("\n" + "="*60)
print("✅ Tüm testler başarıyla tamamlandı!")
print("📖 Daha fazla örnek için: usage_examples.py")
print("📋 Kullanım rehberi için: USAGE_GUIDE.md")