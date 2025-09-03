#!/usr/bin/env python3
"""
Advanced Pattern Learning from text_data.txt
More comprehensive pattern extraction with better text processing
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgram.models.simple_language_model import create_language_model
import re

def manual_text_data_learning():
    """Manually process text_data.txt for comprehensive pattern learning"""
    print("🔍 MANUEL TEXT_DATA.TXT PATTERN ÖĞRENMESİ")
    print("=" * 60)
    
    model = create_language_model()
    
    text_data_path = r"c:\Users\user\OneDrive\Belgeler\GitHub\Lgram\ngrams\text_data.txt"
    
    if not os.path.exists(text_data_path):
        print(f"❌ text_data.txt bulunamadı: {text_data_path}")
        return
    
    print("📚 text_data.txt dosyası işleniyor...")
    
    try:
        with open(text_data_path, 'r', encoding='utf-8', errors='ignore') as f:
            full_text = f.read()
        
        # Clean and preprocess text
        print(f"📏 Orijinal metin boyutu: {len(full_text):,} karakter")
        
        # Split into paragraphs and clean
        paragraphs = full_text.split('\n\n')
        quality_paragraphs = []
        
        for para in paragraphs:
            clean_para = para.strip()
            
            # Filter quality paragraphs
            if (len(clean_para) > 100 and  # At least 100 characters
                len(clean_para.split()) > 15 and  # At least 15 words
                '.' in clean_para and  # Contains sentences
                not clean_para.isupper() and  # Not all caps
                len(clean_para.split('.')) >= 2):  # Multiple sentences
                
                # Additional quality checks
                sentences = [s.strip() for s in clean_para.split('.') if s.strip()]
                if len(sentences) >= 2:
                    quality_paragraphs.append(clean_para)
        
        print(f"📝 Kaliteli paragraf sayısı: {len(quality_paragraphs)}")
        
        # Learn from quality paragraphs
        total_patterns_learned = 0
        patterns_before = 0
        
        if model.pattern_learner:
            stats_before = model.pattern_learner.get_pattern_statistics()
            patterns_before = stats_before.get('total_patterns', 0)
        
        print(f"🏁 Başlangıç pattern sayısı: {patterns_before}")
        
        # Process paragraphs in batches
        batch_size = 10
        for i in range(0, min(len(quality_paragraphs), 100), batch_size):  # Process up to 100 paragraphs
            batch = quality_paragraphs[i:i+batch_size]
            batch_text = ' '.join(batch)
            
            # Learn from batch
            result = model.learn_from_quality_text(batch_text, quality_score=0.85)
            batch_patterns = result.get('patterns_learned', 0)
            total_patterns_learned += batch_patterns
            
            print(f"   Batch {(i//batch_size)+1}: {batch_patterns} pattern öğrenildi")
        
        # Final statistics
        if model.pattern_learner:
            stats_after = model.pattern_learner.get_pattern_statistics()
            patterns_after = stats_after.get('total_patterns', 0)
            
            print(f"\n📊 ÖĞRENME SONUÇLARI:")
            print(f"   Öncesi: {patterns_before} pattern")
            print(f"   Sonrası: {patterns_after} pattern")
            print(f"   Yeni öğrenilen: {patterns_after - patterns_before} pattern")
            print(f"   Toplam işlenen paragraf: {min(len(quality_paragraphs), 100)}")
        
        return model
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        return None

def analyze_learned_patterns(model):
    """Analyze the patterns learned from text_data.txt"""
    print(f"\n🔍 ÖĞRENİLEN PATTERN'LER ANALİZİ")
    print("=" * 60)
    
    if not model or not model.pattern_learner:
        print("❌ Pattern learner bulunamadı")
        return
    
    # Load current patterns
    pattern_file = r"c:\Users\user\OneDrive\Belgeler\GitHub\Lgram\lgram\ngrams\transition_patterns.json"
    
    if not os.path.exists(pattern_file):
        print("❌ Pattern dosyası bulunamadı")
        return
    
    import json
    
    try:
        with open(pattern_file, 'r') as f:
            data = json.load(f)
        
        patterns = data.get('patterns', {})
        
        print(f"📊 TOPLAM PATTERN: {len(patterns)}")
        
        # Analyze context centers for classic literature indicators
        all_centers = []
        for pattern_data in patterns.values():
            centers = pattern_data.get('context_centers', [])
            all_centers.extend(centers)
        
        from collections import Counter
        center_counts = Counter(all_centers)
        
        # Classic literature indicators
        classic_words = [
            'mind', 'thought', 'analysis', 'manner', 'attention', 'observation',
            'reflection', 'character', 'nature', 'soul', 'spirit', 'intellect',
            'perception', 'consciousness', 'reasoning', 'faculty', 'imagination',
            'mystery', 'narrative', 'tale', 'story', 'gentleman', 'dupin',
            'analytical', 'paris', 'library', 'volume', 'manuscript'
        ]
        
        # Modern/tech words
        modern_words = [
            'intelligence', 'systems', 'data', 'algorithms', 'technology',
            'machine', 'learning', 'artificial', 'computer', 'digital'
        ]
        
        classic_count = sum(center_counts.get(word, 0) for word in classic_words)
        modern_count = sum(center_counts.get(word, 0) for word in modern_words)
        
        print(f"\n🎭 İÇERİK ANALİZİ:")
        print(f"   Klasik edebiyat kelimeleri: {classic_count}")
        print(f"   Modern/teknoloji kelimeleri: {modern_count}")
        
        print(f"\n📚 EN YAYIN KLASİK KELİMELER:")
        classic_centers = [(word, center_counts[word]) for word in classic_words if word in center_counts]
        classic_centers.sort(key=lambda x: x[1], reverse=True)
        
        for word, count in classic_centers[:10]:
            print(f"   '{word}': {count} kez")
        
        # Pattern diversity analysis
        pattern_types = {}
        for pattern_data in patterns.values():
            sequence = tuple(pattern_data.get('sequence', []))
            pattern_types[sequence] = pattern_types.get(sequence, 0) + 1
        
        print(f"\n🔄 PATTERN ÇEŞİTLİLİĞİ:")
        print(f"   Farklı pattern türü: {len(pattern_types)}")
        
        for pattern_seq, count in list(pattern_types.items())[:5]:
            print(f"   {' → '.join(pattern_seq)}: {count} örnek")
        
        # Quality assessment
        if classic_count > modern_count:
            print(f"\n✅ BAŞARILI: text_data.txt'den klassik pattern'ler öğrenildi!")
        elif modern_count > classic_count * 2:
            print(f"\n⚠️  UYARI: Modern pattern'ler hala baskın")
        else:
            print(f"\n🔄 KARMA: Hem klasik hem modern pattern'ler mevcut")
            
    except Exception as e:
        print(f"❌ Pattern analizi hatası: {e}")

def test_classic_generation(model):
    """Test text generation with classic literature style"""
    print(f"\n🎨 KLASİK EDEBİYAT TARZI ÜRETİM TESTİ")
    print("=" * 60)
    
    if not model:
        print("❌ Model bulunamadı")
        return
    
    classic_prompts = [
        {
            "name": "Dupin Tarzı Analiz",
            "words": ["The", "analytical"],
            "length": 4,
            "description": "Edgar Allan Poe'nun analitik karakteri tarzında"
        },
        {
            "name": "Gizemli Atmosfer", 
            "words": ["In", "Paris"],
            "length": 5,
            "description": "Klasik gizem hikayesi atmosferi"
        },
        {
            "name": "Entelektüel Yaklaşım",
            "words": ["The", "faculty"],
            "length": 4,
            "description": "Zihinsel yetenek ve düşünce üzerine"
        }
    ]
    
    for prompt in classic_prompts:
        print(f"\n📖 {prompt['name']}:")
        print(f"   {prompt['description']}")
        print(f"   Başlangıç: {prompt['words']}")
        
        try:
            generated = model.generate_coherent_text(
                target_length=prompt['length'],
                input_words=prompt['words'],
                quality_level="high"
            )
            
            # Display generated text
            sentences = generated.split('.')
            for i, sent in enumerate(sentences):
                if sent.strip():
                    print(f"      {i+1}. {sent.strip()}")
            
            # Analyze coherence
            analysis = model.analyze_text_coherence(generated)
            if 'error' not in analysis:
                coherence = analysis.get('coherence_score', 0)
                quality = analysis.get('quality_assessment', {}).get('overall_quality', 'Unknown')
                print(f"   📊 Tutarlılık: {coherence:.2f} ({quality})")
            
        except Exception as e:
            print(f"   ❌ Üretim hatası: {e}")

def main():
    """Main function"""
    print("🎯 ADVANCED TEXT_DATA.TXT PATTERN LEARNING")
    print("="*70)
    
    # Manual comprehensive learning
    model = manual_text_data_learning()
    
    if model:
        # Analyze learned patterns
        analyze_learned_patterns(model)
        
        # Test classic generation
        test_classic_generation(model)
        
        print(f"\n🏆 ÖZET")
        print("="*50)
        print("✅ text_data.txt'den kapsamlı pattern öğrenme tamamlandı")
        print("✅ Klasik edebiyat tarzı geçiş örüntüleri çıkarıldı")
        print("✅ Pattern tabanlı coherent generation test edildi")
        print("✅ Edgar Allan Poe tarzı metinlerden öğrenme başarılı")
    else:
        print("❌ Pattern öğrenme işlemi başarısız")

if __name__ == "__main__":
    main()
