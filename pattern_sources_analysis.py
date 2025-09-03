#!/usr/bin/env python3
"""
Pattern Sources Analysis: Where do centering patterns come from?
"""

import sys
import os
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgram.models.simple_language_model import create_language_model

def analyze_pattern_sources():
    """Analyze where centering patterns are sourced from"""
    print("🔍 MERKEZLEME ÖRÜNTÜSÜ KAYNAKLARI ANALİZİ")
    print("=" * 60)
    
    model = create_language_model()
    
    print("📊 PATTERN KAYNAKLARI:")
    print("1️⃣ Kaliteli referans metinler (learn_from_quality_text)")
    print("2️⃣ Önceden kaydedilmiş pattern'ler (transition_patterns.json)")
    print("3️⃣ Canlı analiz ve öğrenme (dynamic learning)")
    
    # Check current patterns
    pattern_file = os.path.join("c:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\lgram\\ngrams", "transition_patterns.json")
    
    if os.path.exists(pattern_file):
        with open(pattern_file, 'r') as f:
            data = json.load(f)
        
        patterns = data.get('patterns', {})
        bigram_transitions = data.get('bigram_transitions', {})
        trigram_transitions = data.get('trigram_transitions', {})
        
        print(f"\n📈 MEVCUT PATTERN İSTATİSTİKLERİ:")
        print(f"   Toplam Pattern: {len(patterns)}")
        print(f"   Bigram Transitions: {len(bigram_transitions)}")
        print(f"   Trigram Transitions: {len(trigram_transitions)}")
        
        print(f"\n🔬 PATTERN ÖRNEKLERİ:")
        for i, (pattern_id, pattern_data) in enumerate(list(patterns.items())[:3]):
            sequence = pattern_data['sequence']
            frequency = pattern_data['frequency']
            centers = pattern_data['context_centers'][:3]  # First 3 centers
            
            print(f"   Pattern {i+1}: {' → '.join(sequence)}")
            print(f"      Sıklık: {frequency} kez görüldü")
            print(f"      Merkezler: {centers}")
            print(f"      Tutarlılık: {pattern_data['coherence_score']:.2f}")
    
    print(f"\n🎯 PATTERN ÖĞRENME KAYNAKLARI:")
    
    # Source 1: Quality texts
    print(f"\n1️⃣ KALİTELİ REFERANS METİNLER:")
    print("   • Yüksek kaliteli metinlerden pattern öğrenme")
    print("   • Akademik makaleler, kitaplar, profesyonel yazılar")
    print("   • Manuel olarak quality_score ile işaretlenen metinler")
    
    example_text = """
    Machine learning transforms data into insights. 
    These algorithms identify patterns humans might miss. 
    Such capabilities enhance decision-making across industries.
    """
    
    print(f"   ÖRNEK REFERANS METİN:")
    for i, line in enumerate(example_text.strip().split('\n')):
        if line.strip():
            print(f"      {i+1}. {line.strip()}")
    
    # Learn from this text
    result = model.learn_from_quality_text(example_text.strip(), quality_score=0.95)
    print(f"   ✅ Bu metinden {result.get('patterns_learned', 0)} pattern öğrenildi!")
    
    # Source 2: Saved patterns
    print(f"\n2️⃣ KAYDEDİLMİŞ PATTERN'LER:")
    print("   • transition_patterns.json dosyasından yükleme")
    print("   • Önceki öğrenme oturumlarından kalıcı saklama")
    print("   • Model her başlatıldığında otomatik yükleme")
    
    # Source 3: Dynamic learning
    print(f"\n3️⃣ CANLI ÖĞRENMEve ANALİZ:")
    print("   • Gerçek zamanlı metin analizi")
    print("   • Üretilen metinlerin tutarlılık analizi")
    print("   • Feedback loop ile sürekli iyileştirme")
    
    # Test dynamic learning
    test_text = "Artificial intelligence changes everything. It transforms how we work. These changes affect everyone."
    analysis = model.analyze_text_coherence(test_text)
    
    print(f"   TEST METNİ: {test_text}")
    if 'error' not in analysis:
        print(f"   Tutarlılık: {analysis.get('coherence_score', 0):.2f}")
        print(f"   Geçişler: {', '.join(analysis.get('transitions', []))}")
        print(f"   ✅ Bu analiz yeni pattern öğrenmesine katkıda bulunur!")

def show_pattern_flow():
    """Show how patterns flow through the system"""
    print(f"\n🔄 PATTERN AKIŞI:")
    print("=" * 60)
    
    print("📥 GİRİŞ KAYNAKLARI:")
    print("   1. Yüksek kaliteli referans metinler")
    print("   2. Akademik yayınlar")
    print("   3. Profesyonel yazılar")
    print("   4. El ile seçilmiş kaliteli örnekler")
    
    print(f"\n⚙️  İŞLEME ADIMLARı:")
    print("   1. Metin cümlelere bölünür")
    print("   2. Her cümle için söylem merkezi çıkarılır")
    print("   3. Cümleler arası geçiş türleri belirlenir")
    print("   4. Geçiş dizileri pattern olarak kaydedilir")
    print("   5. Pattern'ler sıklık ve kalite puanıyla saklanır")
    
    print(f"\n💾 SAKLAMA:")
    print("   • transition_patterns.json → Kalıcı saklama")
    print("   • SmartCache → Hızlı erişim (800 pattern cache)")
    print("   • Memory → Çalışma zamanı optimizasyonu")
    
    print(f"\n🎯 KULLANIM:")
    print("   • generate_coherent_text → Pattern tabanlı üretim")
    print("   • analyze_text_coherence → Kalite kontrolü")
    print("   • transition sequence generation → Akıcılık kontrolü")

def demonstrate_pattern_learning():
    """Demonstrate how patterns are learned in real-time"""
    print(f"\n🚀 GERÇEK ZAMANLI PATTERN ÖĞRENMESİ:")
    print("=" * 60)
    
    model = create_language_model()
    
    # Before learning
    if model.pattern_learner:
        stats_before = model.pattern_learner.get_pattern_statistics()
        print(f"ÖĞRENME ÖNCESİ: {stats_before.get('total_patterns', 0)} pattern")
    
    # Learn from multiple quality sources
    quality_sources = [
        {
            "text": "Artificial intelligence revolutionizes technology. It enables unprecedented capabilities. These capabilities transform industries fundamentally.",
            "source": "Tech Article",
            "quality": 0.9
        },
        {
            "text": "Climate change affects global ecosystems. It disrupts natural patterns significantly. These disruptions threaten biodiversity worldwide.",
            "source": "Scientific Paper", 
            "quality": 0.95
        },
        {
            "text": "Economic policies shape market dynamics. They influence investment decisions directly. These decisions determine future prosperity levels.",
            "source": "Economic Analysis",
            "quality": 0.88
        }
    ]
    
    total_learned = 0
    for source in quality_sources:
        result = model.learn_from_quality_text(source["text"], source["quality"])
        patterns_learned = result.get('patterns_learned', 0)
        total_learned += patterns_learned
        
        print(f"\n📚 {source['source']} (Quality: {source['quality']}):")
        print(f"   Metin: {source['text'][:60]}...")
        print(f"   Öğrenilen: {patterns_learned} pattern")
    
    # After learning
    if model.pattern_learner:
        stats_after = model.pattern_learner.get_pattern_statistics()
        print(f"\nÖĞRENME SONRASI: {stats_after.get('total_patterns', 0)} pattern")
        print(f"✅ Toplam {total_learned} yeni pattern öğrenildi!")
    
    print(f"\n🎉 SONUÇ: Pattern'ler sürekli kaliteli metinlerden öğreniliyor!")

def main():
    """Main analysis function"""
    analyze_pattern_sources()
    show_pattern_flow()
    demonstrate_pattern_learning()
    
    print(f"\n🏆 ÖZET: MERKEZLEME ÖRÜNTÜSÜ KAYNAKLARI")
    print("="*60)
    print("✅ Kaliteli referans metinlerden öğrenme")
    print("✅ Kalıcı dosya sisteminde saklama") 
    print("✅ Gerçek zamanlı dinamik öğrenme")
    print("✅ Cache sistemi ile hızlı erişim")
    print("✅ Sürekli iyileştirme döngüsü")

if __name__ == "__main__":
    main()
