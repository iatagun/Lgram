#!/usr/bin/env python3
"""
Text Sources Analysis: Where do the pattern learning texts actually come from?
Complete analysis of all text sources used for pattern learning
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_text_sources():
    """Analyze all sources of text used for pattern learning"""
    print("🔍 PATTERN ÖĞRENMESİ İÇİN KULLANILAN METİN KAYNAKLARI")
    print("=" * 70)
    
    print("📚 MEVCUT METİN KAYNAKLARI:")
    
    # 1. Training data files
    print(f"\n1️⃣ TEMEL EĞİTİM VERİSETLERİ:")
    
    # text_data.txt
    text_data_path = r"c:\Users\user\OneDrive\Belgeler\GitHub\Lgram\ngrams\text_data.txt"
    if os.path.exists(text_data_path):
        size = os.path.getsize(text_data_path) / (1024 * 1024)  # MB
        with open(text_data_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = sum(1 for _ in f)
        print(f"   📖 text_data.txt: {size:.2f}MB, ~{lines:,} lines")
        print(f"      İçerik: Klasik edebiyat metinleri (Edgar Allan Poe)")
        print(f"      Amaç: N-gram modeli eğitimi")
        print(f"      Kaynak: Project Gutenberg klasikleri")
        
        # Show sample
        with open(text_data_path, 'r', encoding='utf-8', errors='ignore') as f:
            sample = f.read(200)
        print(f"      Örnek: {sample[:100]}...")
    
    # more.txt
    more_path = r"c:\Users\user\OneDrive\Belgeler\GitHub\Lgram\ngrams\more.txt"
    if os.path.exists(more_path):
        size = os.path.getsize(more_path) / 1024  # KB
        with open(more_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = sum(1 for _ in f)
        print(f"\n   📝 more.txt: {size:.2f}KB, ~{lines:,} lines")
        print(f"      İçerik: Kısa cümle örnekleri")
        print(f"      Amaç: Basit pattern örnekleri")
        print(f"      Format: Akademik kelime kombinasyonları")
        
        # Show sample
        with open(more_path, 'r', encoding='utf-8', errors='ignore') as f:
            sample_lines = [f.readline().strip() for _ in range(3)]
        print(f"      Örnek: {sample_lines}")
    
    # 2. Test files with quality texts
    print(f"\n2️⃣ KALİTE TEST METİNLERİ:")
    
    test_files = [
        "test_pattern_learning.py",
        "analyze_model_vision.py", 
        "pattern_sources_analysis.py"
    ]
    
    for test_file in test_files:
        file_path = os.path.join(r"c:\Users\user\OneDrive\Belgeler\GitHub\Lgram", test_file)
        if os.path.exists(file_path):
            print(f"\n   📋 {test_file}:")
            
            # Extract quality texts from test files
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Find quality text examples
                import re
                quality_patterns = [
                    r'"""(.*?)"""',  # Triple quoted strings
                    r'"([^"]{50,})"'  # Long strings in quotes
                ]
                
                examples = []
                for pattern in quality_patterns:
                    matches = re.findall(pattern, content, re.DOTALL)
                    for match in matches:
                        clean_match = match.strip()
                        if len(clean_match) > 50 and any(word in clean_match.lower() 
                                                       for word in ['artificial', 'intelligence', 'climate', 'brain', 'machine']):
                            examples.append(clean_match[:100] + "...")
                
                if examples:
                    print(f"      Kaliteli örnekler bulundu: {len(examples)}")
                    for i, example in enumerate(examples[:2]):
                        print(f"         {i+1}. {example}")
                else:
                    print(f"      Kaliteli metin örneği bulunamadı")
                    
            except Exception as e:
                print(f"      Dosya okunamadı: {e}")
    
    # 3. Daily logs (if used for training)
    print(f"\n3️⃣ GÜNLİK LOG DOSYALARI:")
    
    logs_dir = r"c:\Users\user\OneDrive\Belgeler\GitHub\Lgram\logs"
    if os.path.exists(logs_dir):
        log_files = [f for f in os.listdir(logs_dir) if f.startswith("daily_log_") and f.endswith(".txt")]
        total_size = 0
        
        for log_file in log_files[:5]:  # First 5 files
            log_path = os.path.join(logs_dir, log_file)
            size = os.path.getsize(log_path)
            total_size += size
        
        print(f"   📊 Toplam {len(log_files)} log dosyası")
        print(f"   📏 Ortalama boyut: {total_size/len(log_files)/1024:.2f}KB")
        print(f"   🎯 Amaç: Model performans logları (pattern öğrenmesi için değil)")
        
        # Sample log
        if log_files:
            sample_log = os.path.join(logs_dir, log_files[0])
            try:
                with open(sample_log, 'r', encoding='utf-8', errors='ignore') as f:
                    sample = f.read(100)
                print(f"   📝 Örnek log: {sample[:50]}...")
            except:
                print(f"   📝 Log örneği okunamadı")

def analyze_pattern_learning_flow():
    """Show how texts flow into pattern learning"""
    print(f"\n🔄 PATTERN ÖĞRENMESİ AKIŞI")
    print("=" * 70)
    
    print("📥 METİN GİRİŞ KAYNAKLARI:")
    print("   1. Manuel kaliteli metinler (test scriptlerinde)")
    print("   2. Kullanıcı tarafından sağlanan referans metinler")
    print("   3. Dinamik olarak analiz edilen metinler")
    print("   4. Code içinde hardcoded kaliteli örnekler")
    
    print(f"\n⚙️  İŞLEME SÜRECİ:")
    print("   1. Metin learn_from_quality_text() fonksiyonuna girer")
    print("   2. TransitionPatternLearner.learn_from_text() çalışır")
    print("   3. Cümleler ayrıştırılır ve analiz edilir")
    print("   4. Centering Theory analizi yapılır")
    print("   5. Transition sequence'ler çıkarılır")
    print("   6. Pattern'ler frekans ve kalite ile saklanır")
    
    print(f"\n💾 SAKLAMA YERLERİ:")
    print("   • transition_patterns.json → Ana pattern deposu")
    print("   • SmartCache → Hızlı erişim cache'i")
    print("   • Memory → Çalışma zamanı geçici saklama")
    
    print(f"\n🎯 KULLANIM YERLERİ:")
    print("   • generate_coherent_text() → Pattern-based generation")
    print("   • analyze_text_coherence() → Quality assessment")
    print("   • generate_coherent_transition_sequence() → Sequence planning")

def show_current_pattern_sources():
    """Show what's actually being used right now"""
    print(f"\n🔍 ŞU AN AKTİF OLAN PATTERN KAYNAKLARI")
    print("=" * 70)
    
    # Check transition patterns file
    pattern_file = r"c:\Users\user\OneDrive\Belgeler\GitHub\Lgram\lgram\ngrams\transition_patterns.json"
    if os.path.exists(pattern_file):
        import json
        try:
            with open(pattern_file, 'r') as f:
                data = json.load(f)
            
            patterns = data.get('patterns', {})
            
            print(f"✅ MEVCUT PATTERN'LER: {len(patterns)} adet")
            
            # Analyze pattern sources by examining centers
            all_centers = []
            for pattern_data in patterns.values():
                centers = pattern_data.get('context_centers', [])
                all_centers.extend(centers)
            
            # Count center frequencies
            from collections import Counter
            center_counts = Counter(all_centers)
            
            print(f"\n📊 EN YAYIN SÖYLEM MERKEZLERİ:")
            for center, count in center_counts.most_common(10):
                print(f"   {center}: {count} kez kullanıldı")
            
            print(f"\n🎭 PATTERN TÜRLERİ:")
            for pattern_id, pattern_data in list(patterns.items())[:3]:
                sequence = pattern_data['sequence']
                frequency = pattern_data['frequency']
                print(f"   {' → '.join(sequence)} (Sıklık: {frequency})")
                
            # Infer sources from centers
            print(f"\n🔍 TAHMİNİ KAYNAK ANALİZİ:")
            tech_words = ['intelligence', 'systems', 'algorithms', 'data', 'technology']
            science_words = ['research', 'scientists', 'discovery', 'analysis']
            academic_words = ['study', 'knowledge', 'understanding', 'theory']
            
            tech_count = sum(center_counts[word] for word in tech_words if word in center_counts)
            science_count = sum(center_counts[word] for word in science_words if word in center_counts)
            academic_count = sum(center_counts[word] for word in academic_words if word in center_counts)
            
            print(f"   Teknoloji metinleri: ~{tech_count} pattern")
            print(f"   Bilimsel metinler: ~{science_count} pattern")  
            print(f"   Akademik metinler: ~{academic_count} pattern")
            
        except Exception as e:
            print(f"❌ Pattern dosyası okunamadı: {e}")
    else:
        print("❌ Pattern dosyası bulunamadı")

def main():
    """Main analysis"""
    analyze_text_sources()
    analyze_pattern_learning_flow()
    show_current_pattern_sources()
    
    print(f"\n🏆 ÖZET: METİN KAYNAKLARI")
    print("=" * 70)
    print("✅ Temel veriset: text_data.txt (Klasik edebiyat)")
    print("✅ Test örnekleri: Hardcoded kaliteli metinler")
    print("✅ Dinamik input: Kullanıcı sağlanan metinler")
    print("✅ Pattern deposu: transition_patterns.json")
    print("✅ Aktif learning: Her analiz ile büyüyor")
    
    print(f"\n🎯 SONUÇ: Pattern'ler çoğunlukla TEST SCRIPTLER içindeki")
    print("   hardcoded kaliteli örneklerden öğreniliyor!")

if __name__ == "__main__":
    main()
