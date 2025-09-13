"""
🌟 Basit Blog Yazısı Tamamlayıcı
Centering-LGram v1.1.2 kullanarak

Bu örnek, başka bir projede centering-lgram'ı nasıl 
kullanabileceğinizi gösterir.
"""

import sys
from lgram.models.simple_language_model import create_language_model

class BlogCompletioner:
    """Blog yazılarını tamamlayan basit sınıf"""
    
    def __init__(self):
        print("📚 Blog Tamamlayıcı başlatılıyor...")
        self.model = create_language_model()
        print("✅ Model hazır!")
    
    def complete_blog_post(self, title, intro, target_sentences=5):
        """Blog yazısını tamamla"""
        print(f"\n📝 Blog Yazısı Tamamlanıyor...")
        print(f"📰 Başlık: {title}")
        print(f"🎯 Giriş: {intro}")
        print("-" * 50)
        
        # Giriş metininden kelimeler al
        intro_words = intro.strip().split()[-3:]  # Son 3 kelime
        
        # Centering theory ile devam ettir
        continuation = self.model.generate_text_with_centering(
            input_words=intro_words,
            num_sentences=target_sentences,
            length=15,
            use_progress_bar=False
        )
        
        # Sonuçları birleştir
        full_post = f"{intro} {continuation}"
        
        print("📄 Tamamlanan Yazı:")
        print(full_post)
        print("-" * 50)
        
        return full_post

def main():
    """Ana fonksiyon"""
    print("🌟 Blog Tamamlayıcı Demo")
    print("=" * 40)
    
    # Blog tamamlayıcıyı başlat
    completer = BlogCompletioner()
    
    # Test blog yazıları
    blog_examples = [
        {
            "title": "Yapay Zeka ve Gelecek",
            "intro": "Yapay zeka teknolojisi hızla gelişiyor. İnsanlık tarihinde bu kadar önemli bir dönüm noktası nadiren yaşanmıştır.",
            "sentences": 3
        },
        {
            "title": "Doğa ve İnsan",
            "intro": "Doğa ile insan arasındaki bağ her geçen gün zayıflıyor. Şehirleşme ile birlikte doğadan uzaklaşıyoruz.",
            "sentences": 4
        },
        {
            "title": "Sanat ve Teknoloji",
            "intro": "Modern sanat artık teknoloji ile iç içe. Dijital eserler geleneksel sanatın yerini alıyor.",
            "sentences": 3
        }
    ]
    
    # Her örneği tamamla
    for i, blog in enumerate(blog_examples, 1):
        print(f"\n🔷 Örnek {i}")
        completer.complete_blog_post(
            title=blog["title"],
            intro=blog["intro"],
            target_sentences=blog["sentences"]
        )
    
    print("\n✅ Tüm blog yazıları tamamlandı!")
    print("💡 Kendi projelerinizde benzer şekilde kullanabilirsiniz!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️ İşlem kullanıcı tarafından durduruldu")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        sys.exit(1)
