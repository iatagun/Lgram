TransitionAnalyzer Sınıfı
TransitionAnalyzer, bir metindeki cümleler arasındaki geçişleri analiz etmek için tasarlanmış bir sınıftır. Bu sınıf, metni cümlelere ayırarak cümle çiftleri oluşturur ve ardından her cümle çifti arasındaki isim tamlamalarını ve geçiş türlerini belirler.

Özellikler:
*Önişleme: Metni cümlelere ayırarak cümle çiftleri oluşturur.
*İsim Tamlamaları Çıkarma: Her cümledeki isim tamlamalarını çıkarır.
*Geçiş Sınıflandırma: Cümleler arasındaki geçişleri sınıflandırarak daha iyi anlamayı sağlar (örneğin, merkez kurma, devam etme, yeni konu geçişi).
*Anaforik İlişkileri Etiketleme: Cümlelerdeki anaforik ilişkileri tanımlayarak bağlamı güçlendirir.
Kullanım:
Sınıfın analyze metodu, metni analiz eder ve her cümle çifti için geçiş bilgilerini içeren bir liste döndürür. Her bir sonuç, cümlelerin içeriği, geçiş türü, isim tamlamaları ve anaforik ilişkileri hakkında detaylı bilgi sunar.