centering-lgram v2.0'ı paylaşmaktan mutluluk duyuyorum — Centering Theory'yi (Grosz, Joshi & Weinstein, 1983/1995) temel alan, söylem tutarlılığını ölçmeye yarayan bir Python kütüphanesi.

Centering Theory, tümceler arasındaki konu akışını üç merkez üzerinden modeller: İleri Merkezler (Cf), Geri Merkez (Cb) ve Tercih Edilen Merkez (Cp). Bu merkezler arasındaki ilişki dört geçiş tipiyle sınıflandırılır: Continue, Retain, Smooth-Shift ve Rough-Shift. Hesaplamalı söylem çözümlemesinin en köklü kuramlarından biri olmasına rağmen, erişilebilir ve kullanıma hazır uygulamaları oldukça sınırlıydı.

**Kütüphanenin sundukları:**

- Gramer rolü, sözcük türü, konum ve varlık tipine dayalı, yapılandırılabilir ağırlıklandırma ile ileri/geri/tercih edilen merkez hesaplaması
- Kişi, nesne, çoğul ve iyelik zamirlerini kapsayan, bileşik öncül tespiti içeren kademeli zamir çözümlemesi
- Dört kanonik geçiş tipine ek olarak söylem başlangıcı için Establish tipi
- Geçiş dağılımlarının ağırlıklı ortalamasıyla tutarlılık puanlaması
- Bağımlılık çözümlemesi tabanlı iç tümce ayırma ile tümce-içi çözümleme (sıralı, zarf, tamlayıcı ve ilgi tümcecikleri)
- Birleşik tümceler-arası ve tümce-içi analiz
- spaCy dışında hiçbir bağımlılığı olmayan komut satırı arayüzü

**Teknik detaylar:**

- Tek bağımlılık: `spacy>=3.4.0`
- 49 test, kapsamlı uç durum ve yanlış-pozitif denetimi
- Durumdan bağımsız analiz (mevcut durumu bozmadan değerlendirme)
- MIT lisanslı, Python 3.8+

**Kurulum:** `pip install centering-lgram`

**Depo:** https://github.com/iatagun/Lgram

Bu sürüm, daha önceki çalışmaları istatistiksel dil modeli bileşenlerinden arındırarak, yalnızca Centering Theory'ye odaklanmış, akademik temelli bir söylem analiz aracı haline getiriyor.

Hesaplamalı dilbilim, NLP ve ilgili alanlarda çalışan araştırmacıların geri bildirimlerini, katkılarını ve tartışmalarını bekliyorum.

#hesaplamalıDilbilim #NLP #söylemÇözümlemesi #centeringTheory #python #açıkKaynak #dilbilim