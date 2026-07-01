centering-lgram v2.0'ı paylaşmaktan mutluluk duyuyorum — Centering Theory'yi (Grosz, Joshi & Weinstein, 1983/1995) temel alan, söylem bağdaşıklığını ölçmeye yarayan bir Python kütüphanesi.

**Dilbilimsel not:** Bağdaşıklık (cohesion) ve tutarlılık (coherence) farklı kavramlardır. Bağdaşıklık zamirler, bağlaçlar, sözcüksel tekrarlar gibi yüzeysel dilbilgisel bağlantılarla ilgiliyken; tutarlılık metnin anlamsal/kavramsal bütünlüğüne işaret eder. Centering Theory özünde bir bağdaşıklık kuramıdır — tümceler arasındaki varlık takibini modeller.

Centering Theory, tümceler arasındaki varlık akışını üç merkez üzerinden modeller: İleri Merkezler (Cf), Geri Merkez (Cb) ve Tercih Edilen Merkez (Cp). Bu merkezler arasındaki ilişki dört geçiş tipiyle sınıflandırılır: Continue, Retain, Smooth-Shift ve Rough-Shift. Hesaplamalı söylem çözümlemesinin en köklü kuramlarından biri olmasına rağmen, erişilebilir ve kullanıma hazır uygulamaları oldukça sınırlıydı.

**Kütüphanenin sundukları:**

- Gramer rolü, sözcük türü, konum ve varlık tipine dayalı, yapılandırılabilir ağırlıklandırma ile ileri/geri/tercih edilen merkez hesaplaması
- Kişi, nesne, çoğul ve iyelik zamirlerini kapsayan, bileşik öncül tespiti içeren kademeli zamir çözümlemesi
- Dört kanonik geçiş tipine ek olarak söylem başlangıcı için Establish tipi
- Geçiş dağılımlarının ağırlıklı ortalamasıyla bağdaşıklık puanlaması
- Bağımlılık çözümlemesi tabanlı iç tümce ayırma ile tümce-içi çözümleme
- Birleşik tümceler-arası ve tümce-içi analiz
- spaCy dışında hiçbir bağımlılığı olmayan CLI

**Teknik detaylar:**

- Tek bağımlılık: `spacy>=3.4.0`
- 49 test, kapsamlı uç durum ve yanlış-pozitif denetimi
- MIT lisanslı, Python 3.8+

**Kurulum:** `pip install centering-lgram`

**Depo:** https://github.com/iatagun/Lgram

Bu sürüm, daha önceki çalışmaları istatistiksel dil modeli bileşenlerinden arındırarak, yalnızca Centering Theory'ye odaklanmış, dilbilimsel temelli bir söylem bağdaşıklığı analiz aracı haline getiriyor.

Hesaplamalı dilbilim, NLP ve ilgili alanlarda çalışan araştırmacıların geri bildirimlerini ve katkılarını bekliyorum.

#hesaplamalıDilbilim #NLP #söylemÇözümlemesi #bağdaşıklık #centeringTheory #python #açıkKaynak #dilbilim