# en_transitions_gold.csv — etiketleme kılavuzu

20 paragraf (10 Wikipedia, 10 Grimm), her birinde ilk 6 cümle. 116 cümle, 96 geçiş.
Kaynak: İngilizce Wikipedia (CC BY-SA 4.0, makale paragrafları) ve Grimm masalları
(Project Gutenberg #2591, kamu malı).
Her paragraf bağımsızdır; paragrafın ilk cümlesi (`idx=0`) etiketlenmez.

Her cümle için yalnızca iki sütunu doldur. Geçiş türünü yazma, script onu türetir.

## gold_cp — tercih edilen merkez (Cp)

Cümledeki varlıklardan dilbilgisel role göre **en üst sıradaki**:

    özne > dolaylı nesne > nesne > diğerleri (edat tümleci, iyelik...)

- Zamir ise gönderdiği varlığı yaz (`he` → `king`).
- Yan cümlede değil, ana cümlenin öznesinde olanı tercih et.
- Varlık yoksa (ör. "It was raining.") `-` yaz.

## gold_cb — geri-bakan merkez (Cb)

Önceki cümlenin varlıkları (Cf) arasından **bu cümlede de geçen, önceki cümlede en üst
sırada olanı** yaz. Geçiş doğrudan tekrar, zamir ya da açık eş-gönderge ("the animal" →
`wolf`) ile olabilir.

- Önceki cümleyle ortak hiçbir varlık yoksa `-` yaz → NOCB.
- Yalnızca bir önceki cümleye bak, daha gerisine değil.

## Adlandırma

- Varlığın **baş adını küçük harfle** yaz: "the old king" → `king`, "Louis II of Hungary" → `louis`.
- Aynı varlık için paragraf boyunca **aynı adı** kullan; eşleşmeyi script string eşitliğiyle
  yapıyor (`king` ≠ `kings`).
- Emin olmadığın satıra `note` sütununa kısa not düş.

## Örnek

| cümle | gold_cp | gold_cb | türetilen geçiş |
|---|---|---|---|
| John went to the store. | - | - | (ilk cümle) |
| He bought milk. | john | john | Continue |
| The milk was sour. | milk | milk | Smooth-Shift |
| It rained all day. | - | - | NOCB |

## Değerlendirme

    python -m lgram.transition_eval

Sistem aynı cümleler üzerinde canlı çalıştırılır; geçiş doğruluğu, Cb var/yok uyumu ve
karışıklık matrisi yazdırılır. Eksik satır varsa ilk eksik id'yi söyler.
