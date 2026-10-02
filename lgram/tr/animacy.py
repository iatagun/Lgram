"""
Animacy cues for Turkish mentions: a small hand-made list, not a lexicon.

On Turkish-ITCC, ranking animate entities before inanimate ones makes the top of the
Cf list a much better predictor of what the next sentence is about (0.68 -> 0.73), and
the parser gives no animacy. ponytail: a few hundred common nouns for people and
animals plus a short gazetteer; swap in a wordnet (KeNet) lookup if coverage matters.
"""

import re
from functools import lru_cache

# people and animals, in dictionary form; inflected forms match too
ANIMATE_NOUNS = """
anne baba ana ata kardeş abla ağabey abi oğul kız çocuk bebek torun dede nine babaanne
anneanne amca dayı hala teyze yeğen kuzen eş koca karı gelin damat enişte yenge akraba
ebeveyn ikiz
insan kişi adam kadın erkek oğlan delikanlı genç yaşlı ihtiyar arkadaş dost düşman
komşu misafir konuk sevgili bey hanım bayan efendi beyefendi hanımefendi birey yabancı
herkes kimse vatandaş yurttaş
öğrenci öğretmen hoca profesör doktor hekim cerrah hemşire hasta avukat hakim savcı
polis jandarma asker subay komutan general albay yüzbaşı binbaşı teğmen şoför pilot
kaptan mühendis mimar yazar şair sanatçı ressam oyuncu şarkıcı müzisyen gazeteci
muhabir editör başkan başbakan cumhurbaşkanı bakan milletvekili vali müdür memur işçi
usta çırak patron işveren çalışan personel müşteri satıcı esnaf tüccar bakkal kasap
berber terzi aşçı garson çiftçi köylü çoban balıkçı avcı hırsız katil lider yönetici
temsilci danışman uzman sekreter üye aday okur okuyucu izleyici seyirci turist yolcu
kral kraliçe prens prenses padişah sultan vezir imam papaz rahip peygamber tanrı melek
şeytan dev cin peri cadı kahraman
aile halk
hayvan köpek kedi at eşek inek öküz koyun kuzu keçi tavuk horoz kuş balık kurt ayı
tilki aslan kaplan fare yılan tavşan kaplumbağa karga serçe güvercin arı sinek böcek
deve fil maymun domuz ördek kaz
""".split()

# proper names are people unless they are one of these (or an all-capitals acronym)
PLACES = """
türkiye istanbul ankara izmir bursa adana antalya konya diyarbakır erzurum trabzon
samsun kayseri gaziantep eskişehir mersin sivas van kars edirne çanakkale muğla bodrum
anadolu trakya karadeniz akdeniz ege marmara kıbrıs avrupa asya afrika amerika
almanya fransa ingiltere italya ispanya yunanistan rusya çin japonya hindistan iran
ırak suriye mısır israil bulgaristan romanya hollanda belçika isveç isviçre avusturya
paris londra berlin roma atina moskova washington brüksel new york bağdat şam kahire
osmanlı cumhuriyet meclis devlet dünya güneş ay
""".split()

# pronouns that only stand for people
HUMAN_PRONOUNS = ("kendi", "kim", "herkes", "kimse", "biri", "birbir", "hepimiz")

_VOWELS = "aeıioöuü"
# an inflected form of a listed noun: plural, possessive, case ("annesi", "çocuklarına")
_SUFFIXES = re.compile(
    r"(l[ae]r)?"
    r"([ıiuü]m|[ıiuü]n|s?[ıiuü]|m|n|[ıiuü]?m[ıiuü]z|[ıiuü]?n[ıiuü]z|l[ae]r[ıi])?"
    r"(n?[ıiuü]|n?[ae]|y[ıiuüae]|n?[dt][ae]n?|n?[ıiuü]n|y?l[ae])?"
)
_SOFTENED = {"k": "ğ", "t": "d", "p": "b", "ç": "c"}  # çocuk -> çocuğu, kurt -> kurdu


def noun_matcher(words):
    """A function telling whether a lower-cased noun form (no apostrophe suffix) is an
    inflected form of one of `words`: "annesi", "çocuklarına", "kurdu".

    The snowball stemmer cannot be used for this: it leaves "annes" for "annesi".
    """
    words = frozenset(words)
    soft = {
        w[:-1] + _SOFTENED[w[-1]] for w in words if w[-1] in _SOFTENED and len(w) > 2
    }
    forms = words | soft | ({"oğl"} if "oğul" in words else set())  # oğul -> oğlu

    @lru_cache(maxsize=50_000)
    def match(base: str) -> bool:
        for k in range(len(base), 1, -1):
            stem, rest = base[:k], base[k:]
            if stem not in forms or not _SUFFIXES.fullmatch(rest):
                continue
            if not rest:
                return stem in words
            # a suffix vowel only follows a consonant, a buffer consonant only a
            # vowel: "kaz" + "an" is no inflection of "kaz", nor "eş" + "ya" of "eş"
            if stem[-1] in _VOWELS and rest[0] in _VOWELS:
                continue
            if stem[-1] not in _VOWELS and rest[0] in "ysn":
                continue
            return True
        return False

    return match


is_animate_noun = noun_matcher(ANIMATE_NOUNS)
