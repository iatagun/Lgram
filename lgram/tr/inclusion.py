"""
Inclusion without a possessive: "Ev güzeldi. Kapı maviydi." (part of a whole).

A first, hand-made list, enough to label the obvious cases. ponytail: a dozen wholes;
replace with a wordnet's part-of relation (KeNet) or a trained linker when the labels
are used for more than a diagnostic.
"""

from .animacy import noun_matcher

# whole -> parts, dictionary forms
PARTS = {
    "ev": "kapı pencere duvar oda mutfak salon banyo balkon çatı bahçe merdiven tavan",
    "apartman": "daire kapı merdiven asansör çatı bodrum",
    "oda": "kapı pencere duvar tavan zemin",
    "araba": "tekerlek lastik motor direksiyon kapı cam koltuk bagaj fren",
    "okul": "sınıf bahçe kantin koridor öğretmen öğrenci müdür",
    "köy": "sokak meydan ev cami tarla çeşme",
    "şehir": "sokak cadde meydan mahalle park",
    "kitap": "sayfa kapak bölüm",
    "ağaç": "dal yaprak kök gövde",
    "gemi": "güverte kaptan yelken dümen",
    "tren": "vagon makinist",
    "masa": "ayak çekmece",
}
_MATCH = {w: (noun_matcher([w]), noun_matcher(p.split())) for w, p in PARTS.items()}


def is_part_of(part: str, whole: str) -> bool:
    """`part`, `whole`: lower-cased noun forms ("kapısı" is not needed here: a possessed
    part is found through its possessor). "kapı" is part of "evin"; "kapı" of "kapı" not.
    """
    return any(
        is_whole(whole) and is_part(part) for is_whole, is_part in _MATCH.values()
    )
