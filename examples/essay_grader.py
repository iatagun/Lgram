"""
Essay Cohesion Analyzer — calibrated with empirical genre thresholds.

Key findings (Brown Corpus, n=30/genre, HIGH confidence):
- Rough-Shift >50% is abnormal for ANY genre
- Narrative: highest Continue rate (50%) — stories are protagonist-centered
- Single Rough-Shift is NOT a problem — only consecutive + no Cb is broken
"""

from lgram import TextAnalyzer

ta = TextAnalyzer("en_core_web_md", similarity_threshold=0.35)

# Empirically calibrated thresholds (Brown Corpus, n=30, Tukey's fence)
GENRE_THRESHOLDS = {
    "narrative": {
        "rough_normal": (0.115, 0.273),  # p25-p75
        "flag_above": 0.510,              # Tukey upper
        "expected_continue": 0.50,
        "note": "Stories are protagonist-centered — high Continue expected",
    },
    "expository": {
        "rough_normal": (0.167, 0.333),
        "flag_above": 0.582,
        "expected_continue": 0.29,
        "note": "News/academic — moderate entity shifts between topics",
    },
    "essay": {
        "rough_normal": (0.115, 0.277),
        "flag_above": 0.520,
        "expected_continue": 0.31,
        "note": "Opinion pieces — balanced between staying on topic and introducing arguments",
    },
}


def analyze(title, text, genre="essay"):
    r = ta.analyze(text)
    eg = ta.entity_grid_score(text)
    issues = ta.suggest_improvements(text)

    cohesion = r.overall_cohesion
    dist = r.transition_distribution
    cont = dist.get("Continue", 0) * 100
    rough = dist.get("Rough-Shift", 0) * 100

    g = GENRE_THRESHOLDS.get(genre, GENRE_THRESHOLDS["essay"])
    rough_min, rough_max = g["rough_normal"]
    rough_normal = rough_min <= rough / 100 <= rough_max
    rough_flagged = rough / 100 > g["flag_above"]

    # Only truly broken = consecutive no-Cb Rough-Shifts
    truly_broken = [s for s in issues if s["issue"] in ("no_backward_center", "consecutive_rough_shifts")]

    # Overall assessment
    if rough_flagged:
        verdict = "FLAGGED — Rough-Shift exceeds Tukey threshold"
    elif len(truly_broken) >= 2:
        verdict = "REVIEW — minor cohesion breaks detected"
    else:
        verdict = "OK — normal for this genre"

    cbar = "#" * int(cohesion * 40) + "-" * (40 - int(cohesion * 40))

    print(f"\n  {title} [{genre}]")
    print(f"  Cohesion:  {cbar} {cohesion:.3f}  |  {verdict}")
    print(f"  Rough: {rough:.0f}%  (normal={rough_min*100:.0f}-{rough_max*100:.0f}%, "
          f"flag>{g['flag_above']*100:.0f}%)  |  Continue: {cont:.0f}%  "
          f"(expected ~{g['expected_continue']*100:.0f}%)")
    print(f"  Entity grid: {eg.score:.3f}  |  Broken links: {len(truly_broken)}")
    if g["note"]:
        print(f"  Note: {g['note']}")

    for s in truly_broken[:2]:
        print(f"    !! [{s.get('index', '?')}] {s['suggestion'][:65]}...")


# ============================================================
print("=" * 75)
print("  ESSAY COHESION ANALYZER — Calibrated (Brown Corpus, n=30)")
print("=" * 75)

essays = [
    ("Industrial Revolution", """The Industrial Revolution fundamentally changed human society in the eighteenth century. It introduced mechanized production methods that replaced manual labor. These new machines dramatically increased manufacturing output and efficiency. They also created new social classes including the industrial working class. Factory owners accumulated enormous wealth during this period. However, workers faced harsh conditions with long hours and low wages. Child labor became widespread in textile mills and coal mines. These problems eventually led to labor reforms and union movements. The revolution also accelerated urbanization as people moved to cities for work. This migration transformed both the physical landscape and social fabric of society.""", "expository"),
    ("Romeo and Juliet", """Shakespeare's Romeo and Juliet explores the destructive power of hatred and prejudice. The play centers on two young lovers from feuding families in Verona. Romeo and Juliet fall deeply in love despite their families' ancient grudge. Their secret marriage sets off a tragic chain of events. Mercutio's death at the hands of Tybalt escalates the conflict dramatically. Romeo then kills Tybalt in a fit of revenge and rage. This act leads to his banishment from the city of Verona. Juliet devises a desperate plan to reunite with her beloved Romeo. However, miscommunication results in both lovers taking their own lives. Their deaths finally bring the feuding families together in grief and reconciliation.""", "narrative"),
    ("Social Media", """Social media has changed how people communicate with each other today. Many young people use platforms like Instagram and TikTok every day. These apps let users share photos and videos with their friends easily. Social media can help people stay connected across long distances. But it also has some negative effects on mental health sometimes. People often compare themselves to others on these platforms. This comparison can make them feel bad about their own lives. Cyberbullying is another serious problem on social media sites. I think social media is both good and bad for society overall. Parents should monitor their children's social media use more carefully.""", "essay"),
    ("Climate Change", """Climate change is a big problem facing the world right now. Scientists say the Earth is getting warmer because of greenhouse gases. These gases come from cars and factories mostly every day. Rising temperatures are causing ice caps to melt at the poles. This melting leads to higher sea levels around the world. Many coastal cities could be underwater in the future eventually. We need to reduce our carbon emissions as soon as possible. Renewable energy like solar and wind power can help solve this problem. Governments should pass laws to limit pollution from companies. Everyone can also help by making small changes in their daily lives.""", "essay"),
    ("Broken thoughts", """Technology is great. I really like my computer a lot. Computers are very useful for many things. The weather today is sunny and warm outside. My favorite subject is mathematics and science. I go to school every day except weekends. Many people use smartphones for social media. I think schools should have more computers. Homework is sometimes boring but necessary. My friends and I study together after school.""", "essay"),
]

for title, text, genre in essays:
    analyze(title, text, genre)

# Summary
print(f"\n{'='*75}")
print("  SUMMARY")
print(f"{'='*75}")

for gname in ["expository", "narrative", "essay"]:
    same = [(t, tx) for t, tx, g in essays if g == gname]
    if len(same) < 2:
        continue
    print(f"\n  {gname.upper()}:")
    scores = []
    for title, text in same:
        r = ta.analyze(text)
        rough = r.transition_distribution.get("Rough-Shift", 0) * 100
        scores.append((title, r.overall_cohesion, rough))
    scores.sort(key=lambda x: x[1], reverse=True)
    thresh = GENRE_THRESHOLDS[gname]
    for title, s, rough in scores:
        bar = "#" * int(s * 40)
        flag = " <<<" if rough > thresh["flag_above"] * 100 else ""
        print(f"    {title:25s} {bar} {s:.3f}  rough={rough:.0f}%{flag}")
print()
