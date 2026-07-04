"""
Essay Cohesion Analyzer — multi-dimensional with calibrated verdict system.

Decision layer (unchanged — Brown Corpus calibrated):
  1. Centering cohesion + genre thresholds → verdict (OK / REVIEW / FLAGGED)
  2. Broken link count → severity (minor < 4, significant >= 4)
  3. Rough-Shift % vs Tukey upper → FLAGGED trigger

Supplementary layers (informational only):
  4. Entity Grid — role persistence (Barzilay & Lapata 2005)
  5. Discourse — connective analysis (PDTB-lite)
  6. Cache — avoids re-computation on repeated texts

NOTE: Supplementary scores do NOT affect verdicts. The verdict system
is empirically calibrated from Brown Corpus (n=30/genre, Tukey's fence).
Adding uncalibrated weights would break that.
"""

from lgram import TextAnalyzer
from lgram.cache import AnalysisCache
from lgram.discourse import DiscourseAnalyzer

ta = TextAnalyzer("en_core_web_md", similarity_threshold=0.35)
cache = AnalysisCache(max_size=64, default_ttl=3600)
da = DiscourseAnalyzer(ta.nlp)

GENRE_THRESHOLDS = {
    "narrative": {
        "rough_normal": (0.115, 0.273),
        "flag_above": 0.510,
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
    cached = cache.get(text)
    if cached:
        cohesion, rough, verdict, cont, eg_score, dr_score = cached
        cache_hit = True
    else:
        r = ta.analyze(text)
        eg = ta.entity_grid_score(text)
        dr = da.analyze(text)
        issues = ta.suggest_improvements(text)

        cohesion = r.overall_cohesion
        dist = r.transition_distribution
        rough_pct = dist.get("Rough-Shift", 0)
        cont = dist.get("Continue", 0) * 100
        g = GENRE_THRESHOLDS.get(genre, GENRE_THRESHOLDS["essay"])
        rough_flagged = rough_pct > g["flag_above"]
        truly_broken = [
            s for s in issues
            if s["issue"] in ("no_backward_center", "consecutive_rough_shifts")
        ]

        if rough_flagged:
            verdict = "FLAGGED"
        elif len(truly_broken) >= 4:
            verdict = "REVIEW (significant)"
        elif len(truly_broken) >= 2:
            verdict = "REVIEW (minor)"
        else:
            verdict = "OK"

        rough = rough_pct * 100
        eg_score = eg.score
        dr_score = dr.cohesion_score

        cache.set(text, (cohesion, rough, verdict, cont, eg_score, dr_score))
        cache_hit = False

    cbar = "#" * int(cohesion * 40) + "-" * (40 - int(cohesion * 40))

    if cache_hit:
        print(f"\n  {title} [{genre}]  [cached]")

    print(f"\n  {title} [{genre}]")
    print(f"  {'-' * 65}")
    print(f"  Cohesion:    {cbar} {cohesion:.3f}")
    print(f"  Verdict:     {verdict}")
    print(f"  Rough: {rough:.0f}%  |  Continue: {cont:.0f}%")
    if not cache_hit:
        print(f"  --- supplementary (not used in verdict) ---")
        print(f"  Entity Grid: {eg_score:.3f}  "
              f"|  Discourse: {dr_score:.3f}")

    return cohesion, rough


# ============================================================
print("=" * 75)
print("  ESSAY COHESION ANALYZER — Calibrated Verdict System")
print("  Decision: Centering + Genre Thresholds + Broken Links")
print("  Info:    Entity Grid + Discourse + Cache")
print("=" * 75)

essays = [
    ("Industrial Revolution", """The Industrial Revolution fundamentally changed human society in the eighteenth century. It introduced mechanized production methods that replaced manual labor. These new machines dramatically increased manufacturing output and efficiency. They also created new social classes including the industrial working class. Factory owners accumulated enormous wealth during this period. However, workers faced harsh conditions with long hours and low wages. Child labor became widespread in textile mills and coal mines. These problems eventually led to labor reforms and union movements. The revolution also accelerated urbanization as people moved to cities for work. This migration transformed both the physical landscape and social fabric of society.""", "expository"),
    ("Romeo and Juliet", """Shakespeare's Romeo and Juliet explores the destructive power of hatred and prejudice. The play centers on two young lovers from feuding families in Verona. Romeo and Juliet fall deeply in love despite their families' ancient grudge. Their secret marriage sets off a tragic chain of events. Mercutio's death at the hands of Tybalt escalates the conflict dramatically. Romeo then kills Tybalt in a fit of revenge and rage. This act leads to his banishment from the city of Verona. Juliet devises a desperate plan to reunite with her beloved Romeo. However, miscommunication results in both lovers taking their own lives. Their deaths finally bring the feuding families together in grief and reconciliation.""", "narrative"),
    ("Social Media", """Social media has changed how people communicate with each other today. Many young people use platforms like Instagram and TikTok every day. These apps let users share photos and videos with their friends easily. Social media can help people stay connected across long distances. But it also has some negative effects on mental health sometimes. People often compare themselves to others on these platforms. This comparison can make them feel bad about their own lives. Cyberbullying is another serious problem on social media sites. I think social media is both good and bad for society overall. Parents should monitor their children's social media use more carefully.""", "essay"),
    ("Climate Change", """Climate change is a big problem facing the world right now. Scientists say the Earth is getting warmer because of greenhouse gases. These gases come from cars and factories mostly every day. Rising temperatures are causing ice caps to melt at the poles. This melting leads to higher sea levels around the world. Many coastal cities could be underwater in the future eventually. We need to reduce our carbon emissions as soon as possible. Renewable energy like solar and wind power can help solve this problem. Governments should pass laws to limit pollution from companies. Everyone can also help by making small changes in their daily lives.""", "essay"),
    ("Broken thoughts", """Technology is great. I really like my computer a lot. Computers are very useful for many things. The weather today is sunny and warm outside. My favorite subject is mathematics and science. I go to school every day except weekends. Many people use smartphones for social media. I think schools should have more computers. Homework is sometimes boring but necessary. My friends and I study together after school.""", "essay"),
    ("Random sentences", """Quantum physics explains subatomic behavior. My neighbor has a loud barking dog. The Roman Empire fell in 476 CE. I enjoy eating pizza on Friday evenings. Basketball requires good eye-hand coordination. The ocean contains millions of species. Beethoven composed nine symphonies. Cats are independent animals. The stock market fluctuates daily. Mountains are formed by tectonic activity.""", "essay"),
]

scores_by_genre = {}
for title, text, genre in essays:
    coh, rough = analyze(title, text, genre)
    scores_by_genre.setdefault(genre, []).append((title, coh, rough))

# Second pass — cache should now hit
print(f"\n{'=' * 75}")
print("  CACHE VERIFICATION — re-running to test hits")
print(f"{'=' * 75}")
print(f"  Before: {cache.stats['hits']} hits / {cache.stats['misses']} misses  "
      f"|  size: {cache.stats['size']}")

for title, text, genre in essays:
    analyze(title, text, genre)

print(f"\n  After:  {cache.stats['hits']} hits / {cache.stats['misses']} misses  "
      f"|  hit rate: {cache.hit_rate:.0%}")

# Summary
print(f"\n{'=' * 75}")
print("  SUMMARY (sorted by cohesion)")
print(f"{'=' * 75}")

for gname in ["expository", "narrative", "essay"]:
    items = scores_by_genre.get(gname, [])
    if len(items) < 2:
        continue
    print(f"\n  {gname.upper()}:")
    items.sort(key=lambda x: x[1], reverse=True)
    thresh = GENRE_THRESHOLDS[gname]
    for title, coh, rough in items:
        bar = "#" * int(coh * 40)
        flag = " <<<" if rough > thresh["flag_above"] * 100 else ""
        print(f"    {title:25s} {bar} {coh:.3f}  rough={rough:.0f}%{flag}")

print()
