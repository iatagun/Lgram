"""
Essay Cohesion Analyzer — feedback, not grading.
Compares essays within the same genre only.
"""

from lgram import TextAnalyzer

ta = TextAnalyzer("en_core_web_md", similarity_threshold=0.35)


def analyze(title, text, genre="essay"):
    r = ta.analyze(text)
    eg = ta.entity_grid_score(text)
    read = ta.readability_score(text)
    issues = ta.suggest_improvements(text)

    cohesion = r.overall_cohesion
    dist = r.transition_distribution
    cont = dist.get("Continue", 0)
    ret = dist.get("Retain", 0)
    smooth = dist.get("Smooth-Shift", 0)
    rough = dist.get("Rough-Shift", 0)

    # Dominant transition pattern
    patterns = {"Continue": cont, "Retain": ret, "Smooth-Shift": smooth, "Rough-Shift": rough}
    dominant = max(patterns, key=patterns.get)

    # Genre expectations
    genre_map = {
        "narrative": {"dominant": "Rough-Shift", "rough_range": (0.25, 0.60)},
        "expository": {"dominant": "Continue", "rough_range": (0.05, 0.30)},
        "argumentative": {"dominant": "Smooth-Shift", "rough_range": (0.15, 0.40)},
        "essay": {"dominant": "Continue", "rough_range": (0.10, 0.40)},
    }
    g = genre_map.get(genre, genre_map["essay"])
    rough_ok = g["rough_range"][0] <= rough <= g["rough_range"][1]

    # Only flag TRULY broken text
    truly_broken = []
    for s in issues:
        if s["issue"] == "no_backward_center":
            truly_broken.append(s)
        elif s["issue"] == "consecutive_rough_shifts":
            truly_broken.append(s)

    cbar = "#" * int(cohesion * 40) + "-" * (40 - int(cohesion * 40))

    print(f"\n  {title} [{genre}]")
    print(f"  Cohesion:   {cbar} {cohesion:.3f}")
    print(f"  Dominant:   {dominant}  (expected for {genre}: {g['dominant']})")
    print(f"  Rough-Shift: {rough*100:.0f}%  {'(normal)' if rough_ok else '(unexpected)'}")
    print(f"  Continue:   {cont*100:.0f}%  Retain: {ret*100:.0f}%  Smooth: {smooth*100:.0f}%")
    print(f"  Entity grid: {eg.score:.3f}  Issues: {len(truly_broken)} broken")

    for s in truly_broken[:2]:
        print(f"    !! [{s.get('index', '?')}] {s['suggestion'][:65]}...")


# ============================================================
# ESSAYS — now with GENRE labels
# ============================================================

essays = [
    # EXPOSITORY
    ("Industrial Revolution", """The Industrial Revolution fundamentally changed human society in the eighteenth century.
It introduced mechanized production methods that replaced manual labor.
These new machines dramatically increased manufacturing output and efficiency.
They also created new social classes including the industrial working class.
Factory owners accumulated enormous wealth during this period.
However, workers faced harsh conditions with long hours and low wages.
Child labor became widespread in textile mills and coal mines.
These problems eventually led to labor reforms and union movements.
The revolution also accelerated urbanization as people moved to cities for work.
This migration transformed both the physical landscape and social fabric of society.""", "expository"),

    # NARRATIVE
    ("Romeo and Juliet", """Shakespeare's Romeo and Juliet explores the destructive power of hatred and prejudice.
The play centers on two young lovers from feuding families in Verona.
Romeo and Juliet fall deeply in love despite their families' ancient grudge.
Their secret marriage sets off a tragic chain of events.
Mercutio's death at the hands of Tybalt escalates the conflict dramatically.
Romeo then kills Tybalt in a fit of revenge and rage.
This act leads to his banishment from the city of Verona.
Juliet devises a desperate plan to reunite with her beloved Romeo.
However, miscommunication results in both lovers taking their own lives.
Their deaths finally bring the feuding families together in grief and reconciliation.""", "narrative"),

    # ESSAY (general)
    ("Social Media", """Social media has changed how people communicate with each other today.
Many young people use platforms like Instagram and TikTok every day.
These apps let users share photos and videos with their friends easily.
Social media can help people stay connected across long distances.
But it also has some negative effects on mental health sometimes.
People often compare themselves to others on these platforms.
This comparison can make them feel bad about their own lives.
Cyberbullying is another serious problem on social media sites.
I think social media is both good and bad for society overall.
Parents should monitor their children's social media use more carefully.""", "essay"),

    # ESSAY (general)
    ("Climate Change", """Climate change is a big problem facing the world right now.
Scientists say the Earth is getting warmer because of greenhouse gases.
These gases come from cars and factories mostly every day.
Rising temperatures are causing ice caps to melt at the poles.
This melting leads to higher sea levels around the world.
Many coastal cities could be underwater in the future eventually.
We need to reduce our carbon emissions as soon as possible.
Renewable energy like solar and wind power can help solve this problem.
Governments should pass laws to limit pollution from companies.
Everyone can also help by making small changes in their daily lives.""", "essay"),

    # TRULY BROKEN (no genre expectation — should fail everywhere)
    ("Broken thoughts", """Technology is great.
I really like my computer a lot.
Computers are very useful for many things.
The weather today is sunny and warm outside.
My favorite subject is mathematics and science.
I go to school every day except weekends.
Many people use smartphones for social media.
I think schools should have more computers.
Homework is sometimes boring but necessary.
My friends and I study together after school.""", "essay"),
]

print("=" * 70)
print("  ESSAY COHESION ANALYZER — Genre-Aware")
print("=" * 70)

for title, text, genre in essays:
    analyze(title, text, genre)


# COMPARISON — only within same genre
print(f"\n{'='*70}")
print("  WITHIN-GENRE COMPARISON")
print(f"{'='*70}")

for genre_name in ["expository", "narrative", "essay"]:
    same_genre = [(t, tx) for t, tx, g in essays if g == genre_name]
    if len(same_genre) < 2:
        continue

    print(f"\n  {genre_name.upper()}:")
    scores = []
    for title, text in same_genre:
        r = ta.analyze(text)
        scores.append((title, r.overall_cohesion))
    scores.sort(key=lambda x: x[1], reverse=True)
    for title, s in scores:
        bar = "#" * int(s * 40)
        print(f"    {title:25s} {bar} {s:.3f}")

print()
