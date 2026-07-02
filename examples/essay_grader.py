"""
Essay Cohesion Grader — evaluates essay fluency and structure.
Usage: python examples/essay_grader.py
Requires: pip install centering-lgram && python -m spacy download en_core_web_md
"""

from lgram import TextAnalyzer

ta = TextAnalyzer("en_core_web_md", similarity_threshold=0.35)


def grade_essay(title, text):
    r = ta.analyze(text)
    read = ta.readability_score(text)
    eg = ta.entity_grid_score(text)
    issues = ta.suggest_improvements(text)

    cohesion = r.overall_cohesion
    flesch = read["flesch_reading_ease"]
    cont = r.transition_distribution.get("Continue", 0) * 100
    rough = r.transition_distribution.get("Rough-Shift", 0) * 100

    # Cohesion bar
    cbar = "#" * int(cohesion * 40) + "-" * (40 - int(cohesion * 40))

    # Grade: cohesion-primary, readability-secondary
    # Academic essays naturally have low Flesch — don't penalize
    if cohesion >= 0.80 and rough <= 20:
        letter, desc = "A", "Excellent flow"
    elif cohesion >= 0.70:
        letter, desc = "B", "Good cohesion"
    elif cohesion >= 0.60:
        letter, desc = "C", "Adequate"
    elif cohesion >= 0.50:
        letter, desc = "D", "Weak connections"
    else:
        letter, desc = "F", "Disconnected"

    # Readability warning (not grade)
    if flesch < 20:
        read_note = "(dense/academic)"
    elif flesch > 70:
        read_note = "(very simple)"
    else:
        read_note = ""

    print(f"\n  [{letter}] {title}")
    print(f"  Cohesion:  {cbar} {cohesion:.3f}  ({desc})")
    print(f"  Read:      Flesch={flesch:.0f} {read_note} | Continue: {cont:.0f}% | Rough: {rough:.0f}%")
    print(f"  Metrics:   Entity grid={eg.score:.3f} | Issues: {len(issues)}")

    if issues:
        for s in issues[:2]:
            sev = "!!" if s["severity"] == "high" else "  "
            print(f"    {sev} [{s['index']:02d}] {s['suggestion'][:65]}...")


# ============================================================
# ESSAYS
# ============================================================

essays = {
    "STRONG — Industrial Revolution": (
        "The Industrial Revolution fundamentally changed human society in the eighteenth century. "
        "It introduced mechanized production methods that replaced manual labor. "
        "These new machines dramatically increased manufacturing output and efficiency. "
        "They also created new social classes including the industrial working class. "
        "Factory owners accumulated enormous wealth during this period. "
        "However, workers faced harsh conditions with long hours and low wages. "
        "Child labor became widespread in textile mills and coal mines. "
        "These problems eventually led to labor reforms and union movements. "
        "The revolution also accelerated urbanization as people moved to cities for work. "
        "This migration transformed both the physical landscape and social fabric of society."
    ),
    "STRONG — Romeo and Juliet": (
        "Shakespeare's Romeo and Juliet explores the destructive power of hatred and prejudice. "
        "The play centers on two young lovers from feuding families in Verona. "
        "Romeo and Juliet fall deeply in love despite their families' ancient grudge. "
        "Their secret marriage sets off a tragic chain of events. "
        "Mercutio's death at the hands of Tybalt escalates the conflict dramatically. "
        "Romeo then kills Tybalt in a fit of revenge and rage. "
        "This act leads to his banishment from the city of Verona. "
        "Juliet devises a desperate plan to reunite with her beloved Romeo. "
        "However, miscommunication results in both lovers taking their own lives. "
        "Their deaths finally bring the feuding families together in grief and reconciliation."
    ),
    "MEDIUM — Social Media": (
        "Social media has changed how people communicate with each other today. "
        "Many young people use platforms like Instagram and TikTok every day. "
        "These apps let users share photos and videos with their friends easily. "
        "Social media can help people stay connected across long distances. "
        "But it also has some negative effects on mental health sometimes. "
        "People often compare themselves to others on these platforms. "
        "This comparison can make them feel bad about their own lives. "
        "Cyberbullying is another serious problem on social media sites. "
        "I think social media is both good and bad for society overall. "
        "Parents should monitor their children's social media use more carefully."
    ),
    "MEDIUM — Climate Change": (
        "Climate change is a big problem facing the world right now. "
        "Scientists say the Earth is getting warmer because of greenhouse gases. "
        "These gases come from cars and factories mostly every day. "
        "Rising temperatures are causing ice caps to melt at the poles. "
        "This melting leads to higher sea levels around the world. "
        "Many coastal cities could be underwater in the future eventually. "
        "We need to reduce our carbon emissions as soon as possible. "
        "Renewable energy like solar and wind power can help solve this problem. "
        "Governments should pass laws to limit pollution from companies. "
        "Everyone can also help by making small changes in their daily lives."
    ),
    "WEAK — Environment": (
        "The environment is important for everyone on Earth today. "
        "I like walking in the park near my house after school. "
        "Trees are very tall and green especially in the summer months. "
        "Pollution is bad and it makes the air dirty to breathe. "
        "My family recycles bottles and cans every week at home. "
        "Some people throw trash on the ground which is not good behavior. "
        "Animals live in the forest and they need clean water to survive. "
        "I think we should protect nature because it is beautiful and nice. "
        "The weather has been strange lately with lots of rain and storms. "
        "We learned about environmental science in our biology class last semester."
    ),
    "WEAK — Reading": (
        "Reading books is fun and educational for many different people. "
        "I really enjoy playing video games on my PlayStation after dinner. "
        "My favorite book is about a wizard who goes to magic school. "
        "Video games can improve hand-eye coordination in young children maybe. "
        "Books have pages made of paper and they can be heavy to carry around. "
        "The library near my house has many interesting books to borrow for free. "
        "My friend likes reading comic books more than regular novels usually. "
        "Computers are important for schoolwork and doing research online nowadays. "
        "Reading helps improve vocabulary and writing skills for students everywhere. "
        "I prefer watching movies instead of reading books most of the time honestly."
    ),
}

print("=" * 65)
print("  ESSAY COHESION GRADER (en_core_web_md, threshold=0.35)")
print("=" * 65)

for title, text in essays.items():
    grade_essay(title, text)

# SUMMARY
print(f"\n{'='*65}")
print("  SUMMARY")
print(f"{'='*65}")

for group_prefix, group_name in [("STRONG", "STRONG"), ("MEDIUM", "MEDIUM"), ("WEAK", "WEAK")]:
    group_essays = {k: v for k, v in essays.items() if k.startswith(group_prefix)}
    scores = []
    for text in group_essays.values():
        r = ta.analyze(text)
        scores.append(r.overall_cohesion)
    avg = sum(scores) / len(scores) if scores else 0
    bar = "#" * int(avg * 50)
    rough_pct = sum(
        ta.analyze(t).transition_distribution.get("Rough-Shift", 0) * 100
        for t in group_essays.values()
    ) / len(group_essays)
    print(f"  {group_name:8s} cohesion={avg:.3f}  rough={rough_pct:.0f}%  {bar}")
print()
