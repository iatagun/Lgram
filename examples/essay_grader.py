"""
Essay Cohesion Grader — evaluates essay fluency and structure.
Usage: python examples/essay_grader.py
Requires: pip install centering-lgram && python -m spacy download en_core_web_md
"""

from lgram import TextAnalyzer

# Use medium model (40 MB, GloVe vectors) for best accuracy
ta = TextAnalyzer("en_core_web_md", similarity_threshold=0.35)


def grade_essay(title: str, text: str):
    r = ta.analyze(text)
    read = ta.readability_score(text)
    combined = ta.combined_score(text)
    eg = ta.entity_grid_score(text)
    issues = ta.suggest_improvements(text)

    cont = r.transition_distribution.get("Continue", 0) * 100
    rough = r.transition_distribution.get("Rough-Shift", 0) * 100
    bar = "#" * int(r.overall_cohesion * 40) + "-" * (40 - int(r.overall_cohesion * 40))

    # Grade: combined score based
    if combined >= 0.70:
        grade, letter = "Excellent", "A"
    elif combined >= 0.60:
        grade, letter = "Good", "B"
    elif combined >= 0.50:
        grade, letter = "Adequate", "C"
    elif combined >= 0.40:
        grade, letter = "Weak", "D"
    else:
        grade, letter = "Poor", "F"

    print(f"\n  {title}")
    print(f"  {'-'*50}")
    print(f"  Cohesion:    {bar} {r.overall_cohesion:.3f}")
    print(f"  Readability: Flesch={read['flesch_reading_ease']:.0f}  Combined={combined:.3f}")
    print(f"  Continue: {cont:.0f}%  Rough-Shift: {rough:.0f}%  Issues: {len(issues)}")
    print(f"  Entity grid: {eg.score:.3f}  Entities: {', '.join(eg.entities[:4])}")
    print(f"  Grade: {letter} ({grade})")

    if issues:
        for s in issues[:3]:
            sev = "!!" if s["severity"] == "high" else "  "
            print(f"    {sev} [{s['index']:02d}] {s['suggestion'][:65]}...")


# ============================================================
# ESSAYS
# ============================================================

essay_a = """Technology has transformed education in the twenty-first century.
It has made learning more accessible to students worldwide.
Digital platforms provide interactive lessons and instant feedback.
These tools help students learn at their own pace and style.
Teachers also benefit from technology in the classroom.
They can track student progress with automated grading systems.
These systems save time and allow more personalized instruction.
However technology should not replace human teachers entirely.
Students still need guidance and emotional support from educators.
The best approach combines technology with traditional teaching methods."""

essay_b = """Technology is important in education today.
Many students use computers and tablets for learning.
Online courses are becoming very popular now.
Teachers also use technology for grading assignments.
I think technology helps students learn better.
But sometimes technology can be distracting in class.
Students might play games instead of studying.
Some schools do not have enough computers for everyone.
Technology can be expensive for school districts.
I believe technology is good but has some problems too."""

essay_c = """Technology is great.
I really like my computer a lot.
Computers are very useful for many things.
Education is very important for everyone in society.
I go to school every day except weekends.
My favorite subject is mathematics and science.
Technology changes very fast these days.
Many people use smartphones for social media.
I think schools should have more computers.
The weather today is sunny and warm outside.
Homework is sometimes boring but necessary.
My friends and I study together after school."""


print("=" * 60)
print("  ESSAY COHESION GRADER (en_core_web_md)")
print("=" * 60)

for title, text in [
    ("A — Good structure", essay_a),
    ("B — Simple but on-topic", essay_b),
    ("C — Disconnected thoughts", essay_c),
]:
    grade_essay(title, text)

# RANKING
print(f"\n{'='*60}")
print("  FINAL RANKING")
print(f"{'='*60}")

batch = ta.analyze_batch(
    [essay_a, essay_b, essay_c],
    labels=["Essay A", "Essay B", "Essay C"],
)
for entry in batch["rankings"]:
    bar = "#" * int(entry["cohesion"] * 50)
    combined = ta.combined_score(
        [essay_a, essay_b, essay_c][["Essay A", "Essay B", "Essay C"].index(entry["label"])]
    )
    print(f"  {entry['label']:10s} {bar} {entry['cohesion']:.3f}  combined={combined:.3f}")

print(f"\n  Mean cohesion: {batch['mean_cohesion']:.3f}")
print()
