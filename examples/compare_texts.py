"""
Compare original vs revised text cohesion.
Usage: python examples/compare_texts.py
"""

from lgram import TextAnalyzer

ta = TextAnalyzer()

original = (
    "Tesla reported record quarterly earnings. The electric vehicle maker "
    "posted revenue of 25 billion. CEO Elon Musk credited strong demand in China. "
    "The company warned about supply chain issues. Its stock rose 3%. "
    "Analysts remain optimistic about the company's outlook. "
    "They cited Tesla's expanding factory network as a key advantage."
)

revised = (
    "Tesla reported strong earnings this quarter with 25 billion in revenue. "
    "CEO Elon Musk credited Chinese demand for the results. "
    "Despite supply chain warnings, the stock rose 3% in after-hours trading."
)

diff = ta.diff_cohesion(original, revised)
print("Diff Analysis: Original vs Revised")
print(f"  Original: {diff['original_score']:.3f} [{diff['original_quality']}]")
print(f"  Revised:  {diff['revised_score']:.3f} [{diff['revised_quality']}]")
print(f"  Delta:    {diff['delta']:+.3f}  ->  {diff['verdict'].upper()}")
print(f"  Continue change:  {diff['continue_delta']:+.3f}")
print(f"  Rough-Shift change: {diff['rough_shift_delta']:+.3f}")

print("\nRanking comparison:")
r = ta.compare_texts(original, revised, labels=["Original", "Revised"])
for entry in r["rankings"]:
    bar = "#" * int(entry["cohesion"] * 40)
    print(f"  {entry['label']:12s} {bar} {entry['cohesion']:.3f} [{entry['quality']}]")
