"""
Basic text cohesion analysis.
Usage: python examples/basic_analysis.py
"""

from lgram import TextAnalyzer

# Initialize with default model (en_core_web_sm, 12 MB)
# For better results, use TextAnalyzer("en_core_web_md") or TextAnalyzer(use_sentence_transformers=True)
ta = TextAnalyzer()

# Example texts
good = "John went to the store. He bought milk and bread. John paid with cash. Then he walked home."
bad = "John went to the store. Quantum physics is fascinating. Pizza is delicious. Trees are green."

for label, text in [("Good", good), ("Bad", bad)]:
    r = ta.analyze(text)
    print(f"\n{'='*50}")
    print(f"  {label} Text")
    print(f"{'='*50}")
    print(ta.to_summary(r))

# Entity grid comparison
print(f"\n{'='*50}")
print(f"  Entity Grid Scores")
print(f"{'='*50}")
print(f"  Good text: {ta.entity_grid_score(good).score:.3f}")
print(f"  Bad text:  {ta.entity_grid_score(bad).score:.3f}")

# Suggestions for bad text
suggs = ta.suggest_improvements(bad)
if suggs:
    print(f"\n  Fix suggestions for bad text:")
    for s in suggs[:3]:
        print(f"    [{s['severity']}] {s['suggestion'][:70]}...")
