"""
Evaluate LLM output quality.
Usage: python examples/llm_evaluation.py
"""

from lgram import TextAnalyzer

# Use medium model for best accuracy
ta = TextAnalyzer("en_core_web_md")

# Simulated LLM outputs
responses = {
    "Good blog post": (
        "Artificial intelligence has transformed healthcare in remarkable ways. "
        "It now enables faster and more accurate diagnosis of diseases. "
        "These AI systems can analyze medical images with precision. "
        "They often detect conditions earlier than human radiologists. "
        "However, the technology also raises important ethical questions. "
        "Many experts worry about potential bias in AI training data. "
        "They argue that regulations are needed to ensure fairness."
    ),
    "Hallucinated output": (
        "AI is very important for the future. "
        "The weather today is quite sunny and warm. "
        "I enjoy eating pizza on Friday evenings. "
        "Machine learning requires large datasets. "
        "My cat likes to sleep on the windowsill. "
        "Neural networks have many layers. "
        "The bus was late this morning."
    ),
}

ta_llm = TextAnalyzer()
for name, text in responses.items():
    r = ta_llm.analyze_llm(text)
    bar = "=" * int(r["response_cohesion"] * 30)
    print(f"\n{name}:")
    print(f"  Cohesion: {bar} {r['response_cohesion']:.3f} [{r['quality']}]")
    print(f"  Segments: {r['response_segments']}")
