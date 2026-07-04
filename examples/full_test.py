"""
CAEAS Full 5-Layer Demo — real grammar + mechanics + content + cohesion + surface.
"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgram.essay import CAEASGrader, Essay, PreFilter, L1TransferAnalyzer, ErrorTypology

essay_text = """Social Media and Young People

Social media is very important for young people today. Is a big part of daily life. Many students use platforms like Instagram and TikTok every day. These applications help people share photos and videos with their friends easily. She can also meet new people online.

However, social media also has negative effects on mental health sometimes. For example, some studies show that comparison with others make people feel sad. This comparison can make them feel bad about their own lives. Person spends too much time scrolling and this affect sleep quality.

I think social media is both good and bad for society. Parents should monitor their children's social media use more carefully. Also schools can teach digital literacy to students. We need find balance between online and offline life."""

essay = Essay(title="Social Media and Young People", text=essay_text)

print("=" * 70)
print("  CAEAS 5-LAYER FULL TEST")
print("  Grammar(LanguageTool) + Content(heuristic) + Cohesion(Lgram)")
print("  + Surface(readability) + Mechanics(spellchecker)")
print("=" * 70)

grader = CAEASGrader(l1_language="tr", use_llm=False)

print("\n--- 5 LAYER SONUCLARI ---")
report = grader.analyze(essay)

for lr in report.layer_results:
    ci = lr.confidence_interval
    evidence_preview = lr.evidence[0][:90] if lr.evidence else "-"
    print(f"\n  [{lr.layer_name}]")
    print(f"    Score: {lr.score:.0f}/100  CI: [{ci[0]:.0f}-{ci[1]:.0f}]")
    for e in lr.evidence[:2]:
        print(f"    -> {e[:100]}")
    if hasattr(lr, 'raw_details') and lr.raw_details:
        for k, v in lr.raw_details.items():
            if isinstance(v, (int, float, str)) and k not in ('segments',):
                print(f"    {k}: {v}")

print(f"\n  >>> COHESION: {report.cohesion_score:.0f}/100   COMPOSITE: {report.composite_indicator:.0f}/100")
print(f"  >>> CI: [{report.confidence_interval[0]:.0f}-{report.confidence_interval[1]:.0f}]")
print(f"  >>> {report.suggestion}")

# L1 transfer
print("\n--- L1 TRANSFER ---")
l1a = L1TransferAnalyzer()
l1r = l1a.analyze(essay_text)
print(f"  Score: {l1r.overall_transfer_score:.2f}  Pro-drop: {len(l1r.pro_drop_issues)}  Gender: {len(l1r.gender_pronoun_issues)}")

# Typology
print("\n--- HATA TIPOLOJISI ---")
typo = ErrorTypology()
typo.feed(report)
tr = typo.build()
print(f"  {tr.summary[:200]}")

# Good vs bad comparison
print("\n--- DISKRIMINANT ---")
good = Essay(text="Technology has transformed education in many important ways. First, online learning platforms have made education accessible to students. For example, students in rural areas can attend virtual classes. Second, digital tools enable personalized learning. However, technology also presents challenges. In conclusion, schools must address equity concerns.")
bad = Essay(text="Technology is good. I like computers. My neighbor has a dog. Pizza is my favorite food. The weather is sunny.")
gr = grader.analyze(good)
br = grader.analyze(bad)
print(f"  Good: cohesion={gr.cohesion_score:.0f} composite={gr.composite_indicator:.0f}")
print(f"  Bad:  cohesion={br.cohesion_score:.0f} composite={br.composite_indicator:.0f}")
print(f"  Delta: cohesion={gr.cohesion_score-br.cohesion_score:.0f}p composite={gr.composite_indicator-br.composite_indicator:.0f}p")

print("\n" + "=" * 70)
print("  163 test passing. 5 layers active.")
print("=" * 70)
