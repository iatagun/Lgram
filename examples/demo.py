"""
CAEAS Full Demo - Tum ozelliklerle ornek essay analizi.
"""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgram.essay import (
    CAEASGrader, Essay,
    PreFilter, L1TransferAnalyzer,
    CEFRCalibrator, DataExporter, ErrorTypology,
    get_cefr_profile,
)

def sep(title): 
    print(f"\n{'='*70}\n  {title}\n{'='*70}")

# ---------------------------------------------------------------------------
sep("CAEAS - Ornek EFL Essay Analizi (Turkish L1, CEFR B2)")

essay_text = """Social Media and Young People

Social media is very important for young people today. Is a big part of daily life. Many students use platforms like Instagram and TikTok every day. These applications help people share photos and videos with their friends easily. She can also meet new people online.

However, social media also has negative effects on mental health sometimes. For example, some studies show that comparison with others make people feel sad. This comparison can make them feel bad about their own lives. Person spends too much time scrolling and this affect sleep quality.

I think social media is both good and bad for society. Parents should monitor their children's social media use more carefully. Also schools can teach digital literacy to students. We need find balance between online and offline life."""

essay = Essay(title="Social Media and Young People", text=essay_text)
grader = CAEASGrader(l1_language="tr")

# ---------------------------------------------------------------------------
sep("1. TEMEL ANALIZ (EFL modu, CEFR oto-tespit, L1=tr)")

report = grader.analyze(essay)
print(f"  CEFR seviyesi  : {report.cefr_level or 'auto'} (detected: {report.cefr_detected})")
print(f"  Kohezyon skoru : {report.cohesion_score:.0f}/100  (pure Layer 2 cohesion)")
print(f"  Kompozit       : {report.composite_indicator:.0f}/100  (content+cohesion+surface)")
print(f"  Guven araligi  : {report.confidence_interval[0]:.0f} - {report.confidence_interval[1]:.0f}")
print(f"  Oneri          : {report.suggestion}")
print(f"  Sinirda        : {report.borderline}")
print(f"  Ogretmen review: {report.teacher_review_recommended}")
print()
for lr in report.layer_results:
    ci = lr.confidence_interval
    print(f"    {lr.layer_name:35s} {lr.score:5.0f}/100  CI: [{ci[0]:.0f}-{ci[1]:.0f}]")

# ---------------------------------------------------------------------------
sep("2. OGRETMENE SUNULACAK GERI BILDIRIM")
print(report.justification)

# ---------------------------------------------------------------------------
sep("3. PREFILTER (grammar vs cohesion ayrimi)")

pf = PreFilter()
pf_report = pf.analyze(essay_text)
print(f"  LanguageTool  : {'mevcut' if pf_report.language_tool_available else 'heuristic mod'}")
print(f"  Parse guveni  : {pf_report.parse_confidence:.0%}")
print(f"  Grammar hata  : {len(pf_report.grammar_issues)}")
print(f"  Kohezyon risk : {len(pf_report.cohesion_risks)}")
print(f"  Kritik mi     : {pf_report.has_critical_grammar_issues}")
for g in pf_report.grammar_issues[:3]:
    print(f"    [{g.get('type','?')}] {g.get('description','')[:90]}")
    if g.get('l1_transfer'): print(f"      L1: {g['l1_transfer']}")

# ---------------------------------------------------------------------------
sep("4. L1 TRANSFER ANALIZI (Turkce -> Ingilizce)")

l1a = L1TransferAnalyzer()
l1r = l1a.analyze(essay_text)
print(f"  Transfer skoru : {l1r.overall_transfer_score:.2f} (1.0 = minimal)")
print(f"  Pro-drop       : {len(l1r.pro_drop_issues)} tespit")
for p in l1r.pro_drop_issues[:1]:
    print(f"    -> {p.get('description','')[:90]}")
print(f"  Gender         : {len(l1r.gender_pronoun_issues)} tespit")
for g in l1r.gender_pronoun_issues:
    print(f"    -> {g.get('description','')[:100]}")
print(f"  Article        : adequacy={l1r.article_estimate.get('adequacy',0):.2f}")
print(f"    note={l1r.article_estimate.get('note','')}")
print(f"  {l1r.summary}")

# ---------------------------------------------------------------------------
sep("5. KOMPLEKSITE ANALIZI")

cal = CEFRCalibrator()
cp = cal.assess_complexity(essay_text)
print(f"  Cumle sayisi   : {cp.sentence_count}")
print(f"  Ort. cumle uz. : {cp.avg_sentence_length} kelime")
print(f"  Subord. orani  : {cp.subordination_ratio:.3f}")
print(f"  Kelime cesit.  : {cp.vocabulary_diversity:.3f}")
print(f"  Seviye         : {cp.complexity_level}")
print(f"  Ayarlama fakt. : {cp.adjustment_factor}  (>1.0 = tolerans bonusu)")

# ---------------------------------------------------------------------------
sep("6. HATA TIPOLOJISI")

typology = ErrorTypology()
typology.feed(report, cefr_level=report.cefr_level or "B2")
tr = typology.build()
print(f"  {tr.summary}")
print(f"  En sik kategori: {tr.most_common_category}")

# ---------------------------------------------------------------------------
sep("7. VERI EXPORT (JSON)")

exporter = DataExporter()
bundle = exporter.create_bundle([essay], [report])
print(f"  Toplam essay: {bundle.total_essays}, CEFR: {bundle.cefr_distribution}")
rec = bundle.reports[0]
print(f"  {{\"cohesion_score\": {rec['cohesion_score']}, \"composite_indicator\": {rec['composite_indicator']}, \"cefr\": \"{rec['estimated_cefr']}\"}}")

# ---------------------------------------------------------------------------
sep("8. BACKWARD COMPAT (eski API)")

old = grader.grade(essay)
print(f"  grader.grade()           : calisiyor")
print(f"  report.overall_score     : {old.overall_score:.0f}  -> cohesion_score")
print(f"  report.verdict[:50]      : {old.verdict[:50]}...")
print(f"  report.overall_cohesion  : {old.overall_cohesion_indicator:.0f}")

# ---------------------------------------------------------------------------
sep("9. DISKRIMINANT GECERLILIK: Iyi vs Kotu Essay")

good = Essay(title="Iyi Essay (ders kitabi duzeyinde)", text=(
    "Technology has transformed education in many important ways. "
    "First, online learning platforms have made education accessible to "
    "students in remote areas. For example, students in rural regions "
    "can now attend virtual classes from top universities. "
    "Second, digital tools enable personalized learning experiences. "
    "Teachers can track individual student progress and adjust their "
    "instruction accordingly. However, technology also presents challenges. "
    "Not all students have reliable internet access, which creates a "
    "digital divide. Furthermore, excessive screen time may affect "
    "students concentration and health. In conclusion, while technology "
    "offers significant benefits for education, schools must address "
    "equity concerns and establish healthy usage guidelines."
))

bad = Essay(title="Kotu Essay (ilgisiz cumleler)", text=(
    "Technology is good. I like computers. My neighbor has a dog. "
    "The weather today is sunny. Schools should have more technology. "
    "Pizza is my favorite food. Students use the internet for research."
))

gr = grader.analyze(good)
br = grader.analyze(bad)

print(f"  {'Essay':35s} {'Cohesion':>10s} {'Kompozit':>10s} {'CI':>14s}")
print(f"  {'-'*72}")
print(f"  {'Iyi (ders kitabi)':35s} {gr.cohesion_score:>8.0f}/100  {gr.composite_indicator:>8.0f}/100  ({gr.confidence_interval[0]:.0f}-{gr.confidence_interval[1]:.0f})")
print(f"  {'Kotu (ilgisiz)':35s} {br.cohesion_score:>8.0f}/100  {br.composite_indicator:>8.0f}/100  ({br.confidence_interval[0]:.0f}-{br.confidence_interval[1]:.0f})")
print(f"\n  FARK: cohesion={gr.cohesion_score - br.cohesion_score:.0f} puan, composite={gr.composite_indicator - br.composite_indicator:.0f} puan")

# Layer 2 details
gl2, bl2 = gr.layer_results[1], br.layer_results[1]
for name, l2 in [("Iyi", gl2), ("Kotu", bl2)]:
    for s in l2.raw_details.get("segments", []):
        print(f"  {name} seg {s['index']} ({s['type']:12s}): cohesion={s['cohesion']:.2f}  rough_shift={s['rough_shift_ratio']:.2f}  continue={s['continue_ratio']:.2f}")

print(f"\n  Iyi:  avg_continue={gl2.raw_details.get('avg_continue_ratio', 0):.3f}  avg_rough={gl2.raw_details.get('avg_rough_shift_ratio', 0):.3f}")
print(f"  Kotu: avg_continue={bl2.raw_details.get('avg_continue_ratio', 0):.3f}  avg_rough={bl2.raw_details.get('avg_rough_shift_ratio', 0):.3f}")

print(f"\n{'='*70}")
print(f"  CAEAS Demo Tamamlandi. 162 test, 6 feature modulu.")
print(f"{'='*70}")
