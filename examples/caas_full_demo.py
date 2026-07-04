"""
CAEAS Full Demo — tüm özelliklerle örnek essay analizi.
"""

from lgram.essay import (
    CAEASGrader, Essay,
    PreFilter, L1TransferAnalyzer,
    CEFRCalibrator, DataExporter, ErrorTypology,
    estimate_cefr_level,
)

print("=" * 72)
print("  CAEAS — Cohesion-Aware Writing Feedback Tool")
print("  EFL (Turkish L1)  |  CEFR B2 odaklı")
print("=" * 72)

# ---------------------------------------------------------------------------
# Örnek: Orta seviye bir EFL öğrencisinin yazdığı tipik bir essay
# ---------------------------------------------------------------------------
essay_text = """Social Media and Young People

Social media is very important for young people today. Is a big part of daily life. Many students use platforms like Instagram and TikTok every day. These applications help people share photos and videos with their friends easily. She can also meet new people online.

However, social media also has negative effects on mental health sometimes. For example, some studies show that comparison with others make people feel sad. This comparison can make them feel bad about their own lives. Person spends too much time scrolling and this affect sleep quality.

I think social media is both good and bad for society. Parents should monitor their children's social media use more carefully. Also schools can teach digital literacy to students. We need find balance between online and offline life."""

essay = Essay(title="Social Media and Young People", text=essay_text)

# ---------------------------------------------------------------------------
# 1. TEMEL ANALİZ (EFL modu varsayılan)
# ---------------------------------------------------------------------------
print("\n" + "-" * 72)
print("  1. TEMEL ANALİZ  (EFL modu, CEFR oto-tespit, L1=tr)")
print("-" * 72)

grader = CAEASGrader(l1_language="tr")

report = grader.analyze(essay)

print(f"\n  CEFR seviyesi  : {report.cefr_level or 'auto-detected'} (detected: {report.cefr_detected})")
print(f"  Kohezyon       : {report.overall_cohesion_indicator:.0f}/100")
print(f"  Güven aralığı  : {report.confidence_interval[0]:.0f} – {report.confidence_interval[1]:.0f}")
print(f"  Öneri          : {report.suggestion}")
print(f"  Sınırda        : {report.borderline}")
print(f"  Öğretmen review: {report.teacher_review_recommended}")

print(f"\n  Boyut bazında:")
for lr in report.layer_results:
    ci = lr.confidence_interval
    print(f"    {lr.layer_name:35s} {lr.score:5.0f}/100  CI: [{ci[0]:.0f}–{ci[1]:.0f}]")

# ---------------------------------------------------------------------------
# 2. GERİ BİLDİRİM (justification — öğretmene sunulacak)
# ---------------------------------------------------------------------------
print("\n" + "-" * 72)
print("  2. ÖĞRETMENE SUNULACAK GERİ BİLDİRİM")
print("-" * 72)
print(report.justification)

# ---------------------------------------------------------------------------
# 3. PREFILTER — Grammar/Kohezyon ayrımı
# ---------------------------------------------------------------------------
print("\n" + "-" * 72)
print("  3. PREFILTER  (grammar vs cohesion ayrımı)")
print("-" * 72)

pf = PreFilter()
pf_report = pf.analyze(essay_text)
print(f"\n  LanguageTool  : {'mevcut' if pf_report.language_tool_available else 'heuristic mod'}")
print(f"  Parse güveni  : {pf_report.parse_confidence:.0%}")
print(f"  Grammar hata  : {len(pf_report.grammar_issues)}")
print(f"  Kohezyon risk : {len(pf_report.cohesion_risks)}")
print(f"  Kritik mi     : {pf_report.has_critical_grammar_issues}")

if pf_report.grammar_issues:
    print(f"\n  Grammar hataları (ilk 3):")
    for g in pf_report.grammar_issues[:3]:
        print(f"    [{g.get('type','?')}] {g.get('description','')[:90]}")
        if g.get('l1_transfer'):
            print(f"      L1: {g['l1_transfer']}")

if pf_report.recommendations:
    print(f"\n  Öneriler:")
    for r in pf_report.recommendations:
        print(f"    -> {r[:100]}")

# ---------------------------------------------------------------------------
# 4. L1 TRANSFER ANALİZİ
# ---------------------------------------------------------------------------
print("\n" + "-" * 72)
print("  4. L1 TRANSFER ANALİZİ  (Türkçe → İngilizce)")
print("-" * 72)

l1a = L1TransferAnalyzer()
l1_report = l1a.analyze(essay_text)
print(f"\n  Transfer skoru : {l1_report.overall_transfer_score:.2f} (1.0 = minimal)")
print(f"\n  Pro-drop       : {len(l1_report.pro_drop_issues)} tespit")
for p in l1_report.pro_drop_issues[:2]:
    print(f"    -> {p.get('description','')[:90]}")
print(f"\n  Gender         : {len(l1_report.gender_pronoun_issues)} tespit")
for g in l1_report.gender_pronoun_issues:
    print(f"    -> {g.get('description','')[:100]}")
print(f"\n  Article        : adequacy={l1_report.article_estimate.get('adequacy',0):.2f}")
print(f"    ratio={l1_report.article_estimate.get('article_ratio',0):.3f}")
print(f"    note={l1_report.article_estimate.get('note','')}")
print(f"\n  Word order     : {len(l1_report.word_order_issues)} tespit")
print(f"\n  {l1_report.summary}")

# ---------------------------------------------------------------------------
# 5. CEFR KALİBRASYON PROFİLİ
# ---------------------------------------------------------------------------
print("\n" + "-" * 72)
print("  5. CEFR B2 PROFİLİ")
print("-" * 72)

from lgram.essay import get_cefr_profile
b2 = get_cefr_profile("B2")
print(f"\n  Seviye         : B2 — {b2['label']}")
print(f"  Beklenen aralık: {b2['expected_score_range'][0]}–{b2['expected_score_range'][1]}/100")
print(f"  Kohezyon eşiği : {b2['cohesion_threshold']}")
print(f"  Rough-Shift tol: {b2['rough_shift_tolerance']}")
print(f"  Min kelime      : {b2['min_length_words']}")
print(f"  Tipik sorunlar  :")
for issue in b2['typical_issues']:
    print(f"    • {issue}")

# ---------------------------------------------------------------------------
# 6. KOMPLEKSİTE ANALİZİ
# ---------------------------------------------------------------------------
print("\n" + "-" * 72)
print("  6. KOMPLEKSİTE ANALİZİ  (syntactic complexity)")
print("-" * 72)

cal = CEFRCalibrator()
cp = cal.assess_complexity(essay_text)
print(f"\n  Cümle sayısı   : {cp.sentence_count}")
print(f"  Ort. cümle uz. : {cp.avg_sentence_length} kelime")
print(f"  Clause marker  : {cp.total_clause_markers}")
print(f"  Subord. oranı  : {cp.subordination_ratio:.3f}")
print(f"  Kelime çeşit.  : {cp.vocabulary_diversity:.3f}")
print(f"  Seviye         : {cp.complexity_level}")
print(f"  Ayarlama fakt. : {cp.adjustment_factor}  (>1.0 = tolerans bonusu)")

# ---------------------------------------------------------------------------
# 7. HATA TİPOLOJİSİ
# ---------------------------------------------------------------------------
print("\n" + "-" * 72)
print("  7. HATA TİPOLOJİSİ")
print("-" * 72)

typology = ErrorTypology()
typology.feed(report, cefr_level=report.cefr_level or "B2")
typo_report = typology.build()
print(f"\n  {typo_report.summary}")
print(f"\n  En sık kategori: {typo_report.most_common_category}")
print(f"  L1 transfer oranı: {typo_report.l1_transfer_ratio:.0%}")

# ---------------------------------------------------------------------------
# 8. EXPORT (JSON)
# ---------------------------------------------------------------------------
print("\n" + "-" * 72)
print("  8. VERİ EXPORT'U")
print("-" * 72)

exporter = DataExporter()
bundle = exporter.create_bundle([essay], [report])

print(f"\n  Toplam essay    : {bundle.total_essays}")
print(f"  CEFR dağılımı   : {bundle.cefr_distribution}")
print(f"\n  Export JSON önizleme (ilk kayıt):")
import json
rec = bundle.reports[0]
print(f"  {json.dumps(rec, indent=2, ensure_ascii=False)[:400]}...")

# ---------------------------------------------------------------------------
# 9. BACKWARD COMPATIBILITY
# ---------------------------------------------------------------------------
print("\n" + "-" * 72)
print("  9. BACKWARD COMPAT (eski API)")
print("-" * 72)

old_report = grader.grade(essay)  # deprecated alias
print(f"\n  grader.grade()        : {'çalışıyor' if old_report else 'hata'}")
print(f"  report.overall_score   : {old_report.overall_score:.0f}  (alias)")
print(f"  report.verdict         : {old_report.verdict[:60]}...  (alias)")
print(f"  report.human_review_rec: {old_report.human_review_recommended}  (alias)")

# ---------------------------------------------------------------------------
# 10. KARŞILAŞTIRMA: İYİ vs KÖTÜ ESSAY
# ---------------------------------------------------------------------------
print("\n" + "-" * 72)
print("  10. KARŞILAŞTIRMA: İyi vs Kötü Essay")
print("-" * 72)

good = Essay(title="İyi Essay", text=(
    "Technology has transformed education in many important ways. "
    "First, online learning platforms have made education accessible to "
    "students in remote areas. For example, students in rural regions "
    "can now attend virtual classes from top universities. "
    "Second, digital tools enable personalized learning experiences. "
    "Teachers can track individual student progress and adjust their "
    "instruction accordingly. However, technology also presents challenges. "
    "Not all students have reliable internet access, which creates a "
    "digital divide. Furthermore, excessive screen time may affect "
    "students' concentration and health. In conclusion, while technology "
    "offers significant benefits for education, schools must address "
    "equity concerns and establish healthy usage guidelines."
))

bad = Essay(title="Kötü Essay", text=(
    "Technology is good. I like computers. My neighbor has a dog. "
    "The weather today is sunny. Schools should have more technology. "
    "Pizza is my favorite food. Students use the internet for research."
))

good_r = grader.analyze(good)
bad_r = grader.analyze(bad)

print(f"\n  {'Essay':25s} {'Kohezyon':>10s} {'CI':>14s} {'Review':>8s}")
print(f"  {'-'*60}")
print(f"  {'İyi':25s} {good_r.overall_cohesion_indicator:>8.0f}/100  "
      f"({good_r.confidence_interval[0]:.0f}–{good_r.confidence_interval[1]:.0f})  "
      f"{'EVET' if good_r.teacher_review_recommended else 'hayır':>6s}")
print(f"  {'Kötü':25s} {bad_r.overall_cohesion_indicator:>8.0f}/100  "
      f"({bad_r.confidence_interval[0]:.0f}–{bad_r.confidence_interval[1]:.0f})  "
      f"{'EVET' if bad_r.teacher_review_recommended else 'hayır':>6s}")

print(f"\n  İyi essay önerisi  : {good_r.suggestion[:90]}...")
print(f"  Kötü essay önerisi : {bad_r.suggestion[:90]}...")

print("\n" + "=" * 72)
print("  CAEAS Demo Tamamlandı. 159 test, 5 feature modülü.")
print("=" * 72)
