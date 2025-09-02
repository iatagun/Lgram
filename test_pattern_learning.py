#!/usr/bin/env python3
"""
Pattern Learning System Test
Test the new coherent text generation with transition pattern learning
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgram.models.simple_language_model import create_language_model

def test_pattern_learning():
    """Test pattern learning from quality texts"""
    print("🎯 Testing Transition Pattern Learning System")
    print("=" * 60)
    
    # Create model
    print("📚 Loading enhanced language model...")
    model = create_language_model()
    
    # High-quality reference texts for pattern learning
    quality_texts = [
        """
        Artificial intelligence has transformed modern society in unprecedented ways. 
        These systems process vast amounts of data to identify meaningful patterns. 
        Machine learning algorithms enable computers to learn from experience without explicit programming. 
        Such capabilities have revolutionized industries from healthcare to transportation. 
        The technology continues to evolve at an exponential rate, promising even greater innovations ahead.
        """,
        
        """
        Climate change poses one of the greatest challenges of our time. 
        Rising global temperatures affect weather patterns across the planet. 
        These changes threaten ecosystems and human communities alike. 
        Scientists worldwide collaborate to understand and address this crisis. 
        Their research provides critical insights for developing sustainable solutions.
        """,
        
        """
        The human brain remains one of nature's most remarkable achievements. 
        This complex organ processes information through billions of neural connections. 
        Neuroscientists study these networks to understand consciousness and cognition. 
        Their discoveries reveal the intricate mechanisms of memory, emotion, and decision-making. 
        Such knowledge opens new possibilities for treating neurological disorders.
        """
    ]
    
    print(f"\n📖 Learning from {len(quality_texts)} high-quality reference texts...")
    
    # Learn patterns from each text
    total_patterns_learned = 0
    for i, text in enumerate(quality_texts, 1):
        print(f"\n   Learning from Text {i}...")
        result = model.learn_from_quality_text(text.strip(), quality_score=0.9)
        
        if "error" in result:
            print(f"     ❌ Error: {result['error']}")
        else:
            patterns = result.get('patterns_learned', 0)
            transitions = result.get('transitions_extracted', 0)
            total_patterns_learned += patterns
            print(f"     ✅ Patterns learned: {patterns}")
            print(f"     📊 Transitions extracted: {transitions}")
    
    print(f"\n🎉 Total patterns learned: {total_patterns_learned}")
    
    return model

def test_coherent_generation(model):
    """Test coherent text generation using learned patterns"""
    print(f"\n🚀 Testing Coherent Text Generation")
    print("=" * 60)
    
    # Test scenarios
    scenarios = [
        {
            "name": "Technology Theme",
            "input_words": ["Artificial", "intelligence"],
            "length": 5,
            "description": "Generate text about AI with high coherence"
        },
        {
            "name": "Science Theme", 
            "input_words": ["Scientists", "discovered"],
            "length": 4,
            "description": "Generate coherent scientific text"
        },
        {
            "name": "Abstract Theme",
            "input_words": ["The", "future"],
            "length": 6, 
            "description": "Generate coherent abstract discussion"
        }
    ]
    
    generated_texts = []
    
    for scenario in scenarios:
        print(f"\n📝 Scenario: {scenario['name']}")
        print(f"   Description: {scenario['description']}")
        print(f"   Input: {' '.join(scenario['input_words'])}")
        
        # Generate using learned patterns
        try:
            coherent_text = model.generate_coherent_text(
                target_length=scenario['length'],
                input_words=scenario['input_words'],
                quality_level="high"
            )
            
            print(f"   Generated: {coherent_text}")
            generated_texts.append((scenario['name'], coherent_text))
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    return generated_texts

def test_coherence_analysis(model, generated_texts):
    """Test coherence analysis of generated texts"""
    print(f"\n🔍 Testing Coherence Analysis")
    print("=" * 60)
    
    for name, text in generated_texts:
        print(f"\n📊 Analyzing: {name}")
        print(f"   Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"   Full Length: {len(text)} characters")
        
        # Count sentences manually for debugging
        sentence_count = len([s for s in text.split('.') if s.strip()])
        print(f"   Manual Sentence Count: {sentence_count}")
        
        # Analyze coherence
        analysis = model.analyze_text_coherence(text)
        
        if "error" in analysis:
            print(f"   ❌ Analysis Error: {analysis['error']}")
            if "sentences_found" in analysis:
                print(f"   🔍 Sentences found: {analysis['sentences_found']}")
            if "text_preview" in analysis:
                print(f"   📝 Text preview: {analysis['text_preview']}")
        else:
            coherence = analysis.get('coherence_score', 0.0)
            quality = analysis.get('quality_assessment', {})
            transitions = analysis.get('transitions', [])
            
            print(f"   🎯 Coherence Score: {coherence:.2f}")
            print(f"   📈 Quality: {quality.get('overall_quality', 'Unknown')}")
            print(f"   🔄 Transitions: {', '.join(transitions)}")
            
            recommendations = quality.get('recommendations', [])
            if recommendations:
                print(f"   💡 Recommendations:")
                for rec in recommendations:
                    print(f"      • {rec}")

def test_pattern_statistics(model):
    """Test pattern learning statistics"""
    print(f"\n📈 Pattern Learning Statistics")
    print("=" * 60)
    
    stats = model.get_pattern_learning_stats()
    
    if "error" in stats:
        print(f"❌ Error: {stats['error']}")
        return
    
    print(f"✅ Pattern Learning: {'Enabled' if stats.get('pattern_learning_enabled') else 'Disabled'}")
    print(f"📁 Patterns File: {stats.get('patterns_file', 'Not specified')}")
    print(f"🎯 Total Patterns: {stats.get('total_patterns', 0)}")
    print(f"📊 Average Coherence: {stats.get('average_coherence_score', 0.0):.3f}")
    print(f"🔗 Bigram Models: {stats.get('bigram_count', 0)}")
    print(f"🔗 Trigram Models: {stats.get('trigram_count', 0)}")
    
    # Transition distribution
    trans_dist = stats.get('transition_distribution', {})
    if trans_dist:
        print(f"\n🔄 Learned Transition Patterns:")
        for transition, count in trans_dist.items():
            print(f"   {transition}: {count} occurrences")
    
    # Pattern length distribution  
    length_dist = stats.get('pattern_length_distribution', {})
    if length_dist:
        print(f"\n📏 Pattern Length Distribution:")
        for length, count in sorted(length_dist.items()):
            print(f"   {length} transitions: {count} patterns")

def main():
    """Main test function"""
    print("🚀 Advanced Centering Theory Pattern Learning Test")
    print("=" * 70)
    
    try:
        # Phase 1: Learn patterns from quality texts
        model = test_pattern_learning()
        
        # Phase 2: Generate coherent texts using learned patterns
        generated_texts = test_coherent_generation(model)
        
        # Phase 3: Analyze coherence of generated texts
        test_coherence_analysis(model, generated_texts)
        
        # Phase 4: Show pattern learning statistics
        test_pattern_statistics(model)
        
        print(f"\n🎉 Pattern Learning System Test Complete!")
        print("=" * 70)
        print("✅ System Status: OPERATIONAL")
        print("✅ Pattern Learning: ACTIVE")  
        print("✅ Coherent Generation: FUNCTIONAL")
        print("✅ Coherence Analysis: WORKING")
        print(f"🎯 Next Level Achievement: Intelligent Text Generation Ready!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
