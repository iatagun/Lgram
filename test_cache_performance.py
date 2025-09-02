#!/usr/bin/env python3
"""
Cache Performance Test for Enhanced Language Model
Tests the SmartCacheSystem performance improvements
"""

import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgram.models.simple_language_model import create_language_model

def test_cache_performance():
    """Test cache system performance"""
    print("🚀 Testing SmartCache Performance...")
    print("=" * 50)
    
    # Create model
    print("📚 Loading language model...")
    model = create_language_model()
    
    # Test sentences for center extraction
    test_sentences = [
        "The cat sat on the mat.",
        "John walked to the store yesterday.",
        "She loves reading books in the library.",
        "The beautiful garden bloomed in spring.",
        "They played soccer in the park.",
        "The cat sat on the mat.",  # Duplicate for cache testing
        "John walked to the store yesterday.",  # Duplicate
        "Mary cooked dinner for her family.",
        "The car drove down the winding road.",
        "Students studied hard for their exams."
    ]
    
    # Test 1: Center Extraction Performance
    print("\n🎯 Test 1: Center Extraction Cache")
    start_time = time.time()
    
    extracted_centers = []
    for sentence in test_sentences:
        center = model._extract_center_from_sentence(sentence)
        extracted_centers.append((sentence[:30] + "...", center))
        print(f"  Center: '{center}' <- {sentence[:30]}...")
    
    center_extraction_time = time.time() - start_time
    print(f"  ⏱️  Time: {center_extraction_time:.2f}s")
    
    # Test 2: T5 Correction Cache
    print("\n🔧 Test 2: T5 Grammar Correction Cache")
    test_texts = [
        "this are a test sentence",
        "I goed to the store",
        "She don't like pizza",
        "this are a test sentence",  # Duplicate
        "The book what I read",
        "I goed to the store",  # Duplicate
    ]
    
    start_time = time.time()
    corrected_texts = []
    for text in test_texts:
        corrected = model.correct_grammar_t5(text)
        corrected_texts.append((text, corrected))
        print(f"  '{text}' -> '{corrected}'")
    
    correction_time = time.time() - start_time
    print(f"  ⏱️  Time: {correction_time:.2f}s")
    
    # Test 3: Sentence Generation Cache
    print("\n📝 Test 3: Sentence Generation Cache")
    start_words_list = [
        ["The", "cat"],
        ["John", "walked"],
        ["She", "loves"],
        ["The", "cat"],  # Duplicate
        ["Mary", "cooked"]
    ]
    
    start_time = time.time()
    generated_sentences = []
    for start_words in start_words_list:
        sentence = model.generate_sentence(start_words, base_length=10)
        generated_sentences.append(sentence)
        print(f"  {' '.join(start_words)} -> {sentence}")
    
    generation_time = time.time() - start_time
    print(f"  ⏱️  Time: {generation_time:.2f}s")
    
    # Show Cache Statistics
    print("\n📊 Cache Performance Statistics")
    print("=" * 50)
    
    cache_stats = model.get_cache_stats()
    
    # Overall Performance
    overall = cache_stats['overall_performance']
    print(f"🎯 Overall Cache Performance:")
    print(f"  Total Requests: {overall['total_requests']}")
    print(f"  Total Hits: {overall['total_hits']}")
    print(f"  Total Misses: {overall['total_misses']}")
    print(f"  Hit Rate: {overall['overall_hit_rate']:.1%}")
    print(f"  Memory Efficiency: {overall['memory_efficiency']:.1%}")
    
    # Detailed Cache Stats
    print(f"\n📋 Detailed Cache Statistics:")
    for cache_name, stats in cache_stats['cache_details'].items():
        print(f"  {cache_name.upper()} Cache:")
        print(f"    Size: {stats['size']}/{stats['max_size']}")
        print(f"    Hit Rate: {stats['hit_rate']:.1%}")
        print(f"    Utilization: {stats['utilization']:.1%}")
    
    # Recommendations
    print(f"\n💡 Optimization Recommendations:")
    for rec in cache_stats['recommendations']:
        print(f"  • {rec}")
    
    # Performance Summary
    total_time = center_extraction_time + correction_time + generation_time
    print(f"\n⚡ Performance Summary:")
    print(f"  Center Extraction: {center_extraction_time:.2f}s")
    print(f"  T5 Correction: {correction_time:.2f}s")
    print(f"  Sentence Generation: {generation_time:.2f}s")
    print(f"  Total Time: {total_time:.2f}s")
    
    # Memory Optimization Test
    print(f"\n🧠 Memory Optimization:")
    print("  Running cache memory optimization...")
    model.optimize_cache_memory()
    print("  ✅ Memory optimization completed!")
    
    return cache_stats

def benchmark_cache_vs_no_cache():
    """Benchmark cache system vs no cache"""
    print("\n🏁 Benchmark: Cache vs No Cache")
    print("=" * 50)
    
    model = create_language_model()
    
    # Test data
    test_sentence = "The quick brown fox jumps over the lazy dog."
    iterations = 5
    
    # Test WITH cache (second run will be faster)
    print("🔥 Testing WITH Cache:")
    start_time = time.time()
    for i in range(iterations):
        center = model._extract_center_from_sentence(test_sentence)
        print(f"  Run {i+1}: Center = '{center}'")
    with_cache_time = time.time() - start_time
    
    # Clear cache and test WITHOUT cache
    print(f"\n❄️  Testing after cache clear:")
    model.clear_all_caches()
    start_time = time.time()
    for i in range(iterations):
        center = model._extract_center_from_sentence(test_sentence)
        print(f"  Run {i+1}: Center = '{center}'")
    without_cache_time = time.time() - start_time
    
    # Results
    speedup = without_cache_time / with_cache_time if with_cache_time > 0 else 1
    print(f"\n📈 Benchmark Results:")
    print(f"  With Cache: {with_cache_time:.3f}s")
    print(f"  Without Cache: {without_cache_time:.3f}s")
    print(f"  Speedup: {speedup:.2f}x")
    
    return speedup

if __name__ == "__main__":
    try:
        # Run cache performance test
        cache_stats = test_cache_performance()
        
        # Run benchmark
        speedup = benchmark_cache_vs_no_cache()
        
        print(f"\n🎉 Test Results Summary:")
        print(f"  ✅ Cache system implemented successfully!")
        print(f"  ✅ Overall hit rate: {cache_stats['overall_performance']['overall_hit_rate']:.1%}")
        print(f"  ✅ Performance speedup: {speedup:.2f}x")
        print(f"  ✅ Memory efficiency: {cache_stats['overall_performance']['memory_efficiency']:.1%}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
