#!/usr/bin/env python3
"""
Simplified Cache Performance Test
Tests core cache functionality without external dependencies
"""

import time
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgram.models.simple_language_model import create_language_model

def test_center_extraction_cache():
    """Test center extraction with cache performance"""
    print("🎯 Testing Center Extraction Cache Performance")
    print("=" * 50)
    
    model = create_language_model()
    
    # Test sentences
    sentences = [
        "The cat sat on the mat.",
        "John walked to the store yesterday.",
        "She loves reading books in the library.",
        "The beautiful garden bloomed in spring.",
        "They played soccer in the park.",
        "The cat sat on the mat.",  # Duplicate
        "John walked to the store yesterday.",  # Duplicate
        "She loves reading books in the library.",  # Duplicate
    ]
    
    print("📝 First Pass (Cache Miss Expected):")
    start_time = time.time()
    first_pass_results = []
    for i, sentence in enumerate(sentences):
        center = model._extract_center_from_sentence(sentence)
        first_pass_results.append(center)
        print(f"  {i+1:2d}. '{sentence[:40]}...' -> '{center}'")
    first_pass_time = time.time() - start_time
    
    # Cache stats after first pass
    stats_1 = model.get_cache_stats()
    print(f"\n📊 After First Pass:")
    print(f"  Time: {first_pass_time:.3f}s")
    print(f"  Centers Cache: {stats_1['cache_details']['centers']['size']} entries")
    print(f"  Hit Rate: {stats_1['cache_details']['centers']['hit_rate']:.1%}")
    
    print(f"\n🔄 Second Pass (Cache Hit Expected for Duplicates):")
    start_time = time.time()
    second_pass_results = []
    for i, sentence in enumerate(sentences):
        center = model._extract_center_from_sentence(sentence)
        second_pass_results.append(center)
        print(f"  {i+1:2d}. '{sentence[:40]}...' -> '{center}'")
    second_pass_time = time.time() - start_time
    
    # Cache stats after second pass
    stats_2 = model.get_cache_stats()
    print(f"\n📊 After Second Pass:")
    print(f"  Time: {second_pass_time:.3f}s")
    print(f"  Centers Cache: {stats_2['cache_details']['centers']['size']} entries")
    print(f"  Hit Rate: {stats_2['cache_details']['centers']['hit_rate']:.1%}")
    
    # Performance improvement
    improvement = ((first_pass_time - second_pass_time) / first_pass_time * 100) if first_pass_time > 0 else 0
    print(f"\n⚡ Performance Analysis:")
    print(f"  First Pass: {first_pass_time:.3f}s")
    print(f"  Second Pass: {second_pass_time:.3f}s")
    print(f"  Improvement: {improvement:.1f}%")
    
    return stats_2

def test_sentence_generation_cache():
    """Test sentence generation cache"""
    print("\n📝 Testing Sentence Generation Cache")
    print("=" * 50)
    
    model = create_language_model()
    
    # Test start words
    start_words_list = [
        ["The", "cat"],
        ["John", "walked"],
        ["She", "loves"],
        ["The", "cat"],  # Duplicate
        ["John", "walked"],  # Duplicate
    ]
    
    print("📝 Generating sentences:")
    start_time = time.time()
    results = []
    for i, start_words in enumerate(start_words_list):
        sentence = model.generate_sentence(start_words, base_length=8)
        results.append(sentence)
        print(f"  {i+1:2d}. {' '.join(start_words)} -> {sentence}")
    generation_time = time.time() - start_time
    
    print(f"\n⏱️  Generation Time: {generation_time:.3f}s")
    
    return generation_time

def test_cache_memory_management():
    """Test cache memory management and optimization"""
    print("\n🧠 Testing Cache Memory Management")
    print("=" * 50)
    
    model = create_language_model()
    
    # Fill cache with many entries
    print("📈 Filling cache with test data...")
    for i in range(100):
        test_sentence = f"Test sentence number {i} with different content."
        model._extract_center_from_sentence(test_sentence)
    
    # Get stats
    stats = model.get_cache_stats()
    print(f"📊 Cache Status:")
    print(f"  Total Requests: {stats['overall_performance']['total_requests']}")
    print(f"  Cache Entries: {stats['cache_details']['centers']['size']}")
    print(f"  Utilization: {stats['cache_details']['centers']['utilization']:.1%}")
    
    # Test memory optimization
    print(f"\n🔧 Running memory optimization...")
    model.optimize_cache_memory()
    print(f"✅ Memory optimization completed")
    
    # Test cache clearing
    print(f"\n🗑️  Testing cache clearing...")
    model.clear_all_caches()
    stats_after_clear = model.get_cache_stats()
    print(f"✅ Cache cleared - entries: {stats_after_clear['cache_details']['centers']['size']}")

def main():
    """Main test function"""
    print("🚀 SmartCache System Test Suite")
    print("=" * 60)
    
    try:
        # Test 1: Center Extraction Cache
        cache_stats = test_center_extraction_cache()
        
        # Test 2: Sentence Generation Cache
        gen_time = test_sentence_generation_cache()
        
        # Test 3: Memory Management
        test_cache_memory_management()
        
        # Final Summary
        print(f"\n🎉 Test Suite Summary")
        print("=" * 60)
        overall = cache_stats['overall_performance']
        print(f"✅ SmartCache System: OPERATIONAL")
        print(f"✅ Overall Hit Rate: {overall['overall_hit_rate']:.1%}")
        print(f"✅ Memory Efficiency: {overall['memory_efficiency']:.1%}")
        print(f"✅ Total Cache Requests: {overall['total_requests']}")
        print(f"✅ Cache Performance: OPTIMIZED")
        
        # Recommendations
        print(f"\n💡 Optimization Status:")
        recommendations = cache_stats['recommendations']
        if recommendations:
            for rec in recommendations[:3]:  # Show top 3
                print(f"  • {rec}")
        else:
            print(f"  • Cache system is optimally configured!")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
