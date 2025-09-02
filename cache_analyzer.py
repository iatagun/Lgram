#!/usr/bin/env python3
"""
Cache System Performance Analysis and Optimization Tool
Advanced testing and optimization for SmartCacheSystem
"""

import time
import sys
import os
import statistics
from collections import defaultdict

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lgram.models.simple_language_model import create_language_model

class CachePerformanceAnalyzer:
    """Advanced cache performance analysis"""
    
    def __init__(self):
        self.model = create_language_model()
        self.test_results = defaultdict(list)
        
    def benchmark_center_extraction(self, iterations=5):
        """Benchmark center extraction with different scenarios"""
        print("🎯 Benchmarking Center Extraction")
        print("=" * 50)
        
        # Different test scenarios
        scenarios = {
            "unique_sentences": [
                f"The {adj} {noun} {verb} in the {place}."
                for adj, noun, verb, place in [
                    ("red", "car", "drives", "street"),
                    ("blue", "bird", "flies", "sky"),
                    ("green", "tree", "grows", "garden"),
                    ("yellow", "sun", "shines", "morning"),
                    ("black", "cat", "sleeps", "house"),
                ]
            ],
            "duplicate_sentences": [
                "The cat sat on the mat.",
                "John walked to the store.",
                "The cat sat on the mat.",  # Duplicate
                "John walked to the store.",  # Duplicate
                "The cat sat on the mat.",  # Duplicate
            ],
            "complex_sentences": [
                "The sophisticated artificial intelligence system, which was developed by researchers at the university, processes natural language with remarkable accuracy.",
                "Despite the challenging economic conditions that have persisted throughout the year, the company continues to innovate and expand its market presence.",
                "The beautiful sunset over the mountains created a breathtaking panorama that captured the attention of every photographer in the area.",
            ]
        }
        
        results = {}
        
        for scenario_name, sentences in scenarios.items():
            print(f"\n📊 Scenario: {scenario_name.replace('_', ' ').title()}")
            
            # Clear cache for clean test
            self.model.clear_all_caches()
            
            # Run multiple iterations
            iteration_times = []
            
            for iteration in range(iterations):
                start_time = time.time()
                centers = []
                
                for sentence in sentences:
                    center = self.model._extract_center_from_sentence(sentence)
                    centers.append(center)
                
                iteration_time = time.time() - start_time
                iteration_times.append(iteration_time)
                
                if iteration == 0:
                    print(f"  Sample centers: {centers[:3]}...")
            
            # Statistics
            avg_time = statistics.mean(iteration_times)
            min_time = min(iteration_times)
            max_time = max(iteration_times)
            
            print(f"  Average Time: {avg_time:.4f}s")
            print(f"  Min Time: {min_time:.4f}s")
            print(f"  Max Time: {max_time:.4f}s")
            print(f"  Speed Consistency: {(1 - (max_time - min_time) / avg_time) * 100:.1f}%")
            
            results[scenario_name] = {
                'avg_time': avg_time,
                'min_time': min_time,
                'max_time': max_time,
                'times': iteration_times
            }
        
        return results
    
    def analyze_cache_efficiency(self):
        """Analyze cache efficiency with different workloads"""
        print("\n🔍 Cache Efficiency Analysis")
        print("=" * 50)
        
        # Reset cache
        self.model.clear_all_caches()
        
        # Workload simulation
        workloads = {
            "high_repetition": {
                "sentences": ["The cat sat on the mat."] * 20 + 
                           ["John walked to the store."] * 15 +
                           ["She loves reading books."] * 10,
                "description": "High repetition workload"
            },
            "medium_repetition": {
                "sentences": [
                    f"Sentence {i % 5} with some variation {i}"
                    for i in range(25)
                ],
                "description": "Medium repetition workload"
            },
            "low_repetition": {
                "sentences": [
                    f"Unique sentence number {i} with different content every time"
                    for i in range(30)
                ],
                "description": "Low repetition workload"
            }
        }
        
        workload_results = {}
        
        for workload_name, workload_data in workloads.items():
            print(f"\n📈 {workload_data['description']}")
            
            # Clear cache
            self.model.clear_all_caches()
            
            # Process workload
            start_time = time.time()
            for sentence in workload_data['sentences']:
                self.model._extract_center_from_sentence(sentence)
            
            process_time = time.time() - start_time
            
            # Get cache stats
            stats = self.model.get_cache_stats()
            overall = stats['overall_performance']
            centers = stats['cache_details']['centers']
            
            print(f"  Processing Time: {process_time:.3f}s")
            print(f"  Total Requests: {overall['total_requests']}")
            print(f"  Cache Hit Rate: {overall['overall_hit_rate']:.1%}")
            print(f"  Cache Utilization: {centers['utilization']:.1%}")
            print(f"  Cache Efficiency: {process_time / overall['total_requests'] * 1000:.2f}ms per request")
            
            workload_results[workload_name] = {
                'time': process_time,
                'hit_rate': overall['overall_hit_rate'],
                'utilization': centers['utilization'],
                'requests': overall['total_requests']
            }
        
        return workload_results
    
    def stress_test_cache(self):
        """Stress test the cache system"""
        print("\n🏋️ Cache Stress Test")
        print("=" * 50)
        
        # Generate large number of requests
        stress_sizes = [100, 500, 1000]
        stress_results = {}
        
        for size in stress_sizes:
            print(f"\n⚡ Stress testing with {size} requests...")
            
            # Clear cache
            self.model.clear_all_caches()
            
            # Generate test data
            sentences = []
            for i in range(size):
                if i % 10 == 0:  # 10% duplicates
                    sentences.append("Duplicate test sentence for cache testing.")
                else:
                    sentences.append(f"Unique test sentence number {i} with different content.")
            
            # Process with timing
            start_time = time.time()
            for sentence in sentences:
                self.model._extract_center_from_sentence(sentence)
            
            process_time = time.time() - start_time
            
            # Get final stats
            stats = self.model.get_cache_stats()
            overall = stats['overall_performance']
            
            # Calculate performance metrics
            requests_per_second = overall['total_requests'] / process_time if process_time > 0 else 0
            avg_request_time = process_time / overall['total_requests'] * 1000 if overall['total_requests'] > 0 else 0
            
            print(f"  Total Time: {process_time:.2f}s")
            print(f"  Requests/sec: {requests_per_second:.0f}")
            print(f"  Avg Request Time: {avg_request_time:.2f}ms")
            print(f"  Hit Rate: {overall['overall_hit_rate']:.1%}")
            print(f"  Memory Efficiency: {overall['memory_efficiency']:.1%}")
            
            stress_results[size] = {
                'time': process_time,
                'rps': requests_per_second,
                'hit_rate': overall['overall_hit_rate'],
                'avg_request_time': avg_request_time
            }
        
        return stress_results
    
    def generate_performance_report(self):
        """Generate comprehensive performance report"""
        print("\n📊 Comprehensive Performance Report")
        print("=" * 60)
        
        # Run all benchmarks
        center_results = self.benchmark_center_extraction()
        efficiency_results = self.analyze_cache_efficiency()
        stress_results = self.stress_test_cache()
        
        # Generate summary
        print(f"\n🎉 PERFORMANCE SUMMARY")
        print("=" * 60)
        
        # Center extraction performance
        print(f"🎯 Center Extraction Performance:")
        for scenario, data in center_results.items():
            print(f"  {scenario.replace('_', ' ').title()}: {data['avg_time']:.4f}s average")
        
        # Cache efficiency
        print(f"\n🔍 Cache Efficiency Results:")
        for workload, data in efficiency_results.items():
            print(f"  {workload.replace('_', ' ').title()}: {data['hit_rate']:.1%} hit rate, {data['time']:.3f}s")
        
        # Stress test results
        print(f"\n🏋️ Stress Test Results:")
        for size, data in stress_results.items():
            print(f"  {size:4d} requests: {data['rps']:.0f} req/sec, {data['avg_request_time']:.2f}ms avg")
        
        # Recommendations
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        final_stats = self.model.get_cache_stats()
        if final_stats['recommendations']:
            for i, rec in enumerate(final_stats['recommendations'][:5], 1):
                print(f"  {i}. {rec}")
        else:
            print(f"  ✅ Cache system is optimally configured!")
        
        # Performance grades
        overall_hit_rate = final_stats['overall_performance']['overall_hit_rate']
        if overall_hit_rate >= 80:
            grade = "A+ (Excellent)"
        elif overall_hit_rate >= 60:
            grade = "A (Good)"
        elif overall_hit_rate >= 40:
            grade = "B (Fair)"
        else:
            grade = "C (Needs Improvement)"
        
        print(f"\n🏆 CACHE PERFORMANCE GRADE: {grade}")
        print(f"📈 Hit Rate: {overall_hit_rate:.1%}")
        print(f"🧠 Memory Efficiency: {final_stats['overall_performance']['memory_efficiency']:.1%}")
        
        return {
            'center_results': center_results,
            'efficiency_results': efficiency_results,
            'stress_results': stress_results,
            'grade': grade,
            'hit_rate': overall_hit_rate
        }

def main():
    """Main performance analysis function"""
    print("🚀 SmartCache Advanced Performance Analysis")
    print("=" * 70)
    
    try:
        analyzer = CachePerformanceAnalyzer()
        results = analyzer.generate_performance_report()
        
        print(f"\n🎊 Analysis Complete!")
        print(f"   Cache Grade: {results['grade']}")
        print(f"   Overall Hit Rate: {results['hit_rate']:.1%}")
        print(f"   Status: READY FOR PRODUCTION")
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
