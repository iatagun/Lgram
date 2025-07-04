import time
import psutil
import os
import sys
import tracemalloc
from memory_profiler import profile
import gc

# Add the models directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def get_cpu_usage():
    """Get current CPU usage percentage"""
    return psutil.cpu_percent(interval=1)

def measure_performance(func, *args, **kwargs):
    """Measure execution time, memory usage, and CPU usage of a function"""
    print(f"\n{'='*60}")
    print(f"Testing: {func.__name__}")
    print(f"{'='*60}")
    
    # Start memory tracking
    tracemalloc.start()
    
    # Initial measurements
    start_memory = get_memory_usage()
    start_time = time.time()
    
    # Force garbage collection before test
    gc.collect()
    
    try:
        # Execute function
        result = func(*args, **kwargs)
        
        # Final measurements
        end_time = time.time()
        end_memory = get_memory_usage()
        
        # Memory tracking results
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Calculate metrics
        execution_time = end_time - start_time
        memory_diff = end_memory - start_memory
        peak_memory_mb = peak / 1024 / 1024
        
        # Display results
        print(f"⏱️  Execution time: {execution_time:.2f} seconds")
        print(f"💾 Memory usage:")
        print(f"   - Start: {start_memory:.1f} MB")
        print(f"   - End: {end_memory:.1f} MB")
        print(f"   - Difference: {memory_diff:+.1f} MB")
        print(f"   - Peak during execution: {peak_memory_mb:.1f} MB")
        print(f"🔥 CPU usage: {get_cpu_usage():.1f}%")
        
        return result, {
            'execution_time': execution_time,
            'memory_diff': memory_diff,
            'peak_memory': peak_memory_mb,
            'start_memory': start_memory,
            'end_memory': end_memory
        }
        
    except Exception as e:
        tracemalloc.stop()
        print(f"❌ Error during execution: {e}")
        return None, None

def test_model_loading():
    """Test model initialization performance"""
    print("🚀 Testing model loading and initialization...")
    
    def load_models():
        # Import here to measure loading time
        from chunk import EnhancedLanguageModel, load_text_from_file, text_path
        
        # Load text
        text = load_text_from_file(text_path)
        
        # Load or create model
        model_file = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\language_model.pkl'
        try:
            language_model = EnhancedLanguageModel.load_model(model_file)
        except (FileNotFoundError, EOFError):
            language_model = EnhancedLanguageModel(text, n=2)
            language_model.save_model(model_file)
        
        return language_model
    
    return measure_performance(load_models)

def test_text_generation(language_model, num_sentences=5, sentence_length=13):
    """Test text generation performance"""
    print(f"📝 Testing text generation ({num_sentences} sentences, {sentence_length} words each)...")
    
    def generate_text():
        input_words = tuple(token.lower() for token in "The crime".split())
        return language_model.generate_and_post_process(
            num_sentences=num_sentences,
            input_words=input_words,
            length=sentence_length
        )
    
    return measure_performance(generate_text)

def test_spacy_operations():
    """Test spaCy caching performance"""
    print("🧠 Testing spaCy operations and caching...")
    
    def spacy_test():
        from chunk import cached_nlp, get_word_vector, get_word_pos
        
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "She sells seashells by the seashore.",
            "The crime scene was carefully examined.",
            "The quick brown fox jumps over the lazy dog.",  # Repeat to test cache
            "Machine learning algorithms are fascinating.",
            "The crime scene was carefully examined."  # Repeat to test cache
        ]
        
        # Test cached_nlp
        for text in test_texts:
            doc = cached_nlp(text)
        
        # Test word vector caching
        test_words = ["crime", "scene", "fox", "dog", "crime", "fox"]  # Some repeats
        for word in test_words:
            vector = get_word_vector(word)
            pos = get_word_pos(word)
        
        return "spaCy operations completed"
    
    return measure_performance(spacy_test)

def test_grammar_correction():
    """Test T5 grammar correction performance"""
    print("✏️ Testing T5 grammar correction...")
    
    def grammar_test():
        from chunk import correct_grammar_t5
        
        test_text = "The crime scene was examine by detective. They find evidence that suggest the suspect was there yesterday."
        return correct_grammar_t5(test_text)
    
    return measure_performance(grammar_test)

def run_comprehensive_test():
    """Run all performance tests"""
    print("🔬 Starting Comprehensive Performance Test")
    print(f"System Info:")
    print(f"  - CPU cores: {psutil.cpu_count()}")
    print(f"  - Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  - Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    results = {}
    
    # Test 1: Model Loading
    language_model, results['model_loading'] = test_model_loading()
    
    if language_model:
        # Test 2: Text Generation (small)
        generated_text, results['text_generation_small'] = test_text_generation(
            language_model, num_sentences=3, sentence_length=10
        )
        
        # Test 3: Text Generation (medium)
        _, results['text_generation_medium'] = test_text_generation(
            language_model, num_sentences=5, sentence_length=15
        )
        
        # Test 4: spaCy Operations
        _, results['spacy_operations'] = test_spacy_operations()
        
        # Test 5: Grammar Correction (only if we have generated text)
        if generated_text:
            _, results['grammar_correction'] = test_grammar_correction()
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 PERFORMANCE SUMMARY")
    print(f"{'='*60}")
    
    total_time = 0
    for test_name, metrics in results.items():
        if metrics:
            print(f"{test_name.replace('_', ' ').title()}:")
            print(f"  ⏱️  Time: {metrics['execution_time']:.2f}s")
            print(f"  💾 Memory: {metrics['memory_diff']:+.1f} MB")
            print(f"  🔝 Peak: {metrics['peak_memory']:.1f} MB")
            total_time += metrics['execution_time']
            print()
    
    print(f"🎯 Total execution time: {total_time:.2f} seconds")
    print(f"🏁 Test completed!")
    
    return results

if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        import psutil
    except ImportError:
        print("Installing psutil...")
        os.system("pip install psutil")
        import psutil
    
    try:
        import memory_profiler
    except ImportError:
        print("Installing memory-profiler...")
        os.system("pip install memory-profiler")
        import memory_profiler
    
    # Run the comprehensive test
    run_comprehensive_test()
