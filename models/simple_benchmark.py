import time
import psutil
import os
import sys
import gc

# Add the models directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # Convert to MB

def simple_benchmark():
    """Simple benchmark without external dependencies"""
    print("🔬 Chunk.py Performance Benchmark")
    print("=" * 50)
    
    # System info
    print(f"💻 System Info:")
    print(f"   CPU cores: {psutil.cpu_count()}")
    print(f"   Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"   Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print()
    
    # Test 1: Import and initialization
    print("📦 Test 1: Import and Model Loading")
    start_time = time.time()
    start_memory = get_memory_usage()
    
    try:
        from chunk import EnhancedLanguageModel, load_text_from_file, text_path
        import_time = time.time() - start_time
        import_memory = get_memory_usage() - start_memory
        
        print(f"   ✅ Import time: {import_time:.2f}s")
        print(f"   💾 Import memory: +{import_memory:.1f} MB")
        
        # Load text file
        load_start = time.time()
        text = load_text_from_file(text_path)
        load_time = time.time() - load_start
        print(f"   📄 Text loading: {load_time:.2f}s")
        
        # Load or create model
        model_start = time.time()
        model_file = 'C:\\Users\\user\\OneDrive\\Belgeler\\GitHub\\Lgram\\ngrams\\language_model.pkl'
        try:
            language_model = EnhancedLanguageModel.load_model(model_file)
            print(f"   📚 Model loaded from cache")
        except (FileNotFoundError, EOFError):
            language_model = EnhancedLanguageModel(text, n=2)
            language_model.save_model(model_file)
            print(f"   🔨 New model created")
        
        model_time = time.time() - model_start
        total_init_time = time.time() - start_time
        total_memory = get_memory_usage() - start_memory
        
        print(f"   ⚙️  Model init: {model_time:.2f}s")
        print(f"   🎯 Total init: {total_init_time:.2f}s")
        print(f"   📊 Total memory: +{total_memory:.1f} MB")
        print()
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return
    
    # Test 2: Text Generation (Small)
    print("📝 Test 2: Text Generation (Small - 3 sentences)")
    gen_start = time.time()
    gen_start_memory = get_memory_usage()
    
    try:
        input_words = tuple(token.lower() for token in "The crime".split())
        generated_text = language_model.generate_and_post_process(
            num_sentences=3,
            input_words=input_words,
            length=10
        )
        
        gen_time = time.time() - gen_start
        gen_memory = get_memory_usage() - gen_start_memory
        
        print(f"   ⏱️  Generation time: {gen_time:.2f}s")
        print(f"   💾 Memory used: +{gen_memory:.1f} MB")
        print(f"   📄 Generated text: {generated_text[:100]}...")
        print()
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        generated_text = None
    
    # Test 3: Text Generation (Medium)
    print("📝 Test 3: Text Generation (Medium - 5 sentences)")
    gen2_start = time.time()
    gen2_start_memory = get_memory_usage()
    
    try:
        generated_text2 = language_model.generate_and_post_process(
            num_sentences=5,
            input_words=input_words,
            length=15
        )
        
        gen2_time = time.time() - gen2_start
        gen2_memory = get_memory_usage() - gen2_start_memory
        
        print(f"   ⏱️  Generation time: {gen2_time:.2f}s")
        print(f"   💾 Memory used: +{gen2_memory:.1f} MB")
        print(f"   📄 Generated text: {generated_text2[:100]}...")
        print()
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: spaCy Caching
    print("🧠 Test 4: spaCy Caching Performance")
    spacy_start = time.time()
    
    try:
        from chunk import cached_nlp, get_word_vector, get_word_pos
        
        # Test with repeated texts to measure cache effectiveness
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "She sells seashells by the seashore.",
            "The crime scene was carefully examined.",
            "The quick brown fox jumps over the lazy dog.",  # Repeat
            "The crime scene was carefully examined."  # Repeat
        ]
        
        # First pass (cache miss)
        cache_miss_start = time.time()
        for text in test_texts[:3]:
            doc = cached_nlp(text)
        cache_miss_time = time.time() - cache_miss_start
        
        # Second pass (cache hit)
        cache_hit_start = time.time()
        for text in test_texts[3:]:
            doc = cached_nlp(text)
        cache_hit_time = time.time() - cache_hit_start
        
        total_spacy_time = time.time() - spacy_start
        
        print(f"   🔄 Cache miss (3 texts): {cache_miss_time:.3f}s")
        print(f"   ⚡ Cache hit (2 texts): {cache_hit_time:.3f}s")
        print(f"   📈 Speed improvement: {cache_miss_time/3 / (cache_hit_time/2 + 0.001):.1f}x")
        print(f"   🎯 Total spaCy time: {total_spacy_time:.3f}s")
        print()
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Grammar Correction (if available)
    if generated_text:
        print("✏️ Test 5: Grammar Correction")
        grammar_start = time.time()
        grammar_start_memory = get_memory_usage()
        
        try:
            from chunk import correct_grammar_t5
            
            test_text = "The crime scene was examine by detective yesterday."
            corrected = correct_grammar_t5(test_text)
            
            grammar_time = time.time() - grammar_start
            grammar_memory = get_memory_usage() - grammar_start_memory
            
            print(f"   ⏱️  Correction time: {grammar_time:.2f}s")
            print(f"   💾 Memory used: +{grammar_memory:.1f} MB")
            print(f"   📝 Original: {test_text}")
            print(f"   ✅ Corrected: {corrected}")
            print()
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Final Summary
    total_runtime = time.time() - start_time
    final_memory = get_memory_usage()
    
    print("=" * 50)
    print("📊 FINAL SUMMARY")
    print("=" * 50)
    print(f"🎯 Total runtime: {total_runtime:.2f} seconds")
    print(f"💾 Final memory usage: {final_memory:.1f} MB")
    print(f"⚡ Average performance: {(3 + 5) / (gen_time + gen2_time):.1f} sentences/second")
    print("✅ Benchmark completed!")

if __name__ == "__main__":
    try:
        simple_benchmark()
    except KeyboardInterrupt:
        print("\n❌ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
