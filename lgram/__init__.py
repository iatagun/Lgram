"""
Centering-Lgram: Advanced Language Model with Centering Theory for Coherent Text Generation

A sophisticated natural language processing library that combines n-gram language models
with Centering Theory to generate coherent and contextually appropriate text.

Key Features:
- N-gram based language modeling (2-gram to 6-gram)
- Centering Theory implementation for discourse coherence
- Grammar correction using T5 transformer models
- Semantic relationship analysis using SpaCy
- Django framework integration ready
- Collocation and thematic consistency analysis
"""

# --- Package Metadata ---
__version__ = "1.0.38"
__author__ = "İlker Atagün"
__email__ = "ilker.atagun@gmail.com"
__license__ = "MIT"
__url__ = "https://github.com/iatagun/Lgram"

import os
import sys
import logging
from typing import Optional

# Import all main classes and functions from core.py for PyPI compatibility
try:
    from .core import (
        EnhancedLanguageModel,
        Config,
        ModelInitializer,
        TextLoader,
        create_language_model
    )
    _import_success = True
except ImportError as e:
    _import_success = False
    import warnings
    warnings.warn(
        f"Could not import core Lgram components. Original error: {e}. "
        "Some functionality may not be available.",
        ImportWarning
    )
    # Create minimal dummy classes
    class EnhancedLanguageModel:
        def __init__(self, *args, **kwargs):
            raise ImportError("EnhancedLanguageModel not available")
    class Config:
        pass
    class ModelInitializer:
        pass
    class TextLoader:
        pass
    def create_language_model(*args, **kwargs):
        raise ImportError("create_language_model not available")

# Import utility functions
def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration for the package."""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging_config = {
        'level': numeric_level,
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    if log_file:
        logging_config['filename'] = log_file
        logging_config['filemode'] = 'a'
    
    logging.basicConfig(**logging_config)

def check_gpu_availability():
    """Check GPU availability and return system information."""
    info = {
        'gpu_available': False,
        'gpu_count': 0,
        'gpu_names': [],
        'cuda_version': None,
        'total_memory': 0
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            info['gpu_available'] = True
            info['gpu_count'] = torch.cuda.device_count()
            info['cuda_version'] = torch.version.cuda
            
            for i in range(info['gpu_count']):
                name = torch.cuda.get_device_name(i)
                info['gpu_names'].append(name)
                
                if i == 0:
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    info['total_memory'] = total_memory // (1024**3)  # GB
                    
    except ImportError:
        pass
    
    return info

def print_system_info() -> None:
    """Print system information relevant to Lgram."""
    gpu_info = check_gpu_availability()

    print("System Information:")
    print(f"   GPU Available: {'YES' if gpu_info['gpu_available'] else 'NO'}")

    if gpu_info['gpu_available']:
        print(f"   GPU Count: {gpu_info['gpu_count']}")
        print(f"   CUDA Version: {gpu_info['cuda_version']}")
        for i, name in enumerate(gpu_info['gpu_names']):
            print(f"   GPU {i}: {name}")
        if gpu_info['total_memory'] > 0:
            print(f"   GPU Memory: {gpu_info['total_memory']} GB")

    print()

def show_info():
    """Show package information"""
    status_icon = "OK" if _import_success else "WARN"
    info_text = f"""
LGRAM - Advanced Language Model with Centering Theory v{__version__} [{status_icon}]
Author: {__author__}
GitHub: {__url__}

Coherent Text Generation with Discourse Analysis
N-gram Models (2-gram to 6-gram) + Centering Theory
Grammar Correction with T5 Transformers
Django Framework Ready

Quick Start:
    from lgram import create_language_model
    model = create_language_model()
    text = model.generate_text(num_sentences=3, input_words=["Hello"])
"""
    print(info_text)

    if not _import_success:
        print("WARNING: Some core components could not be imported.")
        print("   Please ensure all dependencies are installed: pip install centering-lgram[full]")
        print()

# Export main classes and functions
__all__ = [
    # Core classes
    'EnhancedLanguageModel',
    'Config',
    'ModelInitializer', 
    'TextLoader',
    
    # Factory functions
    'create_language_model',
    
    # Utility functions
    'setup_logging',
    'check_gpu_availability',
    'print_system_info',
    'show_info',
    
    # Version info
    '__version__',
    '__author__',
    '__email__',
    '__license__',
]

# Set up logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

def _check_dependencies():
    """Check if all required dependencies are available"""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append('torch')
    
    try:
        import spacy
    except ImportError:
        missing_deps.append('spacy')
    
    try:
        import transformers
    except ImportError:
        missing_deps.append('transformers')
    
        if missing_deps:
            import warnings
            warnings.warn(
                f"Missing dependencies: {', '.join(missing_deps)}. "
                "Some features may not work properly. "
                "Install with: pip install centering-lgram[full]",
                ImportWarning
            )# Check dependencies on import
_check_dependencies()

# Auto-show info if in interactive mode
if hasattr(__builtins__, '__IPYTHON__') or os.environ.get('JUPYTER_RUNNING'):
    show_info()
