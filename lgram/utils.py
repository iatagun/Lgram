"""
Utility functions and helper classes for the Lgram package.
"""

import os
import logging
from typing import Optional, Dict, Any

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration for the package.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
    """
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

def get_package_data_path(filename: str) -> str:
    """
    Get path to package data files.
    
    Args:
        filename: Name of the data file
        
    Returns:
        Full path to the data file
    """
    package_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(package_dir, 'data', filename)

def validate_model_config(config: Dict[str, Any]) -> bool:
    """
    Validate model configuration parameters.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if configuration is valid
    """
    required_keys = [
        'DEFAULT_NUM_SENTENCES',
        'DEFAULT_SENTENCE_LENGTH', 
        'MIN_SENTENCE_LENGTH',
        'SEMANTIC_THRESHOLD'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")
    
    # Validate ranges
    if config.get('DEFAULT_NUM_SENTENCES', 0) <= 0:
        raise ValueError("DEFAULT_NUM_SENTENCES must be positive")
        
    if config.get('SEMANTIC_THRESHOLD', 0) <= 0 or config.get('SEMANTIC_THRESHOLD', 1) > 1:
        raise ValueError("SEMANTIC_THRESHOLD must be between 0 and 1")
    
    return True

def format_generated_text(text: str, max_line_length: int = 80) -> str:
    """
    Format generated text for better readability.
    
    Args:
        text: Input text
        max_line_length: Maximum line length
        
    Returns:
        Formatted text
    """
    import textwrap
    
    # Split into sentences
    sentences = text.split('. ')
    formatted_sentences = []
    
    for i, sentence in enumerate(sentences):
        # Add period back except for last sentence if it already has punctuation
        if i < len(sentences) - 1 and not sentence.endswith(('.', '!', '?')):
            sentence += '.'
        
        # Wrap long sentences
        wrapped = textwrap.fill(sentence, width=max_line_length)
        formatted_sentences.append(wrapped)
    
    return '\n'.join(formatted_sentences)

def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and return system information.
    
    Returns:
        Dictionary with system information
    """
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
                
                # Get memory info for first GPU
                if i == 0:
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    info['total_memory'] = total_memory // (1024**3)  # GB
                    
    except ImportError:
        pass
    
    return info

def print_system_info() -> None:
    """Print system information relevant to Lgram."""
    gpu_info = check_gpu_availability()
    
    print("🖥️  System Information:")
    print(f"   GPU Available: {'✅' if gpu_info['gpu_available'] else '❌'}")
    
    if gpu_info['gpu_available']:
        print(f"   GPU Count: {gpu_info['gpu_count']}")
        print(f"   CUDA Version: {gpu_info['cuda_version']}")
        for i, name in enumerate(gpu_info['gpu_names']):
            print(f"   GPU {i}: {name}")
        if gpu_info['total_memory'] > 0:
            print(f"   GPU Memory: {gpu_info['total_memory']} GB")
    
    print()

# Export functions
__all__ = [
    'setup_logging',
    'get_package_data_path',
    'validate_model_config',
    'format_generated_text',
    'check_gpu_availability',
    'print_system_info'
]
