"""
Core module containing the main EnhancedLanguageModel class and configuration.
This module is the heart of the Lgram package.
"""

# Import the main model from the original location
# This allows backward compatibility while organizing the package structure

import sys
import os

# Add the models directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
models_dir = os.path.join(parent_dir, 'models')

if models_dir not in sys.path:
    sys.path.insert(0, models_dir)

from .models.simple_language_model import (
    EnhancedLanguageModel,
    Config,
    ModelInitializer,
    TextLoader,
    create_language_model
)

__all__ = [
    'EnhancedLanguageModel',
    'Config',
    'ModelInitializer',
    'TextLoader', 
    'create_language_model'
]
