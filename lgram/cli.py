#!/usr/bin/env python3
"""
Command Line Interface for Lgram package.
"""

import argparse
import sys
import os
from typing import List, Optional

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Centering-Lgram: Advanced Language Model with Centering Theory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  centering-lgram generate --input "The weather today" --sentences 3
  centering-lgram generate --input "She founded" --sentences 5 --length 15 --centering
  centering-lgram info
  centering-lgram version
        """
    )
    
    # Add subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    generate_parser = subparsers.add_parser('generate', help='Generate text')
    generate_parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input text to start generation'
    )
    generate_parser.add_argument(
        '--sentences', '-s',
        type=int,
        default=3,
        help='Number of sentences to generate (default: 3)'
    )
    generate_parser.add_argument(
        '--length', '-l',
        type=int,
        default=13,
        help='Average sentence length (default: 13)'
    )
    generate_parser.add_argument(
        '--centering', '-c',
        action='store_true',
        help='Use centering theory for coherent generation'
    )
    generate_parser.add_argument(
        '--correct', '-g',
        action='store_true',
        help='Apply T5 grammar correction'
    )
    generate_parser.add_argument(
        '--progress', '-p',
        action='store_true',
        help='Show progress bar'
    )
    generate_parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file (default: print to console)'
    )
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    # Version command
    version_parser = subparsers.add_parser('version', help='Show version information')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train a new model')
    train_parser.add_argument(
        '--text-file', '-t',
        type=str,
        required=True,
        help='Path to training text file'
    )
    train_parser.add_argument(
        '--model-file', '-m',
        type=str,
        help='Output model file path'
    )
    train_parser.add_argument(
        '--n-gram', '-n',
        type=int,
        default=2,
        choices=[2, 3, 4, 5, 6],
        help='N-gram size (default: 2)'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Handle commands
    if args.command == 'version':
        return handle_version()
    elif args.command == 'info':
        return handle_info()
    elif args.command == 'generate':
        return handle_generate(args)
    elif args.command == 'train':
        return handle_train(args)
    else:
        parser.print_help()
        return 1

def handle_version():
    """Handle version command."""
    try:
        from lgram import __version__, __author__, __url__
        print(f"Centering-Lgram version {__version__}")
        print(f"Author: {__author__}")
        print(f"Homepage: {__url__}")
        return 0
    except ImportError:
        print("Error: Could not import Centering-Lgram package")
        return 1

def handle_info():
    """Handle info command."""
    try:
        from lgram import show_info
        from lgram.utils import print_system_info
        
        show_info()
        print_system_info()
        return 0
    except ImportError as e:
        print(f"Error: Could not import Lgram package: {e}")
        return 1

def handle_generate(args):
    """Handle generate command."""
    try:
        from lgram import create_language_model
        
        print("🚀 Initializing Centering-Lgram model...")
        model = create_language_model()
        
        print(f"📝 Generating text from: '{args.input}'")
        
        input_words = args.input.strip().split()
        
        if args.centering:
            text = model.generate_text_with_centering(
                num_sentences=args.sentences,
                input_words=input_words,
                length=args.length
            )
        else:
            text = model.generate_text(
                num_sentences=args.sentences,
                input_words=input_words,
                length=args.length,
                use_progress_bar=args.progress
            )
        
        if args.correct:
            print("🔧 Applying grammar correction...")
            text = model.correct_grammar_t5(text)
        
        # Format output
        from lgram.utils import format_generated_text
        formatted_text = format_generated_text(text)
        
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            print(f"✅ Text saved to: {args.output}")
        else:
            print("\n" + "="*60)
            print("📖 Generated Text:")
            print("="*60)
            print(formatted_text)
            print("="*60)
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating text: {e}")
        return 1

def handle_train(args):
    """Handle train command."""
    try:
        from lgram import EnhancedLanguageModel, TextLoader
        
        print(f"📚 Loading training text from: {args.text_file}")
        
        if not os.path.exists(args.text_file):
            print(f"❌ Error: File not found: {args.text_file}")
            return 1
        
        text = TextLoader.load_text_from_file(args.text_file)
        if not text:
            print("❌ Error: Could not load text from file")
            return 1
        
        print(f"🧠 Training {args.n_gram}-gram model...")
        model = EnhancedLanguageModel(text, n=args.n_gram)
        
        if args.model_file:
            model.save_model(args.model_file)
            print(f"✅ Model saved to: {args.model_file}")
        else:
            print("✅ Model trained successfully (not saved)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())
