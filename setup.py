from setuptools import setup, find_packages
import os

# Read README.md for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return [line for line in lines if line and not line.startswith('#')]

requirements = [
    # Core ML and NLP dependencies
    'torch>=1.9.0',
    'transformers>=4.20.0',
    'spacy>=3.4.0',
    'scikit-learn>=1.0.0',
    'scipy>=1.7.0',
    'numpy>=1.21.0',
    'tqdm>=4.62.0',
    
    # Optional dependencies for full functionality
    'django>=3.2.0;extra=="django"',
]

setup(
    name='centering-lgram',
        version="1.1.3",
    author='İlker Atagün',
    author_email='ilker.atagun@gmail.com',
    description='Advanced Language Model with Centering Theory for Coherent Text Generation',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/iatagun/Lgram',
    project_urls={
        'Bug Reports': 'https://github.com/iatagun/Lgram/issues',
        'Source': 'https://github.com/iatagun/Lgram',
        'Documentation': 'https://github.com/iatagun/Lgram/blob/main/README.md',
    },
    packages=find_packages(include=["lgram*", "tests*"]),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'django': ['django>=3.2.0'],
        'full': [
            'django>=3.2.0',
            'jupyter>=1.0.0',
            'matplotlib>=3.5.0',
            'plotly>=5.0.0',
        ],
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
        ]
    },
    entry_points={
        'console_scripts': [
            'centering-lgram=lgram.cli:main',
            'lgram=lgram.cli:main',  # Keep backward compatibility
        ],
    },
    include_package_data=True,
    package_data={
        'lgram': [
            '*.py',
            'models/*.py',
            'models/logs/*',
        ],
        'models': [
            '*.py',
            'logs/*',
        ],
        'ngrams': [
            '*.txt', '*.json', '*.pkl', '*.pt'
        ],
        'logs': [
            '*.txt'
        ],
        'tests': [
            '*.py'
        ],
    },
    keywords=[
        'nlp', 'natural language processing', 'text generation', 
        'centering theory', 'coherence', 'language model',
        'n-gram', 'discourse analysis', 'computational linguistics'
    ],
)
