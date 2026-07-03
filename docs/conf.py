# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Lgram'
copyright = '2026, Ilker Atagun'
author = 'Ilker Atagun'
release = '2.2.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'en'

html_theme = 'furo'
html_static_path = ['_static']

autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
