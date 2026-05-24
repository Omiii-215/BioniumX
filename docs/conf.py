# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'Bionium-X'
copyright = '2026, Bionium-X Research Collaborative'
author = 'Bionium-X Research Collaborative'
release = '0.1.0'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.intersphinx',
    'sphinx_copybutton',
]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
}

autosummary_generate = True
numpydoc_show_class_members = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_logo = '_static/bioniumx_logo.jpg'
html_css_files = [
    'custom.css',
]
