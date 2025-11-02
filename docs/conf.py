# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import os
import sys
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))
# sys.path.insert(0, os.path.abspath('../em_simulation.data_updater.data_updater'))

import em_simulation

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'DREME'
copyright = '2025, Jaesung Song'
author = 'Jaesung Song'
release = '2025'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",        # Google/NumPy docstrings
    "sphinx.ext.intersphinx",     # optional
    "sphinx.ext.autodoc.typehints"
]

autosummary_generate = True
autoclass_content = "both"        # include class docstring + __init__ docstring
autodoc_typehints = "description" # cleaner signatures
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
# extensions = []

# templates_path = ['_templates']
# exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
