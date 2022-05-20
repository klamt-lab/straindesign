# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "StrainDesign"
copyright = "2022, Philipp Schneider"
author = "Philipp Schneider"

# The full version, including alpha/beta/rc tags
release = "1.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "autoapi.extension",
    "nbsphinx",
    "sphinx.ext.napoleon",
    "myst_parser",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autoapi_type = "python"
autoapi_dirs = ["../../straindesign"]
autoapi_generate_api_docs = True
autoapi_add_toctree_entry = False
autoapi_options = ["members", "show-inheritance", "special-members"]
# show-module-summary, imported-members
autoapi_python_class_content = "both"
suppress_warnings = ["autoapi.python_import_resolution", "autoapi.not_readable"]
autoapi_member_order = "alphabetical"

intersphinx_mapping = {
    "jinja": ("https://jinja.palletsprojects.com/en/3.0.x/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
    "python": ("https://docs.python.org/3/", None),
}

source_suffix = [".rst", ".md"]  # , '.ipynb' (not required for nbsphinx)

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "default"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

pygments_style = "sphinx"

# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #'preamble': '',
    # Latex figure (float) alignment
    #'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
# latex_documents = [
#     # (master_doc, 'SphinxAutoAPI.tex', u'Sphinx AutoAPI Documentation',
#     #  u'Read the Docs, Inc', 'manual'),
# ]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
# latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
# latex_use_parts = False

# If true, show page references after internal links.
# latex_show_pagerefs = False

# If true, show URL addresses after external links.
# latex_show_urls = False

# Documents to append as an appendix to all manuals.
# latex_appendices = []

# If false, no module index is generated.
# latex_domain_indices = True


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    # (master_doc, 'sphinxautoapi', u'Sphinx AutoAPI Documentation',
    #  [author], 1)
]

# If true, show URL addresses after external links.
# man_show_urls = False


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
# texinfo_documents = [
#     # (master_doc, 'SphinxAutoAPI', u'Sphinx AutoAPI Documentation',
#     #  author, 'SphinxAutoAPI', 'One line description of project.',
#     #  'Miscellaneous'),
# ]

# Documents to append as an appendix to all manuals.
# texinfo_appendices = []

# If false, no module index is generated.
# texinfo_domain_indices = True

# How to display URL addresses: 'footnote', 'no', or 'inline'.
# texinfo_show_urls = 'footnote'

# If true, do not generate a @detailmenu in the "Top" node's menu.
# texinfo_no_detailmenu = False
