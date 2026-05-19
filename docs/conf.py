import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "aljabr"
copyright = "2013, 2026, François Orieux"
author = "François Orieux"
release = "0.5.0"
version = ".".join(release.split(".")[:2])

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
]

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_theme = "furo"
html_static_path = ["_static"]

napoleon_numpy_docstring = True
napoleon_google_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

add_module_names = False
autodoc_member_order = "bysource"
autodoc_preserve_defaults = True
autoclass_content = "class"

myst_enable_extensions = ["dollarmath", "colon_fence"]
