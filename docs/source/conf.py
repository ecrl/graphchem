# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

project = "GraphChem"
copyright = "Energy & Combustion Research Laboratory"
author = "Travis Kessler"

try:
    from graphchem import __version__ as version
except Exception:  # pragma: no cover - docs build without install
    version = "0.0.0+unknown"

release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
]

templates_path = []
exclude_patterns = []

html_theme = "furo"
html_static_path = ["_static"]
html_title = f"GraphChem {release}"

autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
# Avoid autodoc failing -W on incomplete Example blocks in older docstrings.
napoleon_preprocess_types = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "torch": ("https://docs.pytorch.org/docs/stable/", None),
}
