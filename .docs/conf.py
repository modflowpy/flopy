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
import os
import sys

# add flopy root directory to the python path
sys.path.insert(0, os.path.abspath(".."))
from flopy import __version__

# -- determine if running on readthedocs ------------------------------------
on_rtd = os.environ.get('READTHEDOCS') == 'True'

# -- create source rst files ------------------------------------------------
cmd = "sphinx-apidoc -e -o source/ ../flopy/"
print(cmd)
os.system(cmd)

# -- programatically create rst files ---------------------------------------
cmd = ("python", "create_rstfiles.py")
print(" ".join(cmd))
os.system(" ".join(cmd))

# -- convert the tutorial scripts -------------------------------------------
if not on_rtd:
    cmd = ("python", "create_tutorials.py")
    print(" ".join(cmd))
    os.system(" ".join(cmd))

# -- Project information -----------------------------------------------------
project = "flopy Documentation"
copyright = "2020, Bakker, Mark, Post, Vincent, Langevin, C. D., Hughes, J. D., White, J. T., Leaf, A. T., Paulinski, S. R., Larsen, J. D., Toews, M. W., Morway, E. D., Bellino, J. C., Starn, J. J., and Fienen, M. N."
author = "Bakker, Mark, Post, Vincent, Langevin, C. D., Hughes, J. D., White, J. T., Leaf, A. T., Paulinski, S. R., Larsen, J. D., Toews, M. W., Morway, E. D., Bellino, J. C., Starn, J. J., and Fienen, M. N."

# The version.
version = __version__
release = __version__
language = None

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "IPython.sphinxext.ipython_console_highlighting",  # lowercase didn't work
    "sphinx.ext.autosectionlabel",
    "nbsphinx",
    "nbsphinx_link",
    "recommonmark",
]

# Settings for GitHub actions integration
if on_rtd:
    extensions.append("rtds_action")
    rtds_action_github_repo = "modflowpy/flopy"
    rtds_action_path = "_notebooks"
    rtds_action_artifact_prefix = "notebooks-for-"
    rtds_action_github_token = os.environ.get("GITHUB_TOKEN", None)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
source_suffix = ".rst"

# The encoding of source files.
source_encoding = "utf-8"

# The master toctree document.
master_doc = "index"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# A list of ignored prefixes for module index sorting.
# modindex_common_prefix = []

# If true, keep warnings as "system message" paragraphs in the built documents.
# keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_theme_options = {
    "github_url": "https://github.com/modflowpy/flopy",
    "use_edit_page_button": False,
}

autosummary_generate = True
numpydoc_show_class_members = False

html_context = {
    "github_user": "flopy",
    "github_repo": "flopy",
    "github_version": "master",
    "doc_path": "doc",
}

html_css_files = [
    "css/custom.css",
]

# A shorter title for the navigation bar.  Default is the same as html_title.
html_short_title = "flopy"
html_favicon = "_images/flopylogo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".


# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, links to the reST sources are added to the pages.
# html_show_sourcelink = True

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = False

# Output file base name for HTML help builder.
htmlhelp_basename = "flopydoc"
