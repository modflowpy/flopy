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
from pathlib import Path
import subprocess
import sys

import yaml

# add flopy root directory to the python path
sys.path.insert(0, os.path.abspath(".."))
from flopy import __author__, __version__

# -- determine if running on readthedocs ------------------------------------
on_rtd = os.environ.get("READTHEDOCS") == "True"

# -- determine if this version is a release candidate
with open("../README.md") as f:
    lines = f.readlines()
rc_text = ""
for line in lines:
    if line.startswith("### Version"):
        if "release candidate" in line:
            rc_text = "release candidate"
        break

# -- get authors
with open("../CITATION.cff") as f:
    citation = yaml.safe_load(f.read())

# -- update version number in main.rst
rst_name = "main.rst"
with open(rst_name) as f:
    lines = f.readlines()
with open(rst_name, "w") as f:
    for line in lines:
        if line.startswith("**Documentation for version"):
            line = f"**Documentation for version {__version__}"
            if rc_text != "":
                line += f" --- {rc_text}"
            line += "**\n"
        f.write(line)

# -- update authors in introduction.rst
rst_name = "introduction.rst"
with open(rst_name) as f:
    lines = f.readlines()
tag_start = "FloPy Development Team"
tag_end = "How to Cite"
write_line = True
with open(rst_name, "w") as f:
    for line in lines:
        if line.startswith(tag_start):
            write_line = False
            # update author list
            line += (
                "======================\n\n"
                "FloPy is developed by a team of MODFLOW users that have "
                "switched over to using\nPython for model development and "
                "post-processing.  Members of the team\n"
                "currently include:\n\n"
            )
            image_directives = ""
            orcid_image = "_images/orcid_16x16.png"
            parts = ["given-names", "name-particle", "family-names", "name"]
            for author in citation["authors"]:
                name = " ".join([author[pt] for pt in parts if pt in author])
                line += f" * {name}"
                if "orcid" in author:
                    tag = "orcid_" + name.replace(" ", "_").replace(".", "")
                    line += f" |{tag}|"
                    image_directives += f".. |{tag}| image:: {orcid_image}\n"
                    image_directives += f"   :target: {author['orcid']}\n"
                line += "\n"
            line += " * and others\n\n"
            line += image_directives
            line += "\n"
            f.write(line)
        elif line.startswith(tag_end):
            write_line = True
        if write_line:
            f.write(line)


# -- create source rst files ------------------------------------------------
cmd = "sphinx-apidoc -e -o source/ ../flopy/"
print(cmd)
os.system(cmd)

# -- programmatically create rst files ---------------------------------------
cmd = ("python", "create_rstfiles.py")
print(" ".join(cmd))
os.system(" ".join(cmd))

# -- convert tutorial scripts and run example notebooks ----------------------
if not on_rtd:
    nbs = Path("Notebooks").glob("*.py")
    for nb in nbs:
        if nb.with_suffix(".ipynb").exists():
            print(f"{nb} already exists, skipping")
            continue
        cmd = (
            "jupytext",
            "--to",
            "ipynb",
            "--execute",
            str(nb)
        )
        print(" ".join(cmd))
        os.system(" ".join(cmd))

# -- Project information -----------------------------------------------------
project = "FloPy Documentation"
copyright = f"2022, {__author__}"
author = __author__

# The version.
version = __version__
release = __version__
language = "en"

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
    "sphinxcontrib.jquery"  # https://github.com/readthedocs/sphinx_rtd_theme/issues/1452
]

# Settings for GitHub actions integration
if on_rtd:
    extensions.append("rtds_action")
    rtds_action_github_repo = "modflowpy/flopy"
    # This will overwrite the .docs/Notebooks directory
    # with the notebooks downloaded & extracted from CI
    # artifacts, which is fine. We want to render those
    # with output, not clean ones from version control.
    rtds_action_path = "Notebooks"
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
# html_static_path = ["_static"]

html_theme_options = {
    # "github_url": "https://github.com/modflowpy/flopy",
    # "use_edit_page_button": False,
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
html_favicon = "_images/flopylogo_sm.png"

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
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = False

# Output file base name for HTML help builder.
htmlhelp_basename = "flopydoc"

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "matplotlib": ("https://matplotlib.org", None),
    "pyproj": ("https://pyproj4.github.io/pyproj/stable/", None),
}

# disable automatic notebook execution (nbs are built in CI for now)
nbsphinx_execute = "never"

nbsphinx_prolog = r"""
{% set docname = env.doc2path(env.docname, base=None) %}
""" + Path("prolog.rst").read_text()
