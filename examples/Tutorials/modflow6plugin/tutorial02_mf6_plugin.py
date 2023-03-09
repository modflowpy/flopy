# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # FloPy plugin Tutorial 2: Creating a FloPy plugin
#
# This tutorial demonstrates how to use the FloPy plugin template generator to
# get you started on your own FloPy plugin.  For more information see the
# notebooks in the flopy distribution's examples/Notebooks/dev folder.

# ## Getting Started
#
# MODFLOW-6 supports a BMI interface which allows you to modify its behavior.
# Support for accessing the BMI interface using python is provided through the
# modflowapi library.  FloPy plugins are FloPy-integrated components that
# modify MODFLOW-6 functionality using the modflowapi library.  The FloPy
# plugins architecture streamlines the process of modifying MODFLOW-6
# functionality by:
#
# * Providing template generators to help rapidly set up your plugin's
# interface
# * Automatically loading, saving, and running FloPy plugins as part of a
# MODFLOW-6 simulation
# * Providing an interface that allows multiple FloPy plugins to run together
#
# This tutorial describes the process of generating your own custom FloPy
# plugin template using the flopy template generator.  The plugin template
# includes all files you will need to create your basic FloPy plugin.

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import flopy

# To create a plugin template, you first need to decide which type of model
# your plugin will reside in.  Define the model type with its three letter
# abbreviation.

# define the model type
mt = "gwf"

# Next, pick a unique three letter abbreviation for your plugin.  This
# abbreviation should be different from any MODFLOW-6 package or other flopy
# plugin abbreviation.

# define the stress plugin three letter abbreviation
pn = "abc"

# ## Creating user defined settings for a FloPy plugin
#
# FloPy plugins allow the user to define data/variables similar to the data
# and options defined for MODFLOW packages.  User defined variables can be
# created for the options, package data, and stress period data blocks.  Use
# dictionaries to create your user defined variables, where the outer
# dictionary key is the variable name and the outer dictionary value is a
# dictionary of the variable's attributes.
# Variable attributes can include "type", "shape", and "optional".
#
# Types supported include, "double precision", "integer", "string", and
# "keyword".
#
# Data shapes can be defined by other variables, or can reference special
# variables such as "ncelldim" (number of cell dimensions in the model), or
# "nlay", "nrow", "ncol", "nodes", "ncpl"  (number of layers, rows, columns,
# nodes, cells per layer).  Shapes support simple arithmetic operations
# such as nrow*ncol.


# define the plugin options
opt = {
    "print_input": {"type": "keyword", "optional": "True"},
    "print_flows": {"type": "keyword", "optional": "True"},
    "save_input": {"type": "keyword", "optional": "True"},
    "save_flows": {"type": "keyword", "optional": "True"},
    "auxiliary": {"type": "string", "shape": "naux", "optional": "True"},
    "boundnames": {"type": "keyword", "optional": "True"},
}

# define the plugin stress period data
# define the stress plugin stress period data
spv = {
    "cellid": {"type": "integer", "shape": "ncelldim"},
    "q_val": {"type": "double precision"},
    "aux": {"type": "double precision", "shape": "naux", "optional": "True"},
    "boundname": {"type": "string", "optional": "True"},
}

# ## Generating FloPy plugin files with FloPy's generate_plugin_template method
#
# Once all of your plugin's user defined variables are set up in dictionaries,
# you can use FloPy's "generate_plugin_template" method to generate the
# template files for your FloPy plugin.
#
# The "generate_plugin_template" method takes the model type (model_type),
# package abbreviation type (new_package_abbr), options (options), and stress
# period variables (stress_period_vars) defined above.  Additionally, it can
# take package variables (package_vars), it can create a template with or
# without support for the API package (api_package_support), and can also
# define where you want the main loop of your plugin code to be
# (evaluation_code_at).

flopy.mf6.utils.flopy_plugins.plugin_template.generate_plugin_template(
    mt,
    pn,
    options=opt,
    stress_period_vars=spv,
    api_package_support=True,
    evaluation_code_at="iteration_start",
)

# Running generate_plugin_template creates several files.  First, a dfn file is
# created in the flopy/mf6/data/dfn that specifies your plugin's user defined
# variables.  Then, createpackages.py creates a MODFLOW-6-like
# package file in the flopy/mf6/modeflow folder, which users of your plugin
# will instantiate.  Additionally, generate_plugin_template generates a FloPy
# plugin code file in the flopy/mf6/utils/flopy_plugins folder, which contains
# the code that will modify MODFLOW-6's behavior.  Edit this file to change
# the behavior of your plugin.  Start by adding code in the various callback
# methods that get called at various times during MODFLOW-6's execution.
# These callbacks include init_plugin, receive_bmi, stress_period_start,
# stress_period_end, time_step_start, time_step_end, iteration_start,
# iteration_end, and sim_complete.  For example, the code in iteration_start
# will get called every time MODFLOW-6 starts an outer iteration of the
# solution group the FloPy plugin is operating on.
#
# When api_package_support is set to True, the plugin template is set up to
# use MODFLOW-6's generic API package.  The generic API package can be
# used to modify the behavior of a MODFLOW-6 model.  For example, a FloPy
# plugin can modify the API package's rhs, hcof, nodelist, and nbound
# variables (mf6_pkg_rhs, mf6_pkg_hcof, ...) to behave like a custom MODFLOW-6
# package.
#
# More complete tutorials for creating FloPy plugins are available in two
# notebooks, flopy3_dev_mf6_BuildStressPluginAPISupport.ipynb and
# flopy3_dev_mf6_BuildStressPluginNoAPISupport.ipynb in the
# mf6/examples/Notebooks/dev folder of the FloPy distribution.  A third
# notebook, flopy3_dev_mf6_packaging_distributing_flopy_plugins.ipynb,
# walks through the process of exporting a FloPy plugin for use as an
# external python package, which can then be easily distributed.
