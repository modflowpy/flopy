What is FloPy
=============

The FloPy package consists of a set of Python scripts to run MODFLOW, MT3D, SEAWAT and other MODFLOW-related groundwater programs. FloPy enables you to run all these programs with Python scripts. The FloPy project started in 2009 and has grown to a fairly complete set of scripts with a growing user base. FloPy3 was released in December 2014 with a few great enhancements that make FloPy3 backwards incompatible. The first significant change is that FloPy3 uses zero-based indexing everywhere, which means that all layers, rows, columns, and stress periods start numbering at zero. This change was made for consistency as all array-indexing was already zero-based (as are all arrays in Python). This may take a little getting-used-to, but hopefully will avoid confusion in the future. A second significant enhancement concerns the ability to specify time-varying boundary conditions that are specified with a sequence of layer-row-column-values, like the WEL and GHB packages. A variety of flexible and readable ways have been implemented to specify these boundary conditions. FloPy is an open-source project and any assistance is welcomed. Please email the development team if you want to contribute.

Return to the Github `FloPy <https://github.com/modflowpy/flopy>`_ website.


FloPy Installation
==================

FloPy can be installed using conda (from the conda-forge channel) or pip.

conda Installation
------------------

.. code-block:: bash

    conda install -c conda-forge flopy



pip Installation
----------------

To install FloPy type:

.. code-block:: bash

    pip install flopy


To install the bleeding edge version of FloPy from the git repository type:

.. code-block:: bash

    pip install git+https://github.com/modflowpy/flopy.git



FloPy Resources
===============

`Version history <https://github.com/modflowpy/flopy/blob/develop/docs/version_changes.md>`_

`Supported packages <https://github.com/modflowpy/flopy/blob/develop/docs/supported_packages.md>`_

`Model checking capabilities <https://github.com/modflowpy/flopy/blob/develop/docs/model_checks.md>`_


FloPy Development Team
======================

FloPy is developed by a team of MODFLOW users that have switched over to using
Python for model development and post processing.  Members of the team currently
include:

* Mark Bakker
* Vincent Post
* Joe Hughes
* Chris Langevin
* Jeremy White
* Andy Leaf
* Scott Paulinski
* Josh Larsen
* Mike Toews
* Eric Morway
* Jason Bellino
* Jeff Starn
* Mike Fienen

with contributions from:

* Kolja Rotzoll
* Alain Frances
* and others

Feel free to contact one of us if you would like to participate in FloPy
development.


How to Cite
===========

* `Groundwater Paper <https://github.com/modflowpy/flopy#citation-for-flopy>`_
* `Software Citation <https://github.com/modflowpy/flopy#softwarecode-citation-for-flopy>`_
