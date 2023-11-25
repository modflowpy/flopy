Introduction
============

The FloPy package consists of a set of Python scripts to run MODFLOW, MT3D,
SEAWAT and other MODFLOW-related groundwater programs. FloPy enables you to
run all these programs with Python scripts. The FloPy project started in 2009
and has grown to a fairly complete set of scripts with a growing user base.
FloPy3 was released in December 2014 with a few great enhancements that make
FloPy3 backwards incompatible. The first significant change is that FloPy3
uses zero-based indexing everywhere, which means that all layers, rows,
columns, and stress periods start numbering at zero. This change was made
for consistency as all array-indexing was already zero-based (as are
all arrays in Python). This may take a little getting-used-to, but hopefully
will avoid confusion in the future. A second significant enhancement concerns
the ability to specify time-varying boundary conditions that are specified
with a sequence of layer-row-column-values, like the WEL and GHB packages.
A variety of flexible and readable ways have been implemented to specify these
boundary conditions.

Recently, FloPy has been further enhanced to include full support for
MODFLOW 6. The majority of recent development has focused on FloPy
functionality for MODFLOW 6, helper functions to use GIS shapefiles and
raster files to create MODFLOW datasets, and common plotting and
export functionality.

FloPy provides separate APIs for interacting with MF6 and non-MF6 models.
MODFLOW 6 class definitions are automatically generated from definition
(DFN) files, text files describing the format of MF6 input files.

FloPy is an open-source project and any assistance is welcomed. Please email
the development team if you want to contribute.

Return to the Github `FloPy <https://github.com/modflowpy/flopy>`_ website.

Installation
------------

FloPy can be installed using conda (from the conda-forge channel) or pip.

To install with conda:

.. code-block:: bash

    conda install -c conda-forge flopy



To install with pip:

.. code-block:: bash

    pip install flopy


To install the bleeding edge version of FloPy from the git repository type:

.. code-block:: bash

    pip install git+https://github.com/modflowpy/flopy.git

After FloPy is installed, MODFLOW and related programs can be installed using the command:

.. code-block:: bash

    get-modflow :flopy

See documentation `get_modflow.md <https://github.com/modflowpy/flopy/blob/develop/docs/get_modflow.md>`_
for more information.


Resources
---------

`Version history <https://github.com/modflowpy/flopy/blob/develop/docs/version_changes.md>`_

`Supported packages <https://github.com/modflowpy/flopy/blob/develop/docs/supported_packages.md>`_

`Model checking capabilities <https://github.com/modflowpy/flopy/blob/develop/docs/model_checks.md>`_


Development Team
----------------

FloPy is developed by a team of MODFLOW users that have switched over to using
Python for model development and post-processing.  Members of the team
currently include:

 * Mark Bakker |orcid_Mark_Bakker|
 * Vincent Post |orcid_Vincent_Post|
 * Joseph D. Hughes |orcid_Joseph_D_Hughes|
 * Christian D. Langevin |orcid_Christian_D_Langevin|
 * Jeremy T. White |orcid_Jeremy_T_White|
 * Andrew T. Leaf |orcid_Andrew_T_Leaf|
 * Scott R. Paulinski |orcid_Scott_R_Paulinski|
 * Jason C. Bellino |orcid_Jason_C_Bellino|
 * Eric D. Morway |orcid_Eric_D_Morway|
 * Michael W. Toews |orcid_Michael_W_Toews|
 * Joshua D. Larsen |orcid_Joshua_D_Larsen|
 * Michael N. Fienen |orcid_Michael_N_Fienen|
 * Jon Jeffrey Starn |orcid_Jon_Jeffrey_Starn|
 * Davíd A. Brakenhoff |orcid_Davíd_A_Brakenhoff|
 * Wesley P. Bonelli |orcid_Wesley_P_Bonelli|
 * and others

.. |orcid_Mark_Bakker| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0002-5629-2861
.. |orcid_Vincent_Post| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0002-9463-3081
.. |orcid_Joseph_D_Hughes| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0003-1311-2354
.. |orcid_Christian_D_Langevin| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0001-5610-9759
.. |orcid_Jeremy_T_White| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0002-4950-1469
.. |orcid_Andrew_T_Leaf| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0001-8784-4924
.. |orcid_Scott_R_Paulinski| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0001-6548-8164
.. |orcid_Jason_C_Bellino| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0001-9046-9344
.. |orcid_Eric_D_Morway| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0002-8553-6140
.. |orcid_Michael_W_Toews| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0003-3657-7963
.. |orcid_Joshua_D_Larsen| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0002-1218-800X
.. |orcid_Michael_N_Fienen| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0002-7756-4651
.. |orcid_Jon_Jeffrey_Starn| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0001-5909-0010
.. |orcid_Davíd_A_Brakenhoff| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0002-2993-2202
.. |orcid_Wesley_P_Bonelli| image:: _images/orcid_16x16.png
   :target: https://orcid.org/0000-0002-2665-5078

How to Cite
-----------

* `Groundwater Paper <https://github.com/modflowpy/flopy#citation-for-flopy>`_
* `Software Citation <https://github.com/modflowpy/flopy#softwarecode-citation-for-flopy>`_
