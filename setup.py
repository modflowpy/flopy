import sys
from distutils.core import setup

# To use:
#	   python setup.py bdist --format=wininst

from flopy import __version__, __name__, __author__

long_description = \
"""Introduction
-----------------------------------------------

        *FloPy* includes support for MODFLOW-2000, MODFLOW-2005, and MODFLOW-NWT. Other supported MODFLOW-based models include MODPATH (version 6), MT3D and SEAWAT.


Installation
-----------------------------------------------

        To install *FloPy* type:

                ``pip install flopy``

        To update *FloPy* type:

                ``pip install flopy --upgrade``

        To uninstall *FloPy* type:

                ``pip uninstall flopy``


Documentation
-----------------------------------------------

        Documentation for *FloPy* is a work in progress. *FloPy* code documentation is available at: 

                `<http://modflowpy.github.io/flopydoc/>`_


Examples
-----------------------------------------------


MODFLOW Example
++++++++++++++++++++++++++++++++++

        A iPython Notebook for the **Lake Example** problem is available at:

                          `<http://nbviewer.ipython.org/github/modflowpy/flopy/blob/master/examples/basic/lake_example.ipynb>`_

_


SWI2 Test Problems
++++++++++++++++++++++++++++++++++

        A iPython Notebook for SWI2 Problem 1 (rotating interface) is available at:

                `<http://flopy.googlecode.com/svn/examples/SWI2ExampleProblems_flopy.zip>`_
        
         A zip file containing *FloPy* scripts for running and post-processing the SWI2 Examples (examples 1 to 5) that are described in `Bakker et al. (2013) <http://pubs.usgs.gov/tm/6a46/>`_ is available at:

                `<http://flopy.googlecode.com/svn/examples/SWI2ExampleProblems_flopy.zip>`_

        Note that examples 2 and 5 also include *FloPy* scripts for running and post-processing SEAWAT models.


Tutorials
-----------------------------------------------

        A few simple *FloPy* tutorials are available at:

                `<http://modflowpy.github.io/flopydoc/tutorials.html>`_


MODFLOW Resources
-----------------------------------------------

        + `MODFLOW and Related Programs <http://water.usgs.gov/ogw/modflow/>`_
        + `Online guide for MODFLOW-2000 <http://water.usgs.gov/nrp/gwsoftware/modflow2000/Guide/index.html>`_
        + `Online guide for MODFLOW-2005 <http://water.usgs.gov/ogw/modflow/MODFLOW-2005-Guide/>`_
        + `Online guide for MODFLOW-NWT <http://water.usgs.gov/ogw/modflow-nwt/MODFLOW-NWT-Guide/>`_',
""" 

setup(name=__name__,
      description='FloPy is a Python package to create, run, and post-process MODFLOW-based models.',
      long_description=long_description,      
      author=__author__,
      author_email='mark.bakker@tudelft.nl, vincent.post@flinders.edu.au, langevin@usgs.gov, jdhughes@usgs.gov, jwhite@usgs.gov, frances.alain@gmail.com',
      url='https://code.google.com/p/flopy/',
      license='New BSD',
      platforms='Windows, Mac OS-X',
      packages=['flopy','flopy.modflow','flopy.modpath','flopy.mt3d','flopy.seawat','flopy.utils'],
      # use this version ID if .svn data cannot be found
      version=__version__ )
