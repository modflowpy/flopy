import os
import sys
from distutils.core import setup

# To use:
#	   python setup.py bdist --format=wininst

from flopy import __version__, __name__, __author__

long_description = ''
 
try:
   import pypandoc
   long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   long_description = open('README.md').read()  
       
setup(name=__name__,
      description='FloPy is a Python package to create, run, and post-process MODFLOW-based models.',
      long_description=long_description,  
      author=__author__,
      author_email='mark.bakker@tudelft.nl, vincent.post@flinders.edu.au, langevin@usgs.gov, jdhughes@usgs.gov, jwhite@usgs.gov, frances.alain@gmail.com',
      url='https://github.com/modflowpy/flopy/',
      license='New BSD',
      platforms='Windows, Mac OS-X',
      packages=['flopy','flopy.modflow','flopy.modpath','flopy.mt3d','flopy.seawat','flopy.utils'],
      # use this version ID if .svn data cannot be found
      version=__version__ )
