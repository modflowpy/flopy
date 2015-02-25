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
except:
   pass  
       
setup(name=__name__,
      description='FloPy is a Python package to create, run, and post-process MODFLOW-based models.',
      long_description=long_description,  
      author=__author__,
      author_email='mark.bakker@tudelft.nl, vincent.post@flinders.edu.au, langevin@usgs.gov, jdhughes@usgs.gov, jwhite@usgs.gov, jjstarn@usgs.gov, mnfienen@usgs.gov, frances.alain@gmail.com',
      url='https://github.com/modflowpy/flopy/',
      license='New BSD',
      platforms='Windows, Mac OS-X',
      setup_requires=['numpy>=1.9'],
      install_requires=['numpy>=1.9'],
      packages=['flopy', 'flopy.modflow', 'flopy.modpath', 'flopy.mt3d', 'flopy.seawat', 'flopy.utils', 'flopy.plot'],
      # use this version ID if .svn data cannot be found
      version=__version__ )
