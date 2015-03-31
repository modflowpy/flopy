import os
import sys
from distutils.core import setup

# To use:
#	   python setup.py bdist --format=wininst

from flopy import __version__, __name__, __author__

#--trap someone trying to install flopy with python 3
if not sys.version_info[0] == 2:
    print "Sorry, Python 3 is not supported (yet)"
    sys.exit(1) # return non-zero value for failure

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
      #setup_requires=['numpy==1.9','matplotlib==1.4'],
      install_requires=['numpy==1.9', 'matplotlib==1.4'],
      packages=['flopy', 'flopy.modflow', 'flopy.modpath', 'flopy.mt3d', 'flopy.seawat', 'flopy.utils', 'flopy.plot'],
      # use this version ID if .svn data cannot be found
      version=__version__ )
