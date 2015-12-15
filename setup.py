import os
import sys
from setuptools import setup
# To use:
#	   python setup.py bdist --format=wininst

from flopy import __version__, __name__, __author__

# trap someone trying to install flopy with something other
#  than python 2 or 3
if not sys.version_info[0] in [2, 3]:
    print('Sorry, Flopy not supported in your Python version')
    print('  Supported versions: 2 and 3')
    print('  Your version of Python: {}'.format(sys.version_info[0]))
    sys.exit(1)  # return non-zero value for failure

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
      author_email='mark.bakker@tudelft.nl, vincent.post@flinders.edu.au, langevin@usgs.gov, jdhughes@usgs.gov, ' +
                   'jwhite@usgs.gov, jjstarn@usgs.gov, mnfienen@usgs.gov, frances.alain@gmail.com',
      url='https://github.com/modflowpy/flopy/',
      license='New BSD',
      platforms='Windows, Mac OS-X',
      install_requires=['numpy>=1.7', 'matplotlib>=1.3'],
      packages=['flopy', 'flopy.modflow', 'flopy.modpath', 'flopy.mt3d',
                'flopy.seawat', 'flopy.utils', 'flopy.plot', 'flopy.pest',
                'flopy.export'],
      # use this version ID if .svn data cannot be found
      version=__version__)
