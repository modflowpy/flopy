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
    fpth = os.path.join('docs', 'PyPi_release.md')
    long_description = pypandoc.convert(fpth, 'rst')
except:
    pass

setup(name=__name__,
      description='FloPy is a Python package to create, run, and post-process MODFLOW-based models.',
      long_description=long_description,
      author=__author__,
      author_email='mark.bakker@tudelft.nl, Vincent.Post@bgr.de, ' +
                   'langevin@usgs.gov, jdhughes@usgs.gov, jwhite@usgs.gov, ' +
                   'aleaf@usgs.gov, spaulinski@usgs.gov, jjstarn@usgs.gov, ' +
                   'mnfienen@usgs.gov',
      url='https://github.com/modflowpy/flopy/',
      license='CC0',
      platforms='Windows, Mac OS-X, Linux',
      install_requires=['enum34;python_version<"3.4"',
                        'numpy>=1.9'],
      packages=['flopy', 'flopy.modflow', 'flopy.modflowlgr', 'flopy.modpath',
                'flopy.mt3d', 'flopy.seawat', 'flopy.utils', 'flopy.plot',
                'flopy.pest', 'flopy.export',
                'flopy.mf6', 'flopy.mf6.coordinates', 'flopy.mf6.data',
                'flopy.mf6.modflow', 'flopy.mf6.utils'],
      include_package_data=True, # includes files listed in MANIFEST.in
      # use this version ID if .svn data cannot be found
      version=__version__)
