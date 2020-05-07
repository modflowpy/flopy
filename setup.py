import os
import sys
from setuptools import setup

from flopy import __version__, __name__, __author__

# ensure minimum version of Python is running
if sys.version_info[0:2] < (3, 5):
    raise RuntimeError('Flopy requires Python >= 3.5')

try:
    import pypandoc

    fpth = os.path.join('docs', 'PyPi_release.md')
    long_description = pypandoc.convert(fpth, 'rst')
except ImportError:
    long_description = ''

setup(name=__name__,
      description='FloPy is a Python package to create, run, and ' +
                  'post-process MODFLOW-based models.',
      long_description=long_description,
      author=__author__,
      author_email='mark.bakker@tudelft.nl, Vincent.Post@bgr.de, ' +
                   'langevin@usgs.gov, jdhughes@usgs.gov, ' +
                   'j.white@gns.cri.nz, aleaf@usgs.gov, ' +
                   'spaulinski@usgs.gov, jlarsen@usgs.gov,' +
                   'M.Toews@gns.cri.nz, emorway@usgs.gov, ' +
                   'jbellino@usgs.gov, jjstarn@usgs.gov, ' +
                   'mnfienen@usgs.gov',
      url='https://github.com/modflowpy/flopy/',
      license='CC0',
      platforms='Windows, Mac OS-X, Linux',
      install_requires=['numpy'],
      packages=['flopy', 'flopy.modflow', 'flopy.modflowlgr', 'flopy.modpath',
                'flopy.mt3d', 'flopy.seawat', 'flopy.utils', 'flopy.plot',
                'flopy.pest', 'flopy.export', 'flopy.discretization',
                'flopy.mf6', 'flopy.mf6.coordinates', 'flopy.mf6.data',
                'flopy.mf6.modflow', 'flopy.mf6.utils'],
      include_package_data=True,  # includes files listed in MANIFEST.in
      # use this version ID if .svn data cannot be found
      version=__version__,
      classifiers=['Topic :: Scientific/Engineering :: Hydrology'])
