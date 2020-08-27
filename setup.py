import os
import sys
from setuptools import setup

# ensure minimum version of Python is running
if sys.version_info[0:2] < (3, 5):
    raise RuntimeError("Flopy requires Python >= 3.5")

# local import of package variables in flopy/version.py
# imports __version__, __pakname__, __author__, __author_email__
exec(open(os.path.join("flopy", "version.py")).read())

try:
    import pypandoc

    fpth = os.path.join("docs", "PyPi_release.md")
    long_description = pypandoc.convert(fpth, "rst")
except ImportError:
    long_description = ""

setup(
    name=__pakname__,
    description="FloPy is a Python package to create, run, and "
    + "post-process MODFLOW-based models.",
    long_description=long_description,
    author=__author__,
    author_email=__author_email__,
    url="https://github.com/modflowpy/flopy/",
    license="CC0",
    platforms="Windows, Mac OS-X, Linux",
    install_requires=["numpy"],
    packages=[
        "flopy",
        "flopy.modflow",
        "flopy.modflowlgr",
        "flopy.modpath",
        "flopy.mt3d",
        "flopy.seawat",
        "flopy.utils",
        "flopy.plot",
        "flopy.pest",
        "flopy.export",
        "flopy.discretization",
        "flopy.mf6",
        "flopy.mf6.coordinates",
        "flopy.mf6.data",
        "flopy.mf6.modflow",
        "flopy.mf6.utils",
    ],
    include_package_data=True,  # includes files listed in MANIFEST.in
    # use this version ID if .svn data cannot be found
    version=__version__,
    classifiers=["Topic :: Scientific/Engineering :: Hydrology"],
)
