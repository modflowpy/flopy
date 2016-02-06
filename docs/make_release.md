Instructions for making a FloPy release
-----------------------------------------------

## Finalizing the release

1.  Merge the `develop` branch into the `master` branch.
2.  Update travis build status in `master` branch README.md` from:

    ```
    [![Build Status](https://travis-ci.org/modflowpy/flopy.svg?branch=develop)](https://travis-ci.org/modflowpy/flopy)
    ```
    
    to:

    ```
    [![Build Status](https://travis-ci.org/modflowpy/flopy.svg?branch=master)](https://travis-ci.org/modflowpy/flopy)
    ```
3.  Update version number in `flopy/version.py`. Use GitHub website to determine what the next build number is for `__build__`
4.  Commit the modified `README.md` in the `master` branch.
5.  Tag the commit with the `__version__` number using SourceTree (don't forget to commit the tag).
6.  Update the flopy `version` number and `"github_tag"` in [https://github.com/ioos/conda-recipes/blob/master/flopy/meta.yaml](https://github.com/ioos/conda-recipes/blob/master/flopy/meta.yaml)

## Update PyPi

1.  Create the source zip file in a terminal using:

    ```
    python setup.py sdist --format=zip
    ```

2.  Register the release with PyPi using:

    ```
    python setup.py register
    ```
3.  Upload the source zip file for the release to PyPi.
4.  Evaluate the PyPi text for the release and modify if necessary. Probably should remove items with relative links to the repo (Examples {not tutorials}, Supported Packages, Changes). 
5.  If necessary, rerun pandoc from a terminal using:

    ```
    pandoc README.md -f markdown_github -t rst -o README.rst
    ```  
    Paste pandoc results in PyPi text for release. 
6.  Modify the link to the travis status in the PyPi text for the release (see version 3.2.3 for an example).

## Build USGS release notes

1.  Update information in `.\docs\USGS_release.md`
2.  Run pandoc from the terminal in the root directory to create USGS release notes using:

    ```
    pandoc -V geometry:margin=0.75in -o ./docs/USGS_release.pdf ./docs/USGS_release.md ./docs/supported_packages.md ./docs/model_checks.md ./docs/version_changes.md
    ```

## Sync master and develop branches

1.  Merge the `master` branch into the `develop` branch.
2.  Update travis build status in `develop` branch `README.md` from:
    
    ```
    [![Build Status](https://travis-ci.org/modflowpy/flopy.svg?branch=master)](https://travis-ci.org/modflowpy/flopy)
    ```
    
    to:

    ```
    [![Build Status](https://travis-ci.org/modflowpy/flopy.svg?branch=develop)](https://travis-ci.org/modflowpy/flopy)
    ```
2.  Update version number in `flopy/version.py` to the next major, minor, or micro number.
3.  Add new section for next version number in `docs/version_changes.md`.
4.  Commit the modified `README.md`, `flopy/version.py`, and `docs/version_changes.md` to the `develop` branch.
