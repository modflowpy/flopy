Instructions for making a FloPy release
-----------------------------------------------

## Update the release version number

1.  Update information in `.\docs\USGS_release.md`
2.  Rename version number at top of `README.md` from FloPy Version 3.2.X-dev to FloPy Version 3.2.X.
3.  Update version number in `flopy/version.py`. Update the `major`, `minor`, and/or `micro` variables as appropriate. The pre-commit hook will update the `build` variable to 0 since a `tag` has not been created for the release yet.

## Build USGS release notes

1.  Update information in `.\docs\USGS_release.md`
2.  Run pandoc from the terminal in the root directory to create USGS release notes using:

    ```
    pandoc -o ./docs/USGS_release.pdf ./docs/USGS_release.md ./docs/supported_packages.md ./docs/model_checks.md ./docs/version_changes.md
    ```

## Finalizing the release

1.  Commit the changes to the `develop` branch and push to the GitHub site.
2.  Change to the `master` branch in SourceTree.
3.  Merge the `develop` branch into the `master` branch.
4.  Update travis build status in `master` branch `README.md` from:

    ```
    [![Build Status](https://travis-ci.org/modflowpy/flopy.svg?branch=develop)](https://travis-ci.org/modflowpy/flopy)
    ```
    
    to:

    ```
    [![Build Status](https://travis-ci.org/modflowpy/flopy.svg?branch=master)](https://travis-ci.org/modflowpy/flopy)
    ```
5.  Commit the modified `README.md` file in the `master` branch.
6.  Tag the commit with the `__version__` number using SourceTree (don't forget to commit the tag).
7.  Push the commit and tag to the GitHub website.
8.  Make release on [GitHub website](https://github.com/modflowpy/flopy/releases). Add version changes for [current release](https://github.com/modflowpy/flopy/blob/develop/docs/version_changes.md) from to release text. Publish release.

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

## Update flopy-feedstock for conda install

1.  Download the `*.tar.gz` file for the current release from the [GitHub website](https://github.com/modflowpy/flopy/releases).
2.  Rerender [flopy-feedstock fork](https://github.com/jdhughes-usgs/flopy-feedstock) using:

    ```
    conda smithy rerender

    ```

2.  Calculate the sha256 checksum for the `*.tar.gz` using:
  
    ```
    openssl sha256 flopy-X.X.X.tar.gz 
    ```

    from a terminal.

3.  Update the version number in `{% set version = "3.2.6" %}` and sha256 in the [flopy-feedstock fork meta.yaml](https://github.com/jdhughes-usgs/flopy-feedstock/blob/master/recipe/meta.yaml) file.
4.  Commit changes and push to [flopy-feedstock fork](https://github.com/jdhughes-usgs/flopy-feedstock).
5.  Make pull request to [flopy-feedstock](https://github.com/conda-forge/flopy-feedstock)

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

3.  Commit the modified `README.md` to the `develop` branch.
