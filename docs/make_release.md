Instructions for making a FloPy release
-----------------------------------------------

## Update master

1.  Commit the changes to the `develop` branch and push to the GitHub site.
2.  Change to the `master` branch in SourceTree.
3.  Merge the `develop` branch into the `master` branch.


## Update the release version number

1.  Increment `major`, `minor`, and/or `micro` numbers in `flopy/version.py`, as appropriate. The pre-commit hook will update the `build` variable to 0 since a `tag` has not been created for the release yet.


## Build USGS release notes

1.  Run pandoc from the terminal in the root directory to create USGS release notes using:

    ```
    pandoc -o ./docs/USGS_release.pdf ./docs/USGS_release.md ./docs/supported_packages.md ./docs/model_checks.md ./docs/version_changes.md
    ```

## Finalizing the release

1.  Tag the commit with the `__version__` number using SourceTree (don't forget to commit the tag).
2.  Push the commit and tag to the GitHub website.
3.  Make release on [GitHub website](https://github.com/modflowpy/flopy/releases). Add version changes for [current release](https://github.com/modflowpy/flopy/blob/develop/docs/version_changes.md) from to release text. Publish release.


## Update PyPi

1.  Make sure `pypandoc` and `twine` are installed using:

    ```
    conda search pypandoc
    conda search twine
    ```
    
    If they are not installed, install one or both using using:

    ```
    conda install pypandoc
    conda install twine
    ```
 
2.  Create the source zip file in a terminal using:

    ```
    python setup.py sdist --format=zip
    ```

3.  Upload the release to PyPi using (*make sure* `twine` *is installed using conda*):

    ```
    twine upload dist/flopy-version.zip
    ```


## Update flopy-feedstock for conda install

1.  Download the `*.tar.gz` file for the current release from the [GitHub website](https://github.com/modflowpy/flopy/releases).

2.  Calculate the sha256 checksum for the `*.tar.gz` using:
  
    ```
    openssl sha256 flopy-version.tar.gz 
    ```

    from a terminal.

3.  Pull upsteam [flopy-feedstock](https://github.com/conda-forge/flopy-feedstock) into local copy of the [flopy-feedstock fork](https://github.com/jdhughes-usgs/flopy-feedstock) repo:

    ```
    cd /Users/jdhughes/Documents/Development/flopy-feedstock_git
    git fetch upstream
    ```

3.  Update the version number in `{% set version = "3.2.7" %}` and sha256 in the [flopy-feedstock fork meta.yaml](https://github.com/jdhughes-usgs/flopy-feedstock/blob/master/recipe/meta.yaml) file.

5.  Commit changes and push to [flopy-feedstock fork](https://github.com/jdhughes-usgs/flopy-feedstock).

6.  Make pull request to [flopy-feedstock](https://github.com/conda-forge/flopy-feedstock)


## Sync master and develop branches

1.  Merge the `master` branch into the `develop` branch.
2.  Commit and push the modified `develop` branch.
