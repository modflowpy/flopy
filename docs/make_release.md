Instructions for making a FloPy release
-----------------------------------------------

## Make a release branch from develop

1.  Make a release branch from develop (*e.g.* `release3.2.10`)
2.  Update MODFLOW 6 dfn files in the repository and MODFLOW 6 package classes by running: 

    ```
    python -c 'import flopy; flopy.mf6.utils.generate_classes(branch="master", backup=False)'
    ```


## Update the release version number

1.  Increment `major`, `minor`, and/or `micro` numbers in `flopy/version.py`, as appropriate.


## Build USGS release notes

1.  Manually run update-version.py in the `release/` directory to update version information using:

    ```
    python update-version.py
    ```

2.  Run pandoc from the terminal in the root directory to create USGS release notes using:

    ```
    pandoc -o ./docs/USGS_release.pdf ./docs/USGS_release.md ./docs/supported_packages.md ./docs/model_checks.md ./docs/version_changes.md
    ```


## Update the example notebooks

1.  Rerun all of the notebooks in the `examples\Notebooks` directory.
2.  Rerun all of the notebooks in the `examples\Notebooks\groundwater_paper` directory.


## Commit the release branch

1.  Commit the changes to the release (*e.g.* `release3.2.10`) branch.
2.  Push the commit to GitHub.
3.  Wait until the commit successfully runs on [Travis](https://travis-ci.org/modflowpy/flopy/builds).


## Update master branch

1.  Change to the `master` branch in SourceTree.
2.  Merge the release branch (*e.g.* `release3.2.10`) branch into the `master` branch.
3.  Commit changes to `master` branch and push the commit to GitHub.


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


2.  If they are not installed, install one or both using using:


    ```
    conda install pypandoc
    conda install twine
    ```
 
3.  Create the source zip file in a terminal using:

    ```
    python setup.py sdist --format=zip
    ```

4.  Upload the release to PyPi using (*make sure* `twine` *is installed using conda*):

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

3.  Pull upstream [flopy-feedstock](https://github.com/conda-forge/flopy-feedstock) into local copy of the [flopy-feedstock fork](https://github.com/jdhughes-usgs/flopy-feedstock) repo:

    ```
    cd /Users/jdhughes/Documents/Development/flopy-feedstock_git
    git fetch upstream
    git checkout master
    git reset --hard upstream/master  
    git push origin master --force     
    ```

4.  Update the version number in `{% set version = "3.2.7" %}` and sha256 in the [flopy-feedstock fork meta.yaml](https://github.com/jdhughes-usgs/flopy-feedstock/blob/master/recipe/meta.yaml) file.

5.  Commit changes and push to [flopy-feedstock fork](https://github.com/jdhughes-usgs/flopy-feedstock).

6.  Make pull request to [flopy-feedstock](https://github.com/conda-forge/flopy-feedstock)


## Sync develop and master branches

1.  Merge the `master` branch into the `develop` branch.

2.  Increment `major`, `minor`, and/or `micro` numbers in `flopy/version.py`, as appropriate.

3.  Manually run update-version.py in the `release/` directory to update version information using:

    ```
    python update-version.py
    ```    
4.  Commit and push the modified `develop` branch.
    
    
    