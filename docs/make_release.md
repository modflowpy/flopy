# Releasing FloPy

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Release automation](#release-automation)
- [Release procedure](#release-procedure)
  - [Release from master branch](#release-from-master-branch)
  - [Reinitialize develop branch](#reinitialize-develop-branch)
  - [Publish the release](#publish-the-release)
    - [PyPI](#pypi)
    - [Conda forge](#conda-forge)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Release automation

The FloPy release procedure is mostly automated with GitHub Actions in [`release.yml`](../.github/workflows/release.yml). There are a few manual steps that need to be performed, however:
    
1.  Update `usgsprograms.txt` in the [GitHub pymake repository](https://github.com/modflowpy/pymake) with the path to the new MODFLOW 6 release. Also update all other targets in `usgsprograms.txt` with the path to new releases.

2.  Recompile all of the executables released on the [GitHub executables repository](https://github.com/MODFLOW-USGS/executables) using the `buildall.py` pymake script and Intel compilers for all operating systems.

3.  Update the README.md on the [GitHub executables repository](https://github.com/MODFLOW-USGS/executables) with the information in the `code.md` file created by the `buildall.py` pymake script. 

4.  Make a new release on the [GitHub executables repository](https://github.com/MODFLOW-USGS/executables) and add all of the operating system specific zip files containing the compiled executables (`linux.zip`, `mac.zip`, `win64.zip`, `win32.zip`). Publish the new release.

5. Update the authors in `CITATION.cff` for the Software/Code citation for FloPy, if required.
 
   
Next, make a release branch from develop (*e.g.* `v3.3.6`). The branch name should be the version number with a `v` prefix. Pushing this branch to GitHub will trigger the release workflow, detailed below. If the branch name ends with `rc` it is considered a release candidate and the workflow is a dry run, stopping after updating version info & plugin classes and running tests and notebooks. If the branch name does not end with `rc` the release is considered approved and the full procedure runs.

After updating version information, regenerating plugin classes, and generating a changelog, the approved release workflow creates a draft pull request from the release branch into `master`. Merging this pull request triggers another job to create a draft release. Promoting this draft release to a full release will trigger a final job to publish the release to PyPI and reset the `develop` branch from `master`, incrementing the patch version number on `develop`.


## Release procedure

This procedure runs automatically in `release.yml` after a release branch is pushed to GitHub, except for the final step (updating the `conda-forge/flopy-feedstock` repository &mdash; there is a bot which will [automatically detect changes and create a PR](https://github.com/conda-forge/flopy-feedstock/pull/47) to do so).


### Release from master branch

- Update MODFLOW 6 dfn files in the repository and MODFLOW 6 package classes by running `python -c 'import flopy; flopy.mf6.utils.generate_classes(branch="master", backup=False)'`
  
- Run `isort` and `black` on the updated MODFLOW 6 package classes. This can be achieved by running `python scripts/pull_request_prepare.py` from the project root. The commands `isort .` and `black .` can also be run individually instead.

- Run `python scripts/update_version.py -v <semver>` to update the version number stored in `version.txt` and `flopy/version.py`. For an approved release use the `--approve` flag.

- Use `run_notebooks.py` in the `scripts` directory to rerun all of the notebooks in:

    - `examples\Notebooks` directory.
    - `examples\Notebooks\groundwater_paper` directory.
    - `examples\Notebooks\FAQ` directory.

- Generate a changelog with [git cliff](https://github.com/orhun/git-cliff): `git cliff --unreleased --tag=<version number>`.

- Commit the changes to the release branch and push the commit to the [upstream GitHub repository](https://github.com/modflowpy/flopy).

- Build and check the package with:

```shell
python -m build
twine check --strict dist/*
```

- Update master branch from the release branch, e.g. by opening and merging a pull request into `master`. The pull request should be merged, *not* squashed, in order to preserve the project's commit history.

- Tag the merge commit to `master` with the version number. Don't forget to commit the tag. Push the commit and tag to GitHub.

- Make a release on [GitHub website](https://github.com/modflowpy/flopy/releases). Add version changes for [current release](https://github.com/modflowpy/flopy/blob/develop/docs/version_changes.md) from to release text. Publish release.


### Reinitialize develop branch

1.  Merge the `master` branch into the `develop` branch.

2.  Set the version as appropriate: `python scripts/update_version.py -v <semver>`.

3.  Commit and push the updated `develop` branch.


### Publish the release

#### PyPI

1.  Make sure the latest `build` and `twine` tools are installed using:

    ```
    pip install --upgrade build twine
    ```

2.  Create the source and wheel packages with:

    ```
    rm -rf dist
    python -m build
    ```

3.  Check and upload the release to PyPI using:

    ```
    twine check --strict dist/*
    twine upload dist/*
    ```

#### Conda forge

1.  Download the `*.tar.gz` file for the just-created release from the [GitHub website](https://github.com/modflowpy/flopy/releases).

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

4.  Rerender the repo using `conda-smithy` (make sure `conda-smithy` is installed using conda):

    ```
    conda smithy rerender
    ```

4.  Update the version number in `{% set version = "3.2.7" %}` and sha256 in the [flopy-feedstock fork meta.yaml](https://github.com/jdhughes-usgs/flopy-feedstock/blob/master/recipe/meta.yaml) file.

5.  Commit changes and push to [flopy-feedstock fork](https://github.com/jdhughes-usgs/flopy-feedstock).

6.  Make pull request to [flopy-feedstock](https://github.com/conda-forge/flopy-feedstock)
