# Releasing FloPy

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Steps](#steps)
  - [Update citations](#update-citations)
  - [Review deprecations](#review-deprecations)
  - [Regenerate MF6 module](#regenerate-mf6-module)
  - [Create a release branch](#create-a-release-branch)
  - [Merge release branch to master](#merge-release-branch-to-master)
  - [Publish the release](#publish-the-release)
  - [Reset the develop branch](#reset-the-develop-branch)
- [Conda](#conda)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->


## Steps

The FloPy release procedure is largely automated with GitHub Actions in [`release.yml`](../.github/workflows/release.yml), but there are a few manual steps.

A [local development environment](../DEVELOPER.md) is assumed in this document.

To make a release,

1. Update citations
2. Review deprecations
3. Regenerate MF6 module
4. Create a release branch 
5. Merge release branch to master
6. Publish the release
7. Reset the develop branch

### Update citations

Update the authors in `CITATION.cff` for the Software/Code citation for FloPy, if required.

### Review deprecations

If this is a patch release, skip this step. If this is a minor or major release, review deprecation warnings. Correct any deprecation version numbers if necessary &mdash; for instance, if the warning was added in the latest development cycle but incorrectly anticipated the forthcoming release number (e.g., if this release was expected to be minor but was promoted to major). If any deprecations are due this release, remove the corresponding features. FloPy loosely follows [NEP 23](https://numpy.org/neps/nep-0023-backwards-compatibility.html) deprecation guidelines: removal is recommended after at least 1 year or 2 non-patch releases.

    To search for deprecation warnings with git: `git grep [-[A/B/C]N] <pattern>`, where N is the optional number of extra lines of context and A/B/C selects post-context/pre-context/both, respectively. Some terms to search for:

    - deprecated
    - .. deprecated::
    - DEPRECATED
    - DeprecationWarning
    - FutureWarning

### Regenerate MF6 module

The `flopy.mf6.modflow` module must be regenerated to match the latest MF6 version's input specification.

```shell
python -m flopy.mf6.utils.generate_classes --releasemode
```

The `--releasemode` flag omits developmode variables from generated modules.

### Create a release branch

Create a release branch from `develop`. The release branch name should be the version number with a `v` prefix (e.g. `v3.3.6`), with an optional `rc` suffix.

Pushing the release branch to the repository triggers the workflow. 

- update version strings to match the version number in the release branch name
- regenerate plugin classes from MODFLOW 6 DFN files
- rerun tests and notebooks
- generate and update changelogs
- build and test the FloPy package

If the branch name ends with `rc`, it's a dry run and the workflow stops here. If the branch name does not end with `rc` the workflow creates a draft PR from the release branch into the `master` branch.

### Merge release branch to master

Review the PR and merging if it passes inspection.

**Note:** the PR should be merged, not squashed. Squashing removes the commit history from the `master` branch and causes `develop` and `master` to diverge, which can cause future PRs updating `master` to replay commits from previous releases.

Merging the PR triggers a job to draft a release.

### Publish the release

Review the release and publish it. Publishing the release triggers a final job to publish the `flopy` package to PyPI.

The release workflow assumes [trusted publishing](https://docs.pypi.org/trusted-publishers/) has been configured in the PyPI admin interface. A GitHub environment called `release` is required (however it needs no secrets or environment variables).

For the Conda distribution, there [is a bot](https://github.com/regro-cf-autotick-bot) which will [automatically detect new package versions uploaded to PyPI and create a PR](https://github.com/conda-forge/flopy-feedstock/pull/50) to update the `conda-forge/flopy-feedstock` repository. This PR can be reviewed, updated if needed, and merged to update the package on the `conda-forge` channel. If it becomes necessary to manually publish an update to conda forge, see below.

### Reset the develop branch

Make a new branch from `master`:

```shell
git checkout master
git switch -c post-x.y.z-release-reset
```

Update the version number for the next development cycle:

```shell
python scripts/update_version.py -v x.y.z.dev0
```

The version number must comply with [PEP 440](https://peps.python.org/pep-0440/).

Lint and format Python files: `ruff check .` and `ruff format .` from the project root.

Create and merge (don't squash) a pull request from this branch into `develop`.

## Conda

To manually publish a new version to conda forge, substitute one's own fork of `conda-forge/flopy-feedstock` into the following steps:

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
