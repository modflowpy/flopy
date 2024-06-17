# Scripts

This document describes the utility scripts in this directory.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Processing benchmarks](#processing-benchmarks)
- [Running notebooks](#running-notebooks)
- [Updating version](#updating-version)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Processing benchmarks

The `process_benchmarks.py` script converts one or more JSON files created by [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/) into a single CSV file containing benchmark results. The script takes 2 positional arguments:

1. The path to the directory containing the JSON files.
2. The path to the directory to create the results CSV.

Input JSON files are expected to be named according to the `pytest-benchmark` format used when the `--benchmark-autosave` is provided:

```shell
<commitid>_<date>_<time>_<isdirty>.json
```

For instance, `e689af57e7439b9005749d806248897ad550eab5_20150811_041632_uncommitted-changes.json`.

**Note**: the `process_benchmarks.py` script depends on `seaborn`, which is not included as a dependency in either `etc/environment.yml` or in any of the optional groups in `pyproject.toml`, since this is the only place it is used in this repository.

## Updating version

The `update_version.py` script can be used to update FloPy version numbers. Running the script first updates the version in `version.txt`, then propagates the change to various other places version strings or timestamps are embedded in the repository:

- `flopy/version.py`
- `CITATION.cff`
- `README.md`
- `docs/PyPI_release.md`

The script acquires a file lock before writing to files to make sure only one process edits the files at any given time and prevent desynchronization.

If the script is run with no arguments, the version number is not changed, but updated timestamps are written. To set the version number, use the `--version` (short `-v`) option, e.g.:

```shell
python scripts/update_version.py -v 3.3.6
```

To get the current version number, use the `--get` flag (short `-g`):

```shell
python scripts/update_version.py -g
```

This simply returns the contents of `version.txt` and does not write any changes to the repository's files.
