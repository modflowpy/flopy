# Scripts

This document describes the utility scripts in this directory.

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [`make-release.py`](#make-releasepy)
- [`process_benchmarks.py`](#process_benchmarkspy)
- [`pull_request_prepare.py`](#pull_request_preparepy)
- [`run_notebooks.py`](#run_notebookspy)
- [`update_version.py`](#update_versionpy)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## `make-release.py`

TODO

## `process_benchmarks.py`

The `process_benchmarks.py` script consumes JSON files created by `pytest-benchmark` and creates a CSV file containing benchmark results. The script takes 2 positional arguments:

1. The path to the directory containing the JSON files.
2. The path to the directory to create the results CSV.

## `pull_request_prepare.py`

The `pull_request_prepare.py` script lints Python source code files by running `black` and `isort` on the `flopy` subdirectory. This script should be run before opening a pull request, as CI will fail if the code is not properly formatted.

## `run_notebooks.py`

TODO

## `update_version.py`

The `update_version.py` script can be used to update the FloPy version number. If the script is run with no argument, the version number is not changed, but an updated timestamp is written to the `flopy/version.py` file. To set the FloPy version number, use the `--version` (short `-v`) option, e.g.:

```shell
python scripts/update_version.py 3.3.6
```