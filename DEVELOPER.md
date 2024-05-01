# Developing FloPy

This document describes how to set up a FloPy development environment, run the example scripts and notebooks, and use the tests. Testing conventions are also briefly discussed. More detail on how to contribute your code to this repository can be found in [CONTRIBUTING.md](CONTRIBUTING.md).

<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->

- [Requirements & installation](#requirements--installation)
  - [Git](#git)
  - [Python](#python)
    - [Python IDEs](#python-ides)
      - [Visual Studio Code](#visual-studio-code)
      - [PyCharm](#pycharm)
  - [MODFLOW executables](#modflow-executables)
    - [Scripted installation](#scripted-installation)
    - [Manually installing executables](#manually-installing-executables)
      - [Linux](#linux)
      - [Mac](#mac)
  - [Updating FloPy packages](#updating-flopy-packages)
- [Examples](#examples)
  - [Developing new examples](#developing-new-examples)
- [Tests](#tests)
  - [Configuring tests](#configuring-tests)
  - [Running tests](#running-tests)
    - [Selecting tests with markers](#selecting-tests-with-markers)
  - [Writing tests](#writing-tests)
  - [Debugging tests](#debugging-tests)
  - [Performance testing](#performance-testing)
    - [Benchmarking](#benchmarking)
    - [Profiling](#profiling)
  - [Snapshot testing](#snapshot-testing)
- [Branching model](#branching-model)
- [Deprecation policy](#deprecation-policy)
- [Miscellaneous](#miscellaneous)
  - [Locating the root](#locating-the-root)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Requirements & installation

To develop FloPy you must have the following software installed on your machine:

- git
- Python3
- Modflow executables

### Git

You will need [Git](https://git-scm.com) and/or the **GitHub app** (for [Mac](https://mac.github.com) or [Windows](https://windows.github.com)).
GitHub's  [Guide to Installing Git](https://help.github.com/articles/set-up-git) is a good source of information.

### Python

FloPy supports several recent versions of Python, loosely following [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html#implementation).

Install Python >=3.8.1, via [standalone download](https://www.python.org/downloads/) or a distribution like [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). (An [infinite recursion bug](https://github.com/python/cpython/pull/17098) in 3.8.0's [`shutil.copytree`](https://github.com/python/cpython/commit/65c92c5870944b972a879031abd4c20c4f0d7981) can cause test failures if the destination is a subdirectory of the source.)

Then install FloPy and core dependencies from the project root:

    pip install .

The FloPy package has a number of [optional dependencies](.docs/optional_dependencies.md), as well as extra dependencies required for linting, testing, and building documentation. Extra dependencies are listed in the `test`, `lint`, `optional`, and `doc` groups under the `[project.optional-dependencies]` section in `pyproject.toml`. Core, linting, testing and optional dependencies are included in the Conda environment in `etc/environment.yml`. Only core dependencies are included in the PyPI package &mdash; to install extra dependency groups with pip, use `pip install ".[<group>]"`. For instance, to install all development dependencies:

    pip install ".[dev]"

Alternatively, with Anaconda or Miniconda:

    conda env create -f etc/environment.yml
    conda activate flopy

#### Python IDEs

##### Visual Studio Code

VSCode users on Windows may need to run `conda init`, then open a fresh terminal before `conda activate ...` commands are recognized. To set a default Python interpreter and configure IDE terminals to automatically activate the associated environment, add the following to your VSCode's `settings.json`:

```json
{
    "python.defaultInterpreterPath": "/path/to/environment",
    "python.terminal.activateEnvironment": true
}
```

To locate a Conda environment's Python executable, run `where python` with the environment activated.

##### PyCharm

To configure a Python interpreter in PyCharm, navigate to `Settings -> Project -> Python Interpreter`, click the gear icon, then select `Add Interpreter`. This presents a wizard to create a new virtual environment or select an existing one.

### MODFLOW executables

To develop FloPy you will need a number of MODFLOW executables installed.

#### Scripted installation

A utility script is provided to easily download and install executables: after installing FloPy, just run `get-modflow` (see the script's [documentation](.docs/md/get_modflow.md) for more info).

#### Manually installing executables

##### Linux

To download and extract all executables for Linux (e.g., Ubuntu):

```shell
wget https://github.com/MODFLOW-USGS/executables/releases/download/8.0/linux.zip && \
unzip linux.zip -d /path/to/your/install/location
```

Then add the install location to the `PATH`

    export PATH="/path/to/install/location:$PATH"

##### Mac

The same commands should work to download and extract executables for OSX:

```shell
wget https://github.com/MODFLOW-USGS/executables/releases/download/8.0/mac.zip && \
unzip mac.zip -d /path/to/your/install/location
```

Then add the install location to your `PATH`

    export PATH="/path/to/your/install/location:$PATH"

On OSX you may see unidentified developer warnings upon running the executables. To disable warnings and enable permissions for all binaries at once, navigate to the install directory and run

    `for f in *; do xattr -d com.apple.quarantine "$f" && chmod +x "$f"; done;`

When run on OSX, certain tests (e.g., `t032_test.py::test_polygon_from_ij`) may produce errors like

```shell
URLError(SSLCertVerificationError(1, '[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: unable to get local issuer certificate (_ssl.c:1129)'))
```

This can be fixed by running `Install Certificates.command` in your Python installation directory (see the [StackOverflow discussion here](https://stackoverflow.com/a/58525755/6514033) for more information).

### Updating FloPy packages

FloPy must be up-to-date with the version of MODFLOW 6 and other executables it is being used with. Synchronization is achieved via "definition" (DFN) files, which define the format of MODFLOW6 inputs and outputs. FloPy contains Python source code automatically generated from DFN files. This is done with the `generate_classes` function in `flopy.mf6.utils`. See [this document](./.docs/generate_classes.md) for usage examples.

## Examples

A number of examples demonstrating FloPy features are located in `.docs/Notebooks`. These are probably the easiest way to get acquainted with FloPy. Examples are version-controlled as [`jupytext`-managed Python scripts](https://jupytext.readthedocs.io/en/latest/#paired-notebooks). Any `.ipynb` files in `.docs/Notebooks` are ignored by Git. Each script can be invoked by name with Python per usual. The scripts can also be converted to Jupyter notebooks with `jupytext`.

To convert a Python example script to an `.ipynb` notebook, run:

```
jupytext --from py --to ipynb path/to/script.py
```

To work with `.ipynb` notebooks from a browser interface, you will need `jupyter` installed (`jupyter` is included with the `test` optional dependency group in `pyproject.toml`). Some of the notebooks use testing dependencies and [optional dependencies](.docs/optional_dependencies.md) as well. The conda environment provided in `etc/environment.yml` already includes all dependencies needed to run the examples. To install all development dependencies at once using `pip`:

```shell
pip install ".[dev]"
```

To start a local Jupyter notebook server, run:

```shell
jupyter notebook
```

### Developing new examples

Submissions of high-quality examples that demonstrate the use of FloPy are encouraged, as are edits to existing examples to improve the code quality, performance, or clarity of presentation.

There are two kinds of examples: tutorials and full-fledged examples. 

If a script's filename contains "tutorial", it will automatically be assigned to the [Tutorials](https://flopy.readthedocs.io/en/latest/tutorials.html) page on the documentation site.

Tutorials should aim to briefly demonstrate one basic feature. Most tutorials do not create visualizations, so tutorials are simply listed on the documentation site rather than rendered into a thumbnail gallery.

If a script's filename contains "example", it is considered a full-fledged example. These are more broadly scoped than tutorials, and may demonstrate several features at once, typically in the context of a sample model, including pre- and/or post-processing and visualization.

All tutorials and examples should include a header with the following format:

```
# ---
# jupyter
#   jupytext:
#     ...
#   kernelspec:
#     ...
#   metadata:
#     section: ...
#     authors:
#       - name: ......
# ---
```

Contents above the `metadata` attribute can be auto-generated with `jupytext` by first-converting an example script to a notebook, and then back to a script (i.e. a round-trip conversion). For instance:

```shell
jupytext --from py --to ipynb .docs/Notebooks/your_example.py
jupytext --from ipynb --to py .docs/Notebooks/your_example.ipynb
```

The `metadata` attribute should be filled by the example developer, and should contain at minimum:

- `section`: the section within the Tutorials or Examples page
- `authors`: the example's author(s)

The `section` attribute assigns the example to a group within the rendered documentation page. See [the `create_rstfiles.py` script](./.docs/create_rstfiles.py) for a complete list of sections. If your example lacks a `section` attribute, it will be assigned to the "Miscellaneous" section.

**Note**: Examples are rendered into a thumbnail gallery view by [nbsphinx](https://github.com/spatialaudio/nbsphinx) when the [online documentation](https://flopy.readthedocs.io/en/latest/) is built. At least one plot/visualization is recommended in order to provide a thumbnail for each example notebook in the [Examples gallery](https://flopy.readthedocs.io/en/latest/notebooks.html)gallery.

**Note**: Thumbnails for the examples gallery are generated automatically from the notebook header (typically the first line, begining with a single '#'), and by default, the last plot generated. Thumbnails can be customized to use any plot in the notebook, or an external image, as described [here](https://nbsphinx.readthedocs.io/en/0.9.1/subdir/gallery.html).

Each example should create and (attempt to) dispose of its own isolated temporary workspace. On Windows, Python's `TemporaryDirectory` can raise permissions errors, so cleanup is trapped with `try/except`. Some scripts also accept a `--quiet` flag, curtailing verbose output, and a `--keep` option to specify a working directory of the user's choice.

## Tests

To run the tests you will need `pytest` and a few plugins, including [`pytest-xdist`](https://pytest-xdist.readthedocs.io/en/latest/), [`pytest-dotenv`](https://github.com/quiqua/pytest-dotenv), and [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/index.html). Test dependencies are specified in the `test` extras group in `pyproject.toml` (with pip, use `pip install ".[test]"`). Test dependencies are included in the Conda environment `etc/environment`.

**Note:** to prepare your code for a pull request, you will need a few more packages specified in the `lint` extras group in `pyproject.toml` (also included by default for Conda). See the docs on [submitting a pull request](CONTRIBUTING.md) for more info.

### Configuring tests

Some tests require environment variables. Currently the following variables are required:

- `GITHUB_TOKEN`

The `GITHUB_TOKEN` variable is needed because the [`get-modflow`](.docs/md/get_modflow.md) utility invokes the GitHub API &mdash; to avoid rate-limiting, requests to the GitHub API should bear an [authentication token](https://github.com/settings/tokens). A token is automatically provided to GitHub Actions CI jobs via the [`github` context's](https://docs.github.com/en/actions/learn-github-actions/contexts#github-context) `token` attribute, however a personal access token is needed to run the tests locally. To create a personal access token, go to [GitHub -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic)](https://github.com/settings/tokens). The `get-modflow` utility automatically detects and uses the `GITHUB_TOKEN` environment variable if available.

Environment variables can be set as usual, but a more convenient way to store variables for all future sessions is to create a text file called `.env` in the `autotest` directory, containing variables in `NAME=VALUE` format, one on each line. [`pytest-dotenv`](https://github.com/quiqua/pytest-dotenv) will detect and add these to the environment provided to the test process. All `.env` files in the project are ignored in `.gitignore` so there is no danger of checking in secrets unless the file is misnamed.

### Running tests

Tests must be run from the `autotest` directory. To run a single test script in verbose mode:

    pytest -v test_conftest.py

The `test_conftest.py` script tests the test suite's `pytest` configuration. This includes shared fixtures providing a single source of truth for the location of example data, as well as various other fixtures and utilities.

Tests matching a pattern can be run with `-k`, e.g.:

    pytest -v -k "export"

To run all tests in parallel, using however many cores your machine is willing to spare:

    pytest -v -n auto

The `-n auto` option configures the `pytest-xdist` extension to query your computer for the number of processors available. To explicitly set the number of cores, substitute an integer for `auto` in the `-n` argument, e.g. `pytest -v -n 2`. (The space between `-n` and the number of processors can be replaced with `=`, e.g. `-n=2`.)

The above will run all regression tests, benchmarks, and example scripts and notebooks, which can take some time (likely ~30 minutes to an hour, depending on your machine).

#### Selecting tests with markers

Markers are a `pytest` feature that can be used to select subsets of tests. Markers provided in `pytest.ini` include:

- `slow`: tests that don't complete in a few seconds
- `example`: exercise scripts, tutorials and notebooks
- `regression`: tests that compare multiple results

Markers can be used with the `-m <marker>` option. For example, to run only fast tests:

    pytest -v -n auto -m "not slow"

Markers can be applied in boolean combinations with `and` and `not`. For instance, to run fast tests in parallel, excluding example scripts/notebooks and regression tests:

    pytest -v -n auto -m "not slow and not example and not regression"

A CLI option `--smoke` (short form `-S`) is provided as an alias for the above. For instance:

    pytest -v -n auto -S

This should complete in under a minute on most machines. Smoke testing aims to cover a reasonable fraction of the codebase while being fast enough to run often during development. (To preserve this ability, new tests should be marked as slow if they take longer than a second or two to complete.)

**Note:** most the `regression` and `example` tests are `slow`, but there are some other slow tests, e.g. in `test_export.py`, and some regression tests and examples are fast.

### Writing tests

Test functions and files should be named informatively, with related tests grouped in the same file. The test suite runs on GitHub Actions in parallel, so tests should not access the working space of other tests, example scripts, tutorials or notebooks. A number of shared test fixtures are [imported](conftest.py) from [`modflow-devtools`](https://github.com/MODFLOW-USGS/modflow-devtools). These include keepable temporary directory fixtures and miscellanous utilities (see `modflow-devtools` repository README for more information on fixture usage). New tests should use these facilities where possible. See also the [contribution guidelines](CONTRIBUTING.md) before submitting a pull request.

### Debugging tests

To debug a failed test it can be helpful to inspect its output, which is cleaned up automatically by default. `modflow-devtools` provides temporary directory fixtures that allow optionally keeping test outputs in a specified location. To run a test and keep its output, use the `--keep` option to provide a save location:

    pytest test_export.py --keep exports_scratch

This will retain any files created by the test in `exports_scratch` in the current working directory. Any tests using the function-scoped `function_tmpdir` and related fixtures (e.g. `class_tmpdir`, `module_tmpdir`) defined in `modflow_devtools/fixtures` are compatible with this mechanism.

There is also a `--keep-failed <dir>` option which preserves the outputs of failed tests in the given location, however this option is only compatible with function-scoped temporary directories (the `function_tmpdir` fixture).

### Performance testing

Performance testing is accomplished with [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/index.html).

To allow optional separation of performance from correctness concerns, performance test files may be named either as typical test files or may match any of the following patterns:

- `benchmark_*.py`
- `profile_*.py`
- `*_profile*.py`.
- `*_benchmark*.py`

#### Benchmarking

Any test function can be turned into a benchmark by requesting the `benchmark` fixture (i.e. declaring a `benchmark` argument), which can be used to wrap any function call. For instance:

```python
def test_benchmark(benchmark):
    def sleep_1s():
        import time
        time.sleep(1)
        return True

    assert benchmark(sleep_1s)
```

Arguments can be provided to the function as well:

```python
def test_benchmark(benchmark):
    def sleep_s(s):
        import time
        time.sleep(s)
        return True

    assert benchmark(sleep_s, 1)
```

Rather than alter an existing function call to use this syntax, a lambda can be used to wrap the call unmodified:

```python
def test_benchmark(benchmark):
    def sleep_s(s):
        import time
        time.sleep(s)
        return True

    assert benchmark(lambda: sleep_s(1))
```

This can be convenient when the function call is complicated or passes many arguments.

Benchmarked functions are repeated several times (the number of iterations depending on the test's runtime, with faster tests generally getting more reps) to compute summary statistics. To control the number of repetitions and rounds (repetitions of repetitions) use `benchmark.pedantic`, e.g. `benchmark.pedantic(some_function(), iterations=1, rounds=1)`.

Benchmarking is incompatible with `pytest-xdist` and is disabled automatically when tests are run in parallel. When tests are not run in parallel, benchmarking is enabled by default. Benchmarks can be disabled with the `--benchmark-disable` flag.

Benchmark results are only printed to `stdout` by default. To save results to a JSON file, use `--benchmark-autosave`. This will create a `.benchmarks` folder in the current working location (if you're running tests, this should be `autotest/.benchmarks`).

#### Profiling

Profiling is [distinct](https://stackoverflow.com/a/39381805/6514033) from benchmarking in evaluating a program's call stack in detail, while benchmarking just invokes a function repeatedly and computes summary statistics. Profiling is also accomplished with `pytest-benchmark`: use the `--benchmark-cprofile` option when running tests which use the `benchmark` fixture described above. The option's value is the column to sort results by. For instance, to sort by total time, use `--benchmark-cprofile="tottime"`. See the `pytest-benchmark` [docs](https://pytest-benchmark.readthedocs.io/en/stable/usage.html#commandline-options) for more information.

By default, `pytest-benchmark` will only print profiling results to `stdout`. If the `--benchmark-autosave` flag is provided, performance profile data will be included in the JSON files written to the `.benchmarks` save directory as described in the benchmarking section above.

### Snapshot testing

Snapshot testing is a form of regression testing in which a "snapshot" of the results of some computation is verified and captured by the developer to be compared against when tests are subsequently run. This is accomplished with [`syrupy`](https://github.com/tophat/syrupy), which provides a `snapshot` fixture overriding the equality operator to allow comparison with e.g. `snapshot == result`. A few custom fixtures for snapshots of NumPy arrays are provided in `autotest/conftest.py`:

- `array_snapshot`: saves an array in a binary file for compact storage, can be inspected programmatically with `np.load()`
- `text_array_snapshot`: flattens an array and stores it in a text file, compromise between readability and disk usage
- `readable_array_snapshot`: stores an array in a text file in its original shape, easy to inspect but largest on disk

By default, tests run in comparison mode. This means a newly written test using any of the snapshot fixtures will fail until a snapshot is created. Snapshots can be created/updated by running pytest with the `--snapshot-update` flag.

## Branching model

This project follows the [git flow](https://nvie.com/posts/a-successful-git-branching-model/): development occurs on the `develop` branch, while `master` is reserved for the state of the latest release. Development PRs are typically squashed to `develop`, to avoid merge commits. At release time, release branches are merged to `master`, and then `master` is merged back into `develop`.

## Deprecation policy

This project loosely follows [NEP 23](https://numpy.org/neps/nep-0023-backwards-compatibility.html). Basic deprecation policy includes:

- Deprecated features should be removed after at least 1 year or 2 non-patch releases.
- `DeprecationWarning` should be used for features scheduled for removal.
- `FutureWarning` should be used for features whose behavior will change in backwards-incompatible ways.
- Deprecation warning messages should include the deprecation version number (the release in which the deprecation message first appears) to permit timely follow-through later.

See the linked article for more detail.

## Miscellaneous

### Locating the root

Python scripts and notebooks often need to reference files elsewhere in the project. 

To allow scripts to be run from anywhere in the project hierarchy, scripts should locate the project root relative to themselves, then use paths relative to the root for file access, rather than using relative paths (e.g., `../some/path`).

For a script in a subdirectory of the root, for instance, the conventional approach would be:

```Python
project_root_path = Path(__file__).parent.parent
```