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
- [Examples](#examples)
  - [Scripts](#scripts)
  - [Notebooks](#notebooks)
- [Tests](#tests)
  - [Configuring tests](#configuring-tests)
  - [Running tests](#running-tests)
    - [Selecting tests with markers](#selecting-tests-with-markers)
  - [Writing tests](#writing-tests)
  - [Debugging tests](#debugging-tests)
  - [Performance testing](#performance-testing)
    - [Benchmarking](#benchmarking)
    - [Profiling](#profiling)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

## Requirements & installation

To develop `flopy` you must have the following software installed on your machine:

- git
- Python3
- Modflow executables

### Git

You will need [Git](https://git-scm.com) and/or the **GitHub app** (for [Mac](https://mac.github.com) or [Windows](https://windows.github.com)).
GitHub's  [Guide to Installing Git](https://help.github.com/articles/set-up-git) is a good source of information.

### Python

FloPy supports several recent versions of Python, loosely following [NEP 29](https://numpy.org/neps/nep-0029-deprecation_policy.html#implementation).

Install Python >=3.8.1, via [standalone download](https://www.python.org/downloads/) or a distribution like [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). (An [infinite recursion bug](https://github.com/python/cpython/pull/17098) in 3.8.0's [`shutil.copytree`](https://github.com/python/cpython/commit/65c92c5870944b972a879031abd4c20c4f0d7981) can cause test failures if the destination is a subdirectory of the source.)

Then install `flopy` and core dependencies from the project root:

    pip install .

Alternatively, with Anaconda or Miniconda:

    conda env create -f etc/environment.yml
    conda activate flopy

The `flopy` package has a number of [optional dependencies](docs/flopy_method_dependencies.md), as well as extra dependencies required for linting, testing, and building documentation. Extra dependencies are listed in the `test`, `lint`, `optional`, and `doc` groups under the `[project.optional-dependencies]` section in `pyproject.toml`. Core, linting, testing and optional dependencies are included in the Conda environment in `etc/environment.yml`. Only core dependencies are included in the PyPI package &mdash; to install extra dependency groups with pip, use `pip install ".[<group>]"`. For instance, to install all extra dependency groups:

    pip install ".[test, lint, optional, doc]"

#### Python IDEs

##### Visual Studio Code

VSCode users on Windows may need to run `conda init`, then open a fresh terminal before `conda activate ...` commands are recognized. To set a default Python interpreter and configure IDE terminals to automatically activate the associated environment, add the following to your VSCode's `settings.json`:

```json
{
    "python.defaultInterpreterPath": "/path/to/your/virtual/environment",
    "python.terminal.activateEnvironment": true
}
```

To locate a Conda environment's Python executable, run `where python` with the environment activated.

##### PyCharm

To configure a Python interpreter in PyCharm, navigate to `Settings -> Project -> Python Interpreter`, click the gear icon, then select `Add Interpreter`. This presents a wizard to create a new virtual environment or select an existing one.

### MODFLOW executables

To develop `flopy` you will need a number of MODFLOW executables installed.

#### Scripted installation

A utility script is provided to easily download and install executables: after installing `flopy`, just run `get-modflow` (see the script's [documentation](docs/get_modflow.md) for more info).

#### Manually installing executables

##### Linux

To download and extract all executables for Linux (e.g., Ubuntu):

```shell
wget https://github.com/MODFLOW-USGS/executables/releases/download/8.0/linux.zip && \
unzip linux.zip -d /path/to/your/install/location
```

Then add the install location to your `PATH`

    export PATH="/path/to/your/install/location:$PATH"

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

## Examples

A number of scripts and notebooks demonstrating various `flopy` functions and features are located in `examples/`. These are probably the easiest way to get acquainted with `flopy`.

### Scripts

[Example scripts](docs/script_examples.md) are in `examples/scripts` and `examples/Tutorials`. Each can be invoked by name with Python per usual. By default, all scripts create and (attempt to) clean up temporary working directories. (On Windows, Python's `TemporaryDirectory` can raise permissions errors, so cleanup is trapped with `try/except`.) Some scripts also accept a `--quiet` flag, curtailing verbose output, and a `--keep` option to specify a working directory of the user's choice.

Some of the scripts use [optional dependencies](docs/flopy_method_dependencies.md). If you're using `pip` make sure these have been installed with `pip install ".[optional]"`. The conda environment provided in `etc/environment.yml` already includes all dependencies.

### Notebooks

[Example notebooks](docs/notebook_examples.md) are located in `examples/Notebooks`.

To run the example notebooks you will need `jupyter` installed (`jupyter` is included with the `test` optional dependency group in `pyproject.toml`). Some of the notebooks use [optional dependencies](docs/flopy_method_dependencies.md) as well.

To install jupyter and optional dependencies at once:

    pip install jupyter ".[optional]"

To start a local Jupyter notebook server, run

    jupyter notebook

Like the scripts and tutorials, each notebook is configured to create and (attempt to) dispose of its own isolated temporary workspace. (On Windows, Python's `TemporaryDirectory` can raise permissions errors, so cleanup is trapped with `try/except`.)

## Tests

To run the tests you will need `pytest` and a few plugins, including [`pytest-xdist`](https://pytest-xdist.readthedocs.io/en/latest/), [`pytest-dotenv`](https://github.com/quiqua/pytest-dotenv), and [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/index.html). Test dependencies are specified in the `test` extras group in `pyproject.toml` (with pip, use `pip install ".[test]"`). Test dependencies are included in the Conda environment `etc/environment`.

**Note:** to prepare your code for a pull request, you will need a few more packages specified in the `lint` extras group in `pyproject.toml` (also included by default for Conda). See the docs on [submitting a pull request](CONTRIBUTING.md) for more info.

### Configuring tests

Some tests require environment variables. Currently the following variables are required:

- `GITHUB_TOKEN`

The `GITHUB_TOKEN` variable is needed because the [`get-modflow`](docs/get_modflow.md) utility invokes the GitHub API &mdash; to avoid rate-limiting, requests to the GitHub API should bear an [authentication token](https://github.com/settings/tokens). A token is automatically provided to GitHub Actions CI jobs via the [`github` context's](https://docs.github.com/en/actions/learn-github-actions/contexts#github-context) `token` attribute, however a personal access token is needed to run the tests locally. To create a personal access token, go to [GitHub -> Settings -> Developer settings -> Personal access tokens -> Tokens (classic)](https://github.com/settings/tokens). The `get-modflow` utility automatically detects and uses the `GITHUB_TOKEN` environment variable if available.

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
