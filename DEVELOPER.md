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
  - [Running tests](#running-tests)
    - [Selecting tests with markers](#selecting-tests-with-markers)
  - [Debugging tests](#debugging-tests)
  - [Benchmarking](#benchmarking)
  - [Writing tests](#writing-tests)
    - [Keepable temporary directories](#keepable-temporary-directories)
    - [Locating example data](#locating-example-data)
    - [Locating the project root](#locating-the-project-root)
    - [Conditionally skipping tests](#conditionally-skipping-tests)
  - [Miscellaneous](#miscellaneous)
    - [Generating TOCs with `doctoc`](#generating-tocs-with-doctoc)
    - [Testing CI workflows with `act`](#testing-ci-workflows-with-act)

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

Install Python 3.7.x or >=3.8.1, via [standalone download](https://www.python.org/downloads/) or a distribution like [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). (An [infinite recursion bug](https://github.com/python/cpython/pull/17098) in 3.8.0's [`shutil.copytree`](https://github.com/python/cpython/commit/65c92c5870944b972a879031abd4c20c4f0d7981) can cause test failures if the destination is a subdirectory of the source.)

Then install `flopy` and core dependencies from the project root:

    pip install .

Alternatively, with Anaconda or Miniconda:

    conda env create -f etc/environment.yml
    conda activate flopy

Note that `flopy` has a number of [optional dependencies](docs/flopy_method_dependencies.md), as well as dependencies required for linting, testing, and building documentation. All required, linting, testing and optional dependencies are included in the Conda environment in `etc/environment.yml`. Only core dependencies are included in the PyPI package &mdash; to install extra testing, linting and optional packages with pip, use

    pip install ".[test, lint, optional]"

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

To run the example notebooks you will need `jupyter` installed (`jupyter` is included with the `test` optional dependency group in `setup.cfg`). Some of the notebooks use [optional dependencies](docs/flopy_method_dependencies.md) as well.

To install jupyter and optional dependencies at once:

    pip install jupyter ".[optional]"

To start a local Jupyter notebook server, run

    jupyter notebook

Like the scripts and tutorials, each notebook is configured to create and (attempt to) dispose of its own isolated temporary workspace. (On Windows, Python's `TemporaryDirectory` can raise permissions errors, so cleanup is trapped with `try/except`.)

## Tests

To run the tests you will need `pytest` and a few plugins, including [`pytest-xdist`](https://pytest-xdist.readthedocs.io/en/latest/) and [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/index.html). Test dependencies are specified in the `test` extras group in `setup.cfg` (with pip, use `pip install ".[test]"`). Test dependencies are included in the Conda environment `etc/environment`.

**Note:** to prepare your code for a pull request, you will need a few more packages specified in the `lint` extras group in `setup.cfg` (also included by default for Conda). See the docs on [submitting a pull request](CONTRIBUTING.md) for more info.

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

### Debugging tests

To debug a failed test it can be helpful to inspect its output, which is cleaned up automatically by default. To run a failing test and keep its output, use the `--keep` option to provide a save location:

    pytest test_export.py --keep exports_scratch

This will retain the test directories created by the test, which allows files to be evaluated for errors. Any tests using the function-scoped `tmpdir` and related fixtures (e.g. `class_tmpdir`, `module_tmpdir`) defined in `conftest.py` are compatible with this mechanism.

There is also a `--keep-failed <dir>` option which preserves the outputs of failed tests in the given location, however this option is only compatible with function-scoped temporary directories (the `tmpdir` fixture defined in `conftest.py`).

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

### Writing tests

Test functions and files should be named informatively, with related tests grouped in the same file. The test suite runs on GitHub Actions in parallel, so tests must not pollute the working space of other tests, example scripts, tutorials or notebooks. A number of shared test fixtures are provided in `autotest/conftest.py`. New tests should use these facilities where possible, to standardize conventions, help keep maintenance minimal, and prevent shared test state and proliferation of untracked files. See also the [contribution guidelines](CONTRIBUTING.md) before submitting a pull request.

#### Keepable temporary directories

The `tmpdir` fixtures defined in `conftest.py` provide a path to a temporary directory which is automatically created before test code runs and automatically removed afterwards. (The builtin `pytest` `temp_path` fixture can also be used, but is not compatible with the `--keep` command line argument detailed above.)

For instance, using temporary directory fixtures for various scopes:

```python
from pathlib import Path
import inspect

def test_tmpdirs(tmpdir, module_tmpdir):
    # function-scoped temporary directory
    assert isinstance(tmpdir, Path)
    assert tmpdir.is_dir()
    assert inspect.currentframe().f_code.co_name in tmpdir.stem

    # module-scoped temp dir (accessible to other tests in the script)
    assert module_tmpdir.is_dir()
    assert "autotest" in module_tmpdir.stem
```

These fixtures can be substituted transparently for `pytest`'s built-in `tmp_path`, with the additional benefit that when `pytest` is invoked with the `--keep` argument, e.g. `pytest --keep temp`, outputs will automatically be saved to subdirectories of `temp` named according to the test case, class or module. (As described above, this is useful for debugging a failed test by inspecting its outputs, which would otherwise be cleaned up.)

#### Locating example data

Shared fixtures and utility functions are also provided for locating example data on disk. The `example_data_path` fixture resolves to `examples/data` relative to the project root, regardless of the location of the test script (as long as it's somewhere in the `autotest` directory).

```python
def test_with_data(tmpdir, example_data_path):
    model_path = example_data_path / "freyberg"
    # load model...
```

This is preferable to manually handling relative paths as if the location of the example data changes in the future, only a single fixture in `conftest.py` will need to be updated rather than every test case individually.

An equivalent function `get_example_data_path()` is also provided in `conftest.py`. This is useful to dynamically generate data for test parametrization. (Due to a [longstanding `pytest` limitation](https://github.com/pytest-dev/pytest/issues/349), fixtures cannot be used to generate test parameters.) 

#### Locating the project root

A similar `get_project_root_path()` function is also provided, doing what it says on the tin:

```python
from autotest.conftest import get_project_root_path, get_example_data_path

def test_get_paths():
    example_data = get_example_data_path()
    project_root = get_project_root_path()

    assert example_data.parent.parent == project_root
```

Note that this function expects tests to be run from the `autotest` directory, as mentioned above.

#### Conditionally skipping tests

Several `pytest` markers are provided to conditionally skip tests based on executable availability, Python package environment or operating system.

To skip tests if one or more executables are not available on the path:

```python
from shutil import which
from autotest.conftest import requires_exe

@requires_exe("mf6")
def test_mf6():
    assert which("mf6")

@requires_exe("mf6", "mp7")
def test_mf6_and_mp7():
    assert which("mf6")
    assert which("mp7")
```

To skip tests if one or more Python packages are not available:

```python
from autotest.conftest import requires_pkg

@requires_pkg("pandas")
def test_needs_pandas():
    import pandas as pd

@requires_pkg("pandas", "shapefile")
def test_needs_pandas():
    import pandas as pd
    from shapefile import Reader
```

To mark tests requiring or incompatible with particular operating systems:

```python
import os
import platform
from autotest.conftest import requires_platform, excludes_platform

@requires_platform("Windows")
def test_needs_windows():
    assert platform.system() == "Windows"

@excludes_platform("Darwin", ci_only=True)
def test_breaks_osx_ci():
    if "CI" in os.environ:
        assert platform.system() != "Darwin"
```

Platforms must be specified as returned by `platform.system()`.

Both these markers accept a `ci_only` flag, which indicates whether the policy should only apply when the test is running on GitHub Actions CI.

There is also a `@requires_github` marker, which will skip decorated tests if the GitHub API is unreachable.

### Miscellaneous

A few other useful tools for FloPy development include:

- [`doctoc`](https://www.npmjs.com/package/doctoc): automatically generate table of contents sections for markdown files
- [`act`](https://github.com/nektos/act): test GitHub Actions workflows locally (requires Docker)

#### Generating TOCs with `doctoc`

The [`doctoc`](https://www.npmjs.com/package/doctoc) tool can be used to automatically generate table of contents sections for markdown files. `doctoc` is distributed with the [Node Package Manager](https://docs.npmjs.com/cli/v7/configuring-npm/install). With Node installed use `npm install -g doctoc` to install `doctoc` globally. Then just run `doctoc <file>`, e.g.:

```shell
doctoc DEVELOPER.md
```

This will insert HTML comments surrounding an automatically edited region, scanning for headers and creating an appropriately indented TOC tree.  Subsequent runs are idempotent, updating if the file has changed or leaving it untouched if not.

To run `doctoc` for all markdown files in a particular directory (recursive), use `doctoc some/path`.

#### Testing CI workflows with `act`

The [`act`](https://github.com/nektos/act) tool uses Docker to run containerized CI workflows in a simulated GitHub Actions environment. [Docker Desktop](https://www.docker.com/products/docker-desktop/) is required for Mac or Windows and [Docker Engine](https://docs.docker.com/engine/) on Linux.

With Docker installed and running, run `act -l` from the project root to see available CI workflows. To run all workflows and jobs, just run `act`. To run a particular workflow use `-W`:

```shell
act -W .github/workflows/commit.yml
```

To run a particular job within a workflow, add the `-j` option:

```shell
act -W .github/workflows/commit.yml -j build
```

**Note:** GitHub API rate limits are easy to exceed, especially with job matrices. Authenticated GitHub users have a much higher rate limit: use `-s GITHUB_TOKEN=<your token>` when invoking `act` to provide a personal access token. Note that this will log your token in shell history &mdash; leave the value blank for a prompt to enter it more securely.

The `-n` flag can be used to execute a dry run, which doesn't run anything, just evaluates workflow, job and step definitions. See the [docs](https://github.com/nektos/act#example-commands) for more.

**Note:** `act` can only run Linux-based container definitions, so Mac or Windows workflows or matrix OS entries will be skipped.
