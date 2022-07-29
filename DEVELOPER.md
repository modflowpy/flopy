# Building and testing FloPy

This document describes how to set up a development environment for FloPy. Details on how to contribute your code to the repository are found in the separate document [CONTRIBUTING.md](CONTRIBUTING.md).

- [Installation](#installation)
- [Examples](#examples)
- [Tests](#tests)

## Installation

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

#### IDE configuration

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

#### Linux

To download and extract all executables for Linux (e.g., Ubuntu):

```shell
wget https://github.com/MODFLOW-USGS/executables/releases/download/8.0/linux.zip && \
unzip linux.zip -d /path/to/your/install/location
```

Then add the install location to your `PATH`

    export PATH="/path/to/your/install/location:$PATH"

#### Mac

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

The above will run all regression tests, benchmarks, and example scripts and notebooks, which can take some time (likely ~30 minutes to an hour, depending on your machine). To run only fast tests with benchmarking disabled:

    pytest -v -n auto -m "not slow" --benchmark-disable

Fast tests should complete in under a minute on most machines.

A marker `slow` is used above to select a subset of tests. These can be applied in boolean combinations with `and` and `not`. A few more `pytest` markers are provided:

- `regression`: tests comparing the output of multiple runs
- `example`: example scripts, tutorials, and notebooks

Most of the `regression` and `example` tests are also `slow`, however there are some other slow tests, especially in `test_export.py`, and some regression tests are fairly fast.

### Benchmarking

Benchmarking is accomplished with the [`pytest-benchmark`](https://pytest-benchmark.readthedocs.io/en/latest/index.html) plugin. If the `--benchmark-disable` flag is not provided when `pytest` is invoked, benchmarking is enabled and some tests will be repeated several times to establish a performance profile. Benchmarked tests can be identified by the `benchmark` fixture used in the test signature. By default, two kinds of tests are benchmarked:

- model-loading tests
- regression tests

To save benchmarking results to a JSON file, use the `--benchmark-autosave` flag. By default, this will create a `.benchmark` directory in `autotest`.

### Debugging failed tests

To debug a failed test it can be helpful to inspect its output, which is cleaned up automatically by default. To run a failing test and keep its output, use the `--keep` option to provide a save location:

    pytest test_export.py --keep exports_scratch

This will retain the test directories created by the test, which allows files to be evaluated for errors. Any tests using the function-scoped `tmpdir` and related fixtures (e.g. `class_tmpdir`, `module_tmpdir`) defined in `conftest.py` are compatible with this mechanism.

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

An equivalent function `get_example_data_path(path=None)` is also provided in `conftest.py`. This is useful to dynamically generate data for test parametrization. (Due to a [longstanding `pytest` limitation](https://github.com/pytest-dev/pytest/issues/349), fixtures cannot be used to generate test parameters.) This function accepts a path hint, taken as the path to the current test file, but will try to locate the example data even if the current file is not provided.

```python
import pytest
from autotest.conftest import get_example_data_path

# current test script can be provided (or not)

@pytest.mark.parametrize("current_path", [__file__, None])
def test_get_example_data_path(current_path):
    parts = get_example_data_path(current_path).parts
    assert (parts[-1] == "data" and
            parts[-2] == "examples" and
            parts[-3] == "flopy")
```

#### Locating the project root

A similar `get_project_root_path(path=None)` function is also provided, doing what it says on the tin:

```python
from autotest.conftest import get_project_root_path, get_example_data_path

def test_get_paths():
    example_data = get_example_data_path(__file__)
    project_root = get_project_root_path(__file__)

    assert example_data.parent.parent == project_root
```

#### Conditionally skipping tests

Several `pytest` markers are provided to conditionally skip tests based on executable availability or operating system.

To skip tests if an executable is not available on the path:

```python
from shutil import which
from autotest.conftest import requires_exe

@requires_exe("mf6")
def test_mf6():
    assert which("mf6")
```

A variant for multiple executables is also provided:

```python
from shutil import which
from autotest.conftest import requires_exes

exes = ["mfusg", "mfnwt"]

@requires_exes(exes)
def test_mfusg_and_mfnwt():
    assert all(which(exe) for exe in exes)
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

These both accept a `ci_only` flag, which indicates whether the policy should only apply when the test is running on GitHub Actions CI.

There is also a `@requires_github` marker, which will skip decorated tests if the GitHub API is unreachable.
