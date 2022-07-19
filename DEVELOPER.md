# Building and testing FloPy

This document describes how to set up a development environment for FloPy. Details on how to contribute your code to the repository are found in the separate document [CONTRIBUTING.md](CONTRIBUTING.md).

- [Installation](#installation)
- [Running tests](#running-tests)
- [Running examples](#running-examples)

## Installation

To develop `flopy` you must have the following software installed on your machine.

### Git

[Git](https://git-scm.com) and/or the **GitHub app** (for [Mac](https://mac.github.com) or [Windows](https://windows.github.com)).
[GitHub's Guide to Installing Git](https://help.github.com/articles/set-up-git) is a good source of information.

### Python

Install Python 3.7.x or >=3.8.1, via [standalone download](https://www.python.org/downloads/) or a distribution like [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). (An [infinite recursion bug](https://github.com/python/cpython/pull/17098) in 3.8.0's [`shutil.copytree`](https://github.com/python/cpython/commit/65c92c5870944b972a879031abd4c20c4f0d7981) affects a few tests.)

Then install `flopy` and core dependencies from the project root:

    pip install .

Alternatively, with Anaconda or Miniconda:

    conda env create -f etc/environment.yml
    conda activate flopy

Note that `flopy` has a number of [optional dependencies](docs/flopy_method_dependencies.md), as well as dependencies required for running tests. All required, testing and optional dependencies are included in the Conda environment in `etc/environment.yml`. Only core dependencies are included in the PyPI package &mdash; to install extra testing, linting and optional packages with pip, use

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

To develop `flopy` you will need a number of MODFLOW executables installed. Binaries for all major platforms can be downloaded from the [distribution repository](https://github.com/MODFLOW-USGS/executables).

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

## Running tests

To run the autotests you will need to have `pytest` installed. To run the tests in parallel you will also need `pytest-xdist`. Testing dependencies are specified in the `test` extras group in `setup.cfg` (with pip, use `pip install ".[test]"`). Test dependencies are included by default in the Conda environment `etc/environment`. To prepare your code for a pull request, you will need a few more packages specified in the `lint` extras group in `setup.cfg` (also included by default for Conda). See the docs on [submitting a pull request](CONTRIBUTING.md) for more info.

If you want to run a single autotest run the following command from the 
`autotest` directory:

    pytest -v t001_test.py

To run all standard tests (tests matching `tXXX_test*.py`) from the `autotest` directory:

    pytest -v 

The `-v` flag enables verbose output. To run all tests matching a pattern run:

    pytest -v -k "t01"

### Running tests in parallel

To run the tests in parallel add `-n auto` to the `pytest` command. For example,

    pytest -v -n auto

The `auto` keyword configures the `pytest-xdist` extension to query your computer for the number of processors available. You can explicitly set the number of cores, substitute an integer for `auto` in the `-n` argument, e.g. `pytest -v -n 2`. (The space between `-n` and the number of processors can be replaced with `=`, e.g. `-n=2`.)

### Debugging failed tests

To debug a failed test it can be helpful to inspect its output, which is cleaned up automatically by default. To run a failing test and keep its output, use the `--keep` flag:

    python t001_test.py --keep

This will retain the test directories created by the test, which will allow the input or output files to be evaluated for errors.

A similar `-keep` argument is also available for `pytest`. Tests using the function-scoped `tmpdir` and related fixtures (e.g. `class_tmpdir`, `module_tmpdir`) defined in `conftest.py` are compatible with this mechanism, which differs from the Python CLI argument only in that the location to save test outputs must be provided explicitly, e.g.

    pytest t001_test.py --keep t001_output

### Creating an autotest

Limit your autotest to a few tests of the same type. See `t002_test.py` for 
an example of how not to create an autotest. This test includes tests for 
loading data from fixed and free format text, loading binary files, using 
`util2D`, and using `util3D`. Preferably all of these tests should have 
been grouped into like tests and put into a separate autotest script.  

The autotests run on GitHub Actions in parallel. Tests must not pollute the working space of other tests. New tests should write to a unique, appropriately named folder (e.g.,`t013` for test cases `t013_test.py`) and the test name (`t013_mytest`). This can be accomplished programmatically with the `FlopyTestSetup` class in `ci_framework` script:

```python
from pathlib import Path
from ci_framework import base_test_dir, FlopyTestSetup

# set the baseDir variable using the script name
base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)

def test_mytest():
    ws = f"{base_dir}_test_mytest"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=ws)
    
    assert Path(ws).is_dir()
    
    ws = f"{base_dir}_test_mytest_another_directory"
    test_setup.add_test_dir(ws)
    
    assert Path(ws).is_dir()
```

An alternative compatible with the `pytest --keep` argument is to use the `tmpdir` fixture (or its equivalent for the appropriate scope) defined in `conftest.py`. These fixtures can be substituted transparently for `pytest`'s built-in `tmp_path`. When the `--keep` argument is provided, e.g. `pytest --keep temp`, outputs will automatically be saved to subdirectories of `temp` named according to the test case, class or module.

```python
from pathlib import Path
import inspect

def test_something(tmpdir, module_tmpdir):
    # function-scoped temporary directory
    assert isinstance(tmpdir, Path)
    assert tmpdir.is_dir()
    assert inspect.currentframe().f_code.co_name in tmpdir.stem

    # module-scoped temp dir (accessible to other tests in the script)
    assert module_tmpdir.is_dir()
    assert Path(__file__).stem in module_tmpdir.stem
```

Any new tests should use one of these facilities to prevent the proliferation of test outputs and untracked files (see also the [contribution guidelines](CONTRIBUTING.md) for before submitting a pull request).

## Running examples

A number of scripts and notebooks to demonstrate `flopy` usage are located in `examples/`.

### Scripts

[Example scripts](docs/script_examples.md) are in `examples/scripts` and `examples/Tutorials`.

Some of the scripts use [optional dependencies](docs/flopy_method_dependencies.md).

### Notebooks

[Example notebooks](docs/notebook_examples.md) are located in `examples/Notebooks`.

To run the example notebooks you will need `jupyter` installed. Some of the notebooks use [optional dependencies](docs/flopy_method_dependencies.md) as well.

To install jupyter and all optional dependencies at once, use

    pip install jupyter ".[optional]"

To start a local Jupyter notebook server, run

    jupyter notebook
