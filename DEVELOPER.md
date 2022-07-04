# Building and testing FloPy

This document describes how to set up a development environment for FloPy. Details on how to contribute your code to the repository are found in the separate document [CONTRIBUTING.md](CONTRIBUTING.md).

- [Installation](#installation)
- [Running tests](#running-tests)
- [Running example notebooks](#running-example-notebooks)

## Installation

To develop `flopy` you must have the following software installed on your machine.

### Git

[Git](https://git-scm.com) and/or the **GitHub app** (for [Mac](https://mac.github.com) or [Windows](https://windows.github.com)).
[GitHub's Guide to Installing Git](https://help.github.com/articles/set-up-git) is a good source of information.

### Python

Install Python, via [standalone download](https://www.python.org/downloads/) or a distribution like [Anaconda](https://www.anaconda.com/products/individual) or [miniconda](https://docs.conda.io/en/latest/miniconda.html). You will need Python 3.8.1 or greater. (An [infinite recursion bug](https://github.com/python/cpython/pull/17098) in 3.8.0's [`shutil.copytree`](https://github.com/python/cpython/commit/65c92c5870944b972a879031abd4c20c4f0d7981) affects a few tests.)

Then install `flopy` and core dependencies from the project root

    pip install .

Note that `flopy` has a number of [optional dependencies](docs/flopy_method_dependencies.md), as well as dependencies required for running tests. To install all required, testing and optional packages at once, use

    pip install ".[test, optional]"

### MODFLOW executables

To develop `flopy` you will need a number of MODFLOW executables installed. Binaries for all major platforms can be downloaded from the [distribution repository](https://github.com/MODFLOW-USGS/executables).

#### Linux

For example, to download and extract all executables for Linux (e.g., Ubuntu)

```shell
wget https://github.com/MODFLOW-USGS/executables/releases/download/8.0/linux.zip && \
unzip linux.zip -d /path/to/your/install/location
```

Then add the install location to your `PATH`

    export PATH="/path/to/your/install/location:$PATH"

#### Mac

For example, to download and extract all executables for OSX

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

To run the autotests you will need to have `pytest` installed. To run the tests in parallel you will also need `pytest-xdist`. To prepare your code for a pull request, you will need a few more packages (see the docs on [submitting a pull request](CONTRIBUTING.md)).

To install all packages required for testing at once, use

    pip install ".[test]"

If you want to run a single autotest run the following command from the 
`autotest` directory

    pytest -v t001_test.py

If you want to run all of the standard autotests (tests that match 
`tXXX_test_*.py`) run the following command from the `autotest` directory

    pytest -v 

If you want to run all of the autotests the match a pattern run a command like the following
from the `autotest` directory

    pytest -v -k "t01"

This would run autotests 10 through 19.

### Running tests in parallel

If you want to run the autotests in parallel add `-n auto` after `-v` in your
`pytest` command. For example,

    pytest -v -n auto t001_test.py

The `auto` keyword indicates that the `pytest-xdist` extension will query your 
computer to determine the number of processors available. You can specify a 
specific number of processors to use (for example, `-n 2` to run on two 
processors). 

The space between `-n` and the number of processors can be replaced with a
`=`. For example,

    pytest -v -n=auto t001_test.py

### Debugging failed tests

To debug a failed autotest rerun the failed test by running the following command from the autotest directory

    python mffailedtest.py --keep

The `--keep` will retain the test directories created by the test, which will allow the input or output files to be evaluated for errors.

### Creating an autotest

Limit your autotest to a few tests of the same type. See `t002_test.py` for 
an example of how not to create an autotest. This test includes tests for 
loading data from fixed and free format text, loading binary files, using 
`util2D`, and using `util3D`. Preferably all of these tests should have 
been grouped into like tests and put into a separate autotest script.  

The autotests run on GitHub Actions in parallel so new autotests should be
developed so that data written by a specific test is written to a 
unique folder that includes the basename for the script with the test (`t013` 
for `t013_test.py`) and the test name (`t013_mytest`). This can all be done
programatically using functions and classes in `ci_framework` script. An
example of how to construct an autotest is given below

```python
from ci_framework import base_test_dir, FlopyTestSetup

# set the baseDir variable using the script name
base_dir = base_test_dir(__file__, rel_path="temp", verbose=True)

def test_mytest():
    ws = f"{base_dir}_test_mytest"
    test_setup = FlopyTestSetup(verbose=True, test_dirs=ws)
    
    ...test something
    
    assert something is True, "oops"
    
    ws = f"{base_dir}_test_mytest_another_directory"
    test_setup.add_test_dir(ws)
    
    ...test something_else
    
    assert something_else is True, "oops"
    
    return

```

Pull requests with new autotests will not be accepted if tests do not follow
the example provided above. Make sure your pull request also conforms to the [contribution guidelines](CONTRIBUTING.md) before submitting.

## Running example notebooks

To run the example notebooks you will need `jupyter` installed. Some of the notebooks use [optional dependencies](docs/flopy_method_dependencies.md) as well.

To install all optional packages at once, use

    pip install ".[optional]"


