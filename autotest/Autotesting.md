## Running autotests

### Requirements
To run the autotests you will need to install `pytest`

    pip install pytest

If you want to run the autotests you should install `pytest-xd`

    pip install pytest pytest-xd

### Running tests
If you want to run a single autotest run the following command from the 
autotest directory

    pytest -v t001_test.py

If you want to run all of the standard autotests (tests that match 
`tXXX_test_*.py`) run the following command from the autotest directory

    pytest -v 

If you want to run all of the autotests the match a pattern run the following
command from the autotest directory

    pytest -v -k "t01"

This would run autotests 10 through 19.


### Running tests in parallel

If you want to run the autotests in parallel add `-n auto` after `-v` in your
`pytest` command. For example,

    pytest -v -n auto t001_test.py

The `auto` keyword indicates that the `pytest-xd` extension will query your 
computer to determine the number of processors available. You can specify a 
specific number of processors to use (for example, `-n 2` to run on two 
processors). 

The space between `-n` and the number of processors can be replaced with a
`=`. For example,

    pytest -v -n=auto t001_test.py


## Creating an autotest

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
from ci_framework import baseTestDir, flopyTest

# set the baseDir variable using the script name
baseDir = baseTestDir(__file__, relPath="temp", verbose=True)

def test_mytest():
    ws = f"{baseDir}_test_mytest"
    testFramework = flopyTest(verbose=True, testDirs=ws, create=True)
    
    ...test something
    
    assert something is True, "oops"
    
    ws = f"{baseDir}_test_mytest_another_directory"
    testFramework.addTestDir(ws, create=True)
    
    ...test something_else
    
    assert something_else is True, "oops"
    
    return

```

Pull requests with new autotests will not be accepted if tests do not follow
the example provided above.
