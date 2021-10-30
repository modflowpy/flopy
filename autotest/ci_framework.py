import os
import sys
import shutil
import pymake

# command line arguments to:
#   1. keep (--keep) test files
#
for idx, arg in enumerate(sys.argv):
    if "--keep" in arg.lower():
        keep = True
    else:
        keep = False


def get_parent_path():
    """
    Get the relative path to the autotest directory

    Returns
    -------
    parent_path : str
        path to the autotest directory
    """
    if os.path.isdir("autotest"):
        parent_path = "autotest"
    elif os.path.isfile(__file__):
        parent_path = "."
    else:
        parent_path = None
    return parent_path


def baseTestDir(
    filePath, clean=False, create=False, relPath=".", verbose=False
):
    """
    Create test directory name from script name. The assumption is the
    file name is unique, starts with "t", and can be terminated at the first
    underscore.

    Parameters
    ----------
    filePath : file path
        autotest file path
    clean : bool
        boolean indicating if an existing directory should be cleaned
    create : bool
        boolean indicating if the directory should be created
    relPath : str
        path to where the test directory should be located
    verbose : bool
        boolean indicating if diagnostic information should be written
        to the screen

    Returns
    -------
    baseDir : str
        base test directory to create for the autotest

    Example
    -------

    >>> from ci_framework import baseTestDir
    >>> baseDir = baseTestDir(__file__, relPath="temp", create=True)
    >>> print(f"baseDir: {baseDir}")

    """
    fileName = os.path.basename(filePath)
    if not fileName.startswith("t"):
        raise ValueError(
            f"fileName '{fileName}', which is derived from "
            f"'{filePath}', must start with a 't'"
        )
    # parse the file name
    index1 = fileName.index("_")
    if index1 == 0:
        index1 = len(fileName)
    # construct the base directory with the relative path
    baseDir = os.path.join(relPath, fileName[:index1])

    # create the TestDir, if necessary
    if create:
        createTestDir(baseDir, clean=clean, verbose=verbose)

    return baseDir


def createTestDir(testDir, clean=False, verbose=False):
    """
    Create a test directory with the option to remove it first

    Parameters
    ----------
    testDir : str
        path of directory to create
    clean : bool
        boolean indicating if an existing directory should be cleaned
    verbose : bool
        boolean indicating if diagnostic information should be written
        to the screen

    Example
    -------

    >>> from ci_framework import createTestDir
    >>> createTestDir("temp/mydir", clean=True, verbose=True)

    """
    if clean:
        _cleanDir(testDir, verbose=verbose)
    if not os.path.isdir(testDir):
        os.makedirs(testDir, exist_ok=True)
        if verbose:
            print(f"creating test directory...'{testDir}'")


class flopyTest(object):
    """
    The flopyTest class is used to setup test directories for flopy
    autotests.

    Attributes
    ----------
    clean : bool
        boolean indicating if an existing directory should be cleaned
    create : bool
        boolean indicating if the directory should be created
    testDirs : str or list/tuple of strings
        path to where the test directory should be located
    verbose : bool
        boolean indicating if diagnostic information should be written
        to the screen
    retain : bool
        boolean indicating if the test files should be retained

    Methods
    -------
    addTestDir(testDirs, clean=False, create=False)
        Add a testDir or a list of testDirs to the object

    Example
    -------

    >>> from ci_framework import flopyTest
    >>> def test_function():
    ...     testFramework = flopyTest(verbose=True, testDirs="temp/t091_01")
    ...     testFramework.addTestDir("temp/t091_02", create=True)

    """

    def __init__(
        self,
        clean=False,
        create=False,
        testDirs=None,
        verbose=False,
        retain=None,
    ):
        if retain is None:
            retain = keep
        self._clean = clean
        self._verbose = verbose
        self._createDirs = create
        self._retain = retain
        self._testDirs = []
        if testDirs is not None:
            self.addTestDir(testDirs, clean=clean, create=create)

    def __del__(self):
        if not self._retain:
            for testDir in self._testDirs:
                _cleanDir(testDir, verbose=self._verbose)
        else:
            print("Retaining test files")

    def addTestDir(self, testDirs, clean=False, create=False):
        """
        Add a test directory to the flopyTest object.

        Parameters
        ----------
        testDirs : str or list/tuple of strings
            path to where the test directory should be located
        clean : bool
            boolean indicating if an existing directory should be cleaned
        create : bool
            boolean indicating if the directory should be created

        """
        if isinstance(testDirs, str):
            testDirs = [testDirs]
        elif isinstance(testDirs, (int, float, bool)):
            raise ValueError(
                f"testDir '{testDirs}' must be a string, "
                "list of strings, or tuple of strings."
            )
        for testDir in testDirs:
            if testDir not in self._testDirs:
                self._testDirs.append(testDir)
                if self._verbose:
                    print(f"adding test directory...{testDir}")
                if create:
                    createTestDir(testDir, clean=clean, verbose=self._verbose)


def _get_mf6path():
    """
    Get the path for the MODFLOW 6 example problems

    Returns
    -------
    mf6pth : str
        path to the directory containing the MODFLOW 6 example problems.

    """
    parentPath = get_parent_path()
    if parentPath is None:
        parentPath = "."
    dirName = "mf6examples"
    dstpth = os.path.join(parentPath, "temp", dirName)
    return os.path.abspath(dstpth)


def download_mf6_examples(delete_existing=False):
    """
    Download mf6 examples and return location of folder

    Parameters
    ----------
    delete_existing : bool
        boolean flag indicating to delete the existing MODFLOW 6 example
        directory (temp/mf6examples), if it exists.

    Returns
    -------
    mf6pth : str
        path to the directory containing the MODFLOW 6 example problems.

    """
    # save current directory
    cpth = os.getcwd()

    # create folder for mf6 distribution download
    dstpth = _get_mf6path()

    # delete the existing examples
    clean = False
    if delete_existing:
        clean = True

    # download the MODFLOW 6 distribution does not exist
    if clean or not os.path.isdir(dstpth):
        print(f"create...{dstpth}")
        createTestDir(dstpth, clean=clean, verbose=True)

        # Download the distribution
        url = (
            "https://github.com/MODFLOW-USGS/modflow6-examples/releases/"
            "download/current/modflow6-examples.zip"
        )
        pymake.download_and_unzip(
            url=url,
            pth=dstpth,
            verify=True,
        )

        # change back to original path
        os.chdir(cpth)

    # return the absolute path to the distribution
    return dstpth


# private functions


def _cleanDir(testDir, verbose=False):
    """
    Delete a test directory

    Parameters
    ----------
    testDir : str
        path of directory to create
    clean : bool
        boolean indicating if an existing directory should be cleaned

    Returns
    -------

    """

    if os.path.isdir(testDir):
        if verbose:
            print(f"removing test directory...'{testDir}'")

        # remove the tree
        shutil.rmtree(testDir, ignore_errors=True)
