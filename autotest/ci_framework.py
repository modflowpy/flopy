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


def base_test_dir(
    file_path, clean=False, create=False, rel_path=".", verbose=False
):
    """
    Create test directory name from script name. The assumption is the
    file name is unique, starts with "t", and can be terminated at the first
    underscore.

    Parameters
    ----------
    file_path : file path
        autotest file path
    clean : bool
        boolean indicating if an existing directory should be cleaned
    create : bool
        boolean indicating if the directory should be created
    rel_path : str
        path to where the test directory should be located
    verbose : bool
        boolean indicating if diagnostic information should be written
        to the screen

    Returns
    -------
    base_dir : str
        base test directory to create for the autotest

    Example
    -------

    >>> from ci_framework import base_test_dir
    >>> base_dir = basetest_dir(__file__, rel_path="temp")
    >>> print(f"base_dir: {base_dir}")

    """
    fileName = os.path.basename(file_path)
    if not fileName.startswith("t"):
        raise ValueError(
            f"fileName '{fileName}', which is derived from "
            f"'{file_path}', must start with a 't'"
        )
    # parse the file name
    index1 = fileName.index("_")
    if index1 == 0:
        index1 = len(fileName)
    # construct the base directory with the relative path
    base_dir = os.path.join(rel_path, fileName[:index1])

    # create the test_dir, if necessary
    if create:
        create_test_dir(base_dir, clean=clean, verbose=verbose)

    return base_dir


def create_test_dir(test_dir, clean=False, verbose=False):
    """
    Create a test directory with the option to remove it first

    Parameters
    ----------
    test_dir : str
        path of directory to create
    clean : bool
        boolean indicating if an existing directory should be cleaned
    verbose : bool
        boolean indicating if diagnostic information should be written
        to the screen

    Example
    -------

    >>> from ci_framework import createtest_dir
    >>> createtest_dir("temp/mydir", clean=True, verbose=True)

    """
    if clean:
        _clean_dir(test_dir, verbose=verbose)
    if not os.path.isdir(test_dir):
        os.makedirs(test_dir, exist_ok=True)
        if verbose:
            print(f"creating test directory...'{test_dir}'")


class FlopyTestSetup(object):
    """
    The flopyTest class is used to setup test directories for flopy
    autotests.

    Attributes
    ----------
    clean : bool
        boolean indicating if an existing directory should be cleaned
    test_dirs : str or list/tuple of strings
        path to where the test directory should be located
    verbose : bool
        boolean indicating if diagnostic information should be written
        to the screen
    retain : bool
        boolean indicating if the test files should be retained

    Methods
    -------
    addtest_dir(test_dirs, clean=False, create=False)
        Add a test_dir or a list of test_dirs to the object

    Example
    -------

    >>> from ci_framework import FlopyTestSetup
    >>> def test_function():
    ...     test_setup = flopyTest(verbose=True, test_dirs="temp/t091_01")
    ...     test_setup.add_test_dir("temp/t091_02")

    """

    def __init__(
        self,
        clean=False,
        test_dirs=None,
        verbose=False,
        retain=None,
    ):
        if retain is None:
            retain = keep
        self._clean = clean
        self._verbose = verbose
        self._retain = retain
        self._test_dirs = []
        if test_dirs is not None:
            self.add_test_dir(test_dirs, clean=clean)

    def __del__(self):
        if not self._retain:
            for test_dir in self._test_dirs:
                _clean_dir(test_dir, verbose=self._verbose)
        else:
            print("Retaining test files")

    def add_test_dir(self, test_dirs, clean=False):
        """
        Add a test directory to the flopyTest object.

        Parameters
        ----------
        test_dirs : str or list/tuple of strings
            path to where the test directory should be located
        clean : bool
            boolean indicating if an existing directory should be cleaned

        """
        if isinstance(test_dirs, str):
            test_dirs = [test_dirs]
        elif isinstance(test_dirs, (int, float, bool)):
            raise ValueError(
                f"test_dir '{test_dirs}' must be a string, "
                "list of strings, or tuple of strings."
            )
        for test_dir in test_dirs:
            if test_dir not in self._test_dirs:
                self._test_dirs.append(test_dir)
                if self._verbose:
                    print(f"adding test directory...{test_dir}")
                create_test_dir(test_dir, clean=clean, verbose=self._verbose)

    def save_as_artifact(self):
        """
        Save test folder in directory ./failedTests.  When run as CI
        the ./failedTests folder will be stored as a job artifact.

        """
        for test_dir in self._test_dirs:
            dirname = os.path.split(test_dir)[-1]
            dst = f"./failedTests/{dirname}"
            print(f"archiving test folder {dirname} in {dst}")
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(test_dir, dst)
        return

def _get_mf6path():
    """
    Get the path for the MODFLOW 6 example problems

    Returns
    -------
    mf6pth : str
        path to the directory containing the MODFLOW 6 example problems.

    """
    parent_path = get_parent_path()
    if parent_path is None:
        parent_path = "."
    dirName = "mf6examples"
    dstpth = os.path.join(parent_path, "temp", dirName)
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
        create_test_dir(dstpth, clean=clean, verbose=True)

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


def _clean_dir(test_dir, verbose=False):
    """
    Delete a test directory

    Parameters
    ----------
    test_dir : str
        path of directory to create
    clean : bool
        boolean indicating if an existing directory should be cleaned

    Returns
    -------

    """

    if os.path.isdir(test_dir):
        if verbose:
            print(f"removing test directory...'{test_dir}'")

        # remove the tree
        shutil.rmtree(test_dir, ignore_errors=True)
