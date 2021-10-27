import os
import sys
import shutil
import pymake

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

    Returns
    -------

    """
    if clean:
        _cleanDir(testDir, verbose=verbose)
    if not os.path.isdir(testDir):
        os.makedirs(testDir, exist_ok=True)
        if verbose:
            print(f"creating test directory...'{testDir}'")


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
        shutil.rmtree(testDir)
        if verbose:
            print(f"removing test directory...'{testDir}'")


class flopyTest(object):
    def __init__(
        self,
        clean=False,
        create=False,
        testDirs=None,
        verbose=False,
    ):
        self.clean = clean
        self.verbose = verbose
        self.createDirs = create
        self.testDirs = []
        if testDirs is not None:
            self.addTestDir(testDirs, clean=clean, create=create)

    def addTestDir(self, testDirs, clean=False, create=False):
        if isinstance(testDirs, str):
            testDirs = [testDirs]
        elif isinstance(testDirs, (int, float, bool)):
            raise ValueError(
                f"testDir '{testDirs}' must be a string, "
                "list of strings, or tuple of strings."
            )
        for testDir in testDirs:
            if testDir not in self.testDirs:
                self.testDirs.append(testDir)
                if self.verbose:
                    print(f"adding test directory...{testDir}")
                if create:
                    createTestDir(testDir, clean=clean, verbose=self.verbose)

    def teardown(self):
        if not keep:
            for testDir in self.testDirs:
                _cleanDir(testDir, verbose=self.verbose)


def _get_mf6path():
    parentPath = get_parent_path()
    if parentPath is None:
        parentPath = "."
    dirName = "mf6examples"
    dstpth = os.path.join(parentPath, "temp", dirName)
    return os.path.abspath(dstpth)


def download_mf6_examples(delete_existing=False):
    """
    Download mf6 examples and return location of folder

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
