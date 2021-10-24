"""
Script to be used to download any required data prior to autotests
"""
import os
import shutil
from t503_test import download_mf6_examples


def test_build_dirs():
    dirPath = os.path.join(".", "temp")
    if os.path.isdir(dirPath):
        shutil.rmtree(dirPath)
    os.makedirs(dirPath)

    tests = [
        fname[:4]
        for fname in os.listdir(".")
        if fname.startswith("t") and fname.endswith(".py")
    ]
    tests += ["scripts"]
    for test in tests:
        testDir = os.path.join(dirPath, test)
        os.makedirs(testDir)


def test_mf6_download():
    """Download MODFLOW 6 examples in latest release"""
    download_mf6_examples(delete_existing=True)


if __name__ == "__main__":
    test_build_dirs()
    test_mf6_download()
