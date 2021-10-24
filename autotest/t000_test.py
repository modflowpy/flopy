"""
Create the temp directory used by the autotests if it does not exist
"""
import os


def test_setup():
    tempdir = os.path.join(".", "temp")
    if not os.path.isdir(tempdir):
        os.mkdir(tempdir)
    return


if __name__ == "__main__":
    test_setup()
