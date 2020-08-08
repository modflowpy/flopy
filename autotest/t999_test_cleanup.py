# Remove the temp directory if it exists
import os
import shutil


def test_cleanup():
    tempdir = os.path.join('.', 'temp')
    if os.path.isdir(tempdir):
        shutil.rmtree(tempdir)
    return


if __name__ == '__main__':
    test_cleanup()
