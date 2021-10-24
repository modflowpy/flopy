# Remove the temp directory if it exists
import os
import shutil


def cleanup_autotests():
    tempdir = os.path.join(".", "temp")
    if os.path.isdir(tempdir):
        shutil.rmtree(tempdir)
    return


if __name__ == "__main__":
    cleanup_autotests()
