# Remove the temp directory if it exists
import os
import shutil

from ci_framework import get_parent_path


def cleanup_autotests():
    parent_path = get_parent_path()
    if parent_path is None:
        print("can not clean autotests")
    else:
        tempdir = os.path.join(parent_path, "temp")
        if os.path.isdir(tempdir):
            shutil.rmtree(tempdir)
    return


def cleanup_stray_files():
    trashExtensions = (".chk", ".dat")
    parent_path = get_parent_path()

    files = [f for f in os.listdir(parent_path)]
    for file in files:
        if os.path.isfile(file):
            extension = os.path.splitext(file)[1]
            if extension in trashExtensions:
                os.remove(file)


if __name__ == "__main__":
    cleanup_autotests()
    cleanup_stray_files()
