import sys
import os
import tempfile
import shutil
import time
from .createpackages import create_packages


thisfilepath = os.path.dirname(os.path.abspath(__file__))
flopypth = os.path.join(thisfilepath, "..", "..")
flopypth = os.path.abspath(flopypth)
protected_dfns = ["flopy.dfn"]


def delete_files(files, pth, allow_failure=False, exclude=None):
    if exclude is None:
        exclude = []
    else:
        if not isinstance(exclude, list):
            exclude = [exclude]

    for fn in files:
        if fn in exclude:
            continue
        fpth = os.path.join(pth, fn)
        try:
            print("  removing...{}".format(fn))
            os.remove(fpth)
        except:
            print("could not remove...{}".format(fn))
            if not allow_failure:
                return False
    return True


def list_files(pth, exts=["py"]):
    print("\nLIST OF FILES IN {}".format(pth))
    files = [
        entry
        for entry in os.listdir(pth)
        if os.path.isfile(os.path.join(pth, entry))
    ]
    idx = 0
    for fn in files:
        ext = os.path.splitext(fn)[1][1:].lower()
        if ext in exts:
            idx += 1
            print("    {:5d} - {}".format(idx, fn))
    return


def download_dfn(branch, new_dfn_pth):
    pymake = None
    try:
        import pymake
    except:
        pass
    if pymake is None:
        msg = (
            "Error.  The pymake package must be installed in order to "
            "generate the MODFLOW 6 classes.  pymake can be installed using "
            "pip install pymake.  Stopping."
        )
        print(msg)
        return

    mf6url = "https://github.com/MODFLOW-USGS/modflow6/archive/{}.zip"
    mf6url = mf6url.format(branch)
    print("  Downloading MODFLOW 6 repository from {}".format(mf6url))
    with tempfile.TemporaryDirectory() as tmpdirname:
        pymake.download_and_unzip(mf6url, tmpdirname)
        downloaded_dfn_pth = os.path.join(
            tmpdirname, "modflow6-{}".format(branch)
        )
        downloaded_dfn_pth = os.path.join(
            downloaded_dfn_pth, "doc", "mf6io", "mf6ivar", "dfn"
        )
        shutil.copytree(downloaded_dfn_pth, new_dfn_pth)
    return


def backup_existing_dfns(flopy_dfn_path):
    parent_folder = os.path.dirname(flopy_dfn_path)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    backup_folder = os.path.join(parent_folder, "dfn_backup", timestr)
    shutil.copytree(flopy_dfn_path, backup_folder)
    assert os.path.isdir(
        backup_folder
    ), "dfn backup files not found: {}".format(backup_folder)
    return


def replace_dfn_files(new_dfn_pth, flopy_dfn_path):

    # remove the old files, unless the file is protected
    filenames = os.listdir(flopy_dfn_path)
    delete_files(filenames, flopy_dfn_path, exclude=protected_dfns)

    # copy the new ones into the folder
    filenames = os.listdir(new_dfn_pth)
    for filename in filenames:
        filename_w_path = os.path.join(new_dfn_pth, filename)
        print("  copying..{}".format(filename))
        shutil.copy(filename_w_path, flopy_dfn_path)


def delete_mf6_classes():
    pth = os.path.join(flopypth, "mf6", "modflow")
    files = [
        entry
        for entry in os.listdir(pth)
        if os.path.isfile(os.path.join(pth, entry))
    ]
    delete_files(files, pth, exclude="mfsimulation.py")
    return


def generate_classes(branch="master", dfnpath=None, backup=True):
    """
    Generate the MODFLOW 6 flopy classes using definition files from the
    MODFLOW 6 GitHub repository or a set of definition files in a folder
    provided by the user.

    Parameters
    ----------
    branch : str
        Branch name of the MODFLOW 6 repository to use to update the
        definition files and generate the MODFLOW 6 classes. Default is master.
    dfnpath : str
        Path to a definition file folder that will be used to generate the
        MODFLOW 6 classes.  Default is none, which means that the branch
        will be used instead.  dfnpath will take precedence over branch
        if dfnpath is specified.
    backup : bool
        Keep a backup of the definition files in dfn_backup with a date and
        time stamp from when the definition files were replaced.

    """

    # print header
    print(2 * "\n")
    print(72 * "*")
    print("Updating the flopy MODFLOW 6 classes")
    flopy_dfn_path = os.path.join(flopypth, "mf6", "data", "dfn")

    # download the dfn files and put them in flopy.mf6.data or update using
    # user provided dfnpath
    if dfnpath is None:
        print(
            "  Updating the MODFLOW 6 classes using the branch: {}".format(
                branch
            )
        )
        timestr = time.strftime("%Y%m%d-%H%M%S")
        new_dfn_pth = os.path.join(flopypth, "mf6", "data", "dfn_" + timestr)
        download_dfn(branch, new_dfn_pth)
    else:
        print("  Updating the MODFLOW 6 classes using {}".format(dfnpath))
        assert os.path.isdir(dfnpath)
        new_dfn_pth = dfnpath

    if backup:
        print(
            "  Backup existing definition files in: {}".format(flopy_dfn_path)
        )
        backup_existing_dfns(flopy_dfn_path)

    print("  Replacing existing definition files with new ones.")
    replace_dfn_files(new_dfn_pth, flopy_dfn_path)
    if dfnpath is None:
        shutil.rmtree(new_dfn_pth)

    print("  Deleting existing mf6 classes.")
    delete_mf6_classes()

    print("  Create mf6 classes using the downloaded definition files.")
    create_packages()
    list_files(os.path.join(flopypth, "mf6", "modflow"))

    return
