import os
import shutil
import tempfile
import time
from warnings import warn

from .createpackages import create_packages

thisfilepath = os.path.dirname(os.path.abspath(__file__))
flopypth = os.path.join(thisfilepath, "..", "..")
flopypth = os.path.abspath(flopypth)
protected_dfns = ["flopy.dfn"]

default_owner = "MODFLOW-USGS"
default_repo = "modflow6"


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
            print(f"  removing...{fn}")
            os.remove(fpth)
        except:
            print(f"could not remove...{fn}")
            if not allow_failure:
                return False
    return True


def list_files(pth, exts=["py"]):
    print(f"\nLIST OF FILES IN {pth}")
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
            print(f"    {idx:5d} - {fn}")


def download_dfn(owner, repo, ref, new_dfn_pth):
    try:
        from modflow_devtools.download import download_and_unzip
    except ImportError:
        raise ImportError(
            "The modflow-devtools package must be installed in order to "
            "generate the MODFLOW 6 classes. This can be with:\n"
            "     pip install modflow-devtools"
        )

    mf6url = f"https://github.com/{owner}/{repo}/archive/{ref}.zip"
    print(f"  Downloading MODFLOW 6 repository from {mf6url}")
    with tempfile.TemporaryDirectory() as tmpdirname:
        dl_path = download_and_unzip(mf6url, tmpdirname, verbose=True)
        dl_contents = list(dl_path.glob("modflow6-*"))
        proj_path = next(iter(dl_contents), None)
        if not proj_path:
            raise ValueError(
                f"Could not find modflow6 project dir in {dl_path}: found {dl_contents}"
            )
        downloaded_dfn_pth = os.path.join(
            proj_path, "doc", "mf6io", "mf6ivar", "dfn"
        )
        shutil.copytree(downloaded_dfn_pth, new_dfn_pth)


def backup_existing_dfns(flopy_dfn_path):
    parent_folder = os.path.dirname(flopy_dfn_path)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    backup_folder = os.path.join(parent_folder, "dfn_backup", timestr)
    shutil.copytree(flopy_dfn_path, backup_folder)
    assert os.path.isdir(
        backup_folder
    ), f"dfn backup files not found: {backup_folder}"


def replace_dfn_files(new_dfn_pth, flopy_dfn_path):
    # remove the old files, unless the file is protected
    filenames = os.listdir(flopy_dfn_path)
    delete_files(filenames, flopy_dfn_path, exclude=protected_dfns)

    # copy the new ones into the folder
    filenames = os.listdir(new_dfn_pth)
    for filename in filenames:
        filename_w_path = os.path.join(new_dfn_pth, filename)
        print(f"  copying..{filename}")
        shutil.copy(filename_w_path, flopy_dfn_path)


def delete_mf6_classes():
    pth = os.path.join(flopypth, "mf6", "modflow")
    files = [
        entry
        for entry in os.listdir(pth)
        if os.path.isfile(os.path.join(pth, entry))
    ]
    delete_files(files, pth)


def generate_classes(
    owner=default_owner,
    repo=default_repo,
    branch=None,
    ref="master",
    dfnpath=None,
    backup=True,
):
    """
    Generate the MODFLOW 6 flopy classes using definition files from the
    MODFLOW 6 GitHub repository or a set of definition files in a folder
    provided by the user.

    Parameters
    ----------
    owner : str, default "MODFLOW-USGS"
        Owner of the MODFLOW 6 repository to use to update the definition
        files and generate the MODFLOW 6 classes.
    repo : str, default "modflow6"
        Name of the MODFLOW 6 repository to use to update the definition.
    branch : str, optional
        Branch name of the MODFLOW 6 repository to use to update the
        definition files and generate the MODFLOW 6 classes.

        .. deprecated:: 3.5.0
            Use ref instead.
    ref : str, default "master"
        Branch name, tag, or commit hash to use to update the definition.
    dfnpath : str
        Path to a definition file folder that will be used to generate the
        MODFLOW 6 classes.  Default is none, which means that the branch
        will be used instead.  dfnpath will take precedence over branch
        if dfnpath is specified.
    backup : bool, default True
        Keep a backup of the definition files in dfn_backup with a date and
        timestamp from when the definition files were replaced.

    """

    # print header
    print(2 * "\n")
    print(72 * "*")
    print("Updating the flopy MODFLOW 6 classes")
    flopy_dfn_path = os.path.join(flopypth, "mf6", "data", "dfn")

    # download the dfn files and put them in flopy.mf6.data or update using
    # user provided dfnpath
    if dfnpath is None:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        new_dfn_pth = os.path.join(flopypth, "mf6", "data", f"dfn_{timestr}")

        # branch deprecated 3.5.0
        if not ref and not branch:
            raise ValueError("branch or ref must be provided")
        if branch:
            warn("branch is deprecated, use ref instead", DeprecationWarning)
            ref = branch

        print(f"  Updating the MODFLOW 6 classes using {owner}/{repo}/{ref}")

        download_dfn(owner, repo, ref, new_dfn_pth)
    else:
        print(f"  Updating the MODFLOW 6 classes using {dfnpath}")
        assert os.path.isdir(dfnpath)
        new_dfn_pth = dfnpath

    if backup:
        print(f"  Backup existing definition files in: {flopy_dfn_path}")
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


def cli_main():
    """Command-line interface for generate_classes()."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=generate_classes.__doc__.split("\n\n")[0],
    )

    parser.add_argument(
        "--owner",
        type=str,
        default=default_owner,
        help=f"GitHub repository owner; default is '{default_owner}'.",
    )
    parser.add_argument(
        "--repo",
        default=default_repo,
        help=f"Name of GitHub repository; default is '{default_repo}'.",
    )
    parser.add_argument(
        "--ref",
        default="master",
        help="Branch name, tag, or commit hash to use to update the "
        "definition; default is 'master'.",
    )
    parser.add_argument(
        "--dfnpath",
        help="Path to a definition file folder that will be used to generate "
        "the MODFLOW 6 classes.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Set to disable backup. "
        "Default behavior is to keep a backup of the definition files in "
        "dfn_backup with a date and timestamp from when the definition "
        "files were replaced.",
    )

    args = vars(parser.parse_args())
    # Handle flipped logic
    args["backup"] = not args.pop("no_backup")
    try:
        generate_classes(**args)
    except (EOFError, KeyboardInterrupt):
        sys.exit(f" cancelling '{sys.argv[0]}'")


if __name__ == "__main__":
    """Run command-line with: python -m flopy.mf6.utils.generate_classes"""
    cli_main()
