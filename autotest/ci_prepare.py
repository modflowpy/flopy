"""
Script to be used to download any required data prior to autotests
"""
import os
import subprocess
import sys

from ci_framework import download_mf6_examples

import flopy

# os.environ["CI"] = "1"

# determine if running on Travis
is_CI = "CI" in os.environ

bindir = "."
dotlocal = False
if is_CI:
    dotlocal = True

if not dotlocal:
    for arg in sys.argv:
        if "--ci" in arg.lower():
            dotlocal = True
            break
if dotlocal:
    bindir = os.path.join(os.path.expanduser("~"), ".local", "bin")
    print(f"bindir: {bindir}")
    if not os.path.isdir(bindir):
        os.makedirs(bindir, exist_ok=True)

# write where the executables will be downloaded
print(f'modflow executables will be downloaded to:\n\n    "{bindir}"')

run_type = "std"
for arg in sys.argv:
    if "--other" in arg.lower():
        run_type = "other"
        break


def get_branch():
    branch = None

    # determine if branch defined on command line
    for argv in sys.argv:
        if "master" in argv:
            branch = "master"
        elif "develop" in argv.lower():
            branch = "develop"

    if branch is None:
        if is_CI:
            github_ref = os.getenv("GITHUB_REF")
            if github_ref is not None:
                return os.path.basename(os.path.normpath(github_ref)).lower()
        try:
            # determine current branch
            b = subprocess.Popen(
                ("git", "status"),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            ).communicate()[0]
            if isinstance(b, bytes):
                b = b.decode("utf-8")

            for line in b.splitlines():
                if "On branch" in line:
                    branch = line.replace("On branch ", "").rstrip().lower()

        except:
            msg = "Could not determine current branch. Is git installed?"
            raise ValueError(msg)

    return branch


def test_download_and_unzip():
    flopy.utils.get_modflow_main(bindir)


def test_download_nightly_build():
    # get the current branch
    branch = get_branch()
    print(f"current branch: {branch}")

    # No need to replace MODFLOW 6 executables
    if branch == "master":
        print("No need to update MODFLOW 6 executables")
    # Replace MODFLOW 6 executables with the latest versions
    else:
        print("Updating MODFLOW 6 executables from the nightly-build repo")
        flopy.utils.get_modflow_main(bindir, repo="modflow6-nightly-build")


def test_update_flopy():
    branch = get_branch()
    if branch == "master":
        print("No need to update flopy MODFLOW 6 classes")
    else:
        print("Update flopy MODFLOW 6 classes")
        flopy.mf6.utils.generate_classes(branch="develop", backup=False)


def test_download_mf6_examples(delete_existing=True):
    if run_type == "std":
        downloadDir = download_mf6_examples(delete_existing=delete_existing)
    else:
        downloadDir = None
    return downloadDir


if __name__ == "__main__":
    test_download_and_unzip()
    test_download_nightly_build()
    test_update_flopy()
    test_download_mf6_examples()
