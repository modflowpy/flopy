# Build the executables that are used in the flopy autotests
import os
import sys
import shutil
import subprocess

try:
    import pymake
except:
    print("pymake is not installed...will not download executables")
    pymake = None

try:
    import flopy
except:
    print("flopy is not installed...will not update flopy")
    flopy = None

os.environ["CI"] = "1"

# path where downloaded executables will be extracted
exe_pth = "exe_download"
# make the directory if it does not exist
if not os.path.isdir(exe_pth):
    os.makedirs(exe_pth)

# determine if running on Travis
is_CI = "CI" in os.environ

bindir = "."
dotlocal = False
if is_CI:
    dotlocal = True

if not dotlocal:
    for idx, arg in enumerate(sys.argv):
        if "--ci" in arg.lower():
            dotlocal = True
            break
if dotlocal:
    bindir = os.path.join(os.path.expanduser("~"), ".local", "bin")
    bindir = os.path.abspath(bindir)
    print("bindir: {}".format(bindir))
    if not os.path.isdir(bindir):
        os.makedirs(bindir)

# write where the executables will be downloaded
print('modflow executables will be downloaded to:\n\n    "{}"'.format(bindir))


def get_branch():
    branch = None

    # determine if branch defined on command line
    for argv in sys.argv:
        if "master" in argv:
            branch = "master"
        elif "develop" in argv.lower():
            branch = "develop"

    if branch is None:
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
                    branch = line.replace("On branch ", "").rstrip()

        except:
            msg = "Could not determine current branch. Is git installed?"
            raise ValueError(msg)

    return branch


def cleanup():
    if os.path.isdir(exe_pth):
        shutil.rmtree(exe_pth)
    return


def move_exe():
    files = os.listdir(exe_pth)
    for file in files:
        if file.startswith("__"):
            continue
        src = os.path.join(exe_pth, file)
        dst = os.path.join(bindir, file)
        print("moving {} -> {}".format(src, dst))
        shutil.move(src, dst)
    return


def list_exes():
    cmd = "ls -l {}".format(bindir)
    os.system(cmd)
    return


def test_download_and_unzip():
    error_msg = "pymake not installed - cannot download executables"
    assert pymake is not None, error_msg

    pymake.getmfexes(exe_pth, verbose=True)

    move_exe()

    return


def test_download_nightly_build():
    error_msg = "pymake not installed - cannot download executables"
    assert pymake is not None, error_msg

    # Replace MODFLOW 6 executables with the latest versions
    if get_branch() != "master":
        platform = sys.platform.lower()
        if "linux" in platform:
            zip_file = "linux.zip"
        elif "darwin" in platform:
            zip_file = "mac.zip"
        elif "win32" in platform:
            zip_file = "win64.zip"
        url = pymake.get_repo_assets("MODFLOW-USGS/modflow6-nightly-build")[
            zip_file
        ]
        pymake.download_and_unzip(url, exe_pth, verbose=True)

        move_exe()

    return


def test_update_flopy():
    error_msg = "flopy not installed - cannot update flopy"
    assert flopy is not None, error_msg

    if get_branch() != "master":
        flopy.mf6.utils.generate_classes(branch="develop", backup=False)


def test_cleanup():
    cleanup()


def test_list_download():
    list_exes()


if __name__ == "__main__":
    test_download_and_unzip()
    test_download_nightly_build()
    test_update_flopy()
    cleanup()
    list_exes()
