# Build the executables that are used in the flopy autotests
import os
import sys
import shutil

try:
    import pymake
except:
    print('pymake is not installed...will not download executables')
    pymake = None

os.environ["CI"] = "1"

# path where downloaded executables will be extracted
exe_pth = 'exe_download'
# make the directory if it does not exist
if not os.path.isdir(exe_pth):
    os.makedirs(exe_pth)

# determine if running on Travis
is_CI = 'CI' in os.environ

bindir = '.'
dotlocal = False
if is_CI:
    dotlocal = True

if not dotlocal:
    for idx, arg in enumerate(sys.argv):
        if '--ci' in arg.lower():
            dotlocal = True
            break
if dotlocal:
    bindir = os.path.join(os.path.expanduser('~'), '.local', 'bin')
    bindir = os.path.abspath(bindir)
    print("bindir: {}".format(bindir))
    if not os.path.isdir(bindir):
        os.makedirs(bindir)

# write where the executables will be downloaded
print('modflow executables will be downloaded to:\n\n    "{}"'.format(bindir))


def cleanup():
    if os.path.isdir(exe_pth):
        shutil.rmtree(exe_pth)
    return


def list_exes():
    cmd = 'ls -l {}'.format(bindir)
    os.system(cmd)
    return


def test_download_and_unzip():
    if pymake is None:
        print("cannot download executables")
    else:
        pymake.getmfexes(exe_pth, verbose=True)

        # move the exes from exe_pth to bindir
        files = os.listdir(exe_pth)
        for file in files:
            if file.startswith('__'):
                continue
            src = os.path.join(exe_pth, file)
            dst = os.path.join(bindir, file)
            print('moving {} -> {}'.format(src, dst))
            shutil.move(src, dst)
    return


def test_cleanup():
    cleanup()


def test_list_download():
    list_exes()


if __name__ == '__main__':
    test_download_and_unzip()
    cleanup()
    list_exes()
