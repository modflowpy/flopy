# Build the executables that are used in the flopy autotests
import os
import sys
import shutil

try:
    import pymake
except:
    print('pymake is not installed...will not build executables')
    pymake = None

os.environ["TRAVIS"] = "1"

# path where downloaded executables will be extracted
exe_pth = 'exe_download'
# make the directory if it does not exist
if not os.path.isdir(exe_pth):
    os.makedirs(exe_pth)

# determine if running on Travis
is_travis = 'TRAVIS' in os.environ

bindir = '.'
dotlocal = False
if is_travis:
    dotlocal = True

if not dotlocal:
    for idx, arg in enumerate(sys.argv):
        if '--travis' in arg.lower():
            dotlocal = True
            break
if dotlocal:
    bindir = os.path.join(os.path.expanduser('~'), '.local', 'bin')
    bindir = os.path.abspath(bindir)
    if not os.path.isdir(bindir):
        os.makedirs(bindir)

# write where the executables will be downloaded
print('modflow executables will be downloaded to:\n\n    "{}"'.format(bindir))


def cleanup():
    if os.path.isdir(exe_pth):
        shutil.rmtree(exe_pth)

    return


def test_download_and_unzip():
    pymake.getmfexes(exe_pth)


def test_cleanup():
    cleanup()


def main():
    pymake.getmfexes(exe_pth)

    # clean up the download directory
    cleanup()


if __name__ == '__main__':
    main()
