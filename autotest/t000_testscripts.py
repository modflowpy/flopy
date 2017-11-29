# Remove the temp directory and then create a fresh one
from __future__ import print_function
import os
import sys
import shutil

exclude = ['flopy_swi2_ex2.py', 'flopy_swi2_ex5.py']
for arg in sys.argv:
    if arg.lower() == '--all':
        exclude = []

sdir = os.path.join('..', 'examples', 'scripts')

# -- make working directories
testdir = os.path.join('.', 'temp', 'scripts')
if os.path.isdir(testdir):
    shutil.rmtree(testdir)
os.mkdir(testdir)


def get_scripts():
    files = [f for f in os.listdir(sdir) if f.endswith('.py')]
    for e in exclude:
        if e in files:
            files.remove(e)
    return files

def run_scripts(fn):
    pth = os.path.join(sdir, fn)
    opth = os.path.join(testdir, fn)

    # copy script
    print('copying {} from {} to {}'.format(fn, sdir, testdir))
    shutil.copyfile(pth, opth)

    # change to the correct directory
    odir = os.getcwd()
    print('change directory to {}'.format(testdir))
    os.chdir(testdir)

    cmd = 'python {}'.format(fn)
    ival = os.system(cmd)

    print('change directory to {}'.format(testdir))
    os.chdir(odir)

    # make sure script ran successfully
    assert ival == 0, 'could not run {}'.format(fn)


def test_notebooks():
    files = get_scripts()

    for fn in files:
        yield run_scripts, fn


if __name__ == '__main__':
    files = get_scripts()
    print(files)
    for fn in files:
        run_scripts(fn)
