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

# make working directories
tempdir = os.path.join('.', 'temp')
if os.path.isdir(tempdir):
    shutil.rmtree(tempdir)
os.mkdir(tempdir)

testdir = os.path.join('.', 'temp', 'scripts')
if os.path.isdir(testdir):
    shutil.rmtree(testdir)
os.mkdir(testdir)

# add testdir to python path
sys.path.append(testdir)


def copy_scripts():
    files = [f for f in os.listdir(sdir) if f.endswith('.py')]

    # exclude unwanted files
    for e in exclude:
        if e in files:
            files.remove(e)

    # copy files
    for fn in files:
        pth = os.path.join(sdir, fn)
        opth = os.path.join(testdir, fn)

        # copy script
        print('copying {} from {} to {}'.format(fn, sdir, testdir))
        shutil.copyfile(pth, opth)

    return files


def import_from(mod, name):
    mod = __import__(mod)
    main = getattr(mod, name)
    return main


def run_scripts(fn):
    # import run function from scripts
    s = os.path.splitext(fn)[0]
    run = import_from(s, 'run')

    # change to working directory
    opth = os.getcwd()
    print('changing to working directory "{}"'.format(testdir))
    os.chdir(testdir)

    # run the script
    ival = run()

    # change back to starting directory
    print('changing back to starting directory "{}"'.format(opth))
    os.chdir(opth)

    # make sure script ran successfully
    assert ival == 0, 'could not run {}'.format(fn)


def test_notebooks():

    # get list of scripts to run
    files = copy_scripts()

    for fn in files:
        yield run_scripts, fn


if __name__ == '__main__':

    # get list of scripts to run
    files = copy_scripts()

    for fn in files:
        run_scripts(fn)
