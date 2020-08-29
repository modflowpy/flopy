# Remove the temp directory and then create a fresh one
from __future__ import print_function
import os
import sys
import shutil
from subprocess import Popen, PIPE

exclude = ['flopy_swi2_ex2.py', 'flopy_swi2_ex5.py']
for arg in sys.argv:
    if arg.lower() == '--all':
        exclude = []

sdir = os.path.join('..', 'examples', 'scripts')
tdir = os.path.join("..", "examples", "Tutorials")

# make working directories
tempdir = os.path.join('.', 'temp')
if os.path.isdir(tempdir):
    shutil.rmtree(tempdir)
os.mkdir(tempdir)

testdirs = (os.path.join('.', 'temp', 'scripts'),
            os.path.join('.', 'temp', 'tutorials'),
            )
for testdir in testdirs:
    if os.path.isdir(testdir):
        shutil.rmtree(testdir)
    os.mkdir(testdir)

    # add testdir to python path
    sys.path.append(testdir)


def copy_scripts(src_dir, dst_dir):
    files = [f for f in sorted(os.listdir(src_dir)) if f.endswith('.py')]

    # exclude unwanted files
    for e in exclude:
        if e in files:
            files.remove(e)

    # copy files
    for file_name in files:
        src = os.path.join(src_dir, file_name)
        dst = os.path.join(dst_dir, file_name)

        # copy script
        print('copying {} from {} to {}'.format(file_name, src_dir, testdir))
        shutil.copyfile(src, dst)

    return files


def import_from(mod, name):
    mod = __import__(mod)
    main = getattr(mod, name)
    return main


def run_scripts(fn, testdir):
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


def run_tutorial_scripts(fn, testdir):
    args = ("python", fn)
    print("running...'{}'".format(" ".join(args)))
    proc = Popen(args, stdout=PIPE, stderr=PIPE, cwd=testdir)
    stdout, stderr = proc.communicate()
    if stdout:
        print(stdout.decode("utf-8"))
    if stderr:
        print("Errors:\n{}".format(stderr.decode("utf-8")))

    return


def test_scripts():
    # get list of scripts to run
    files = copy_scripts(sdir, testdirs[0])

    for fn in files:
        yield run_scripts, fn, testdirs[0]


def test_tutorial_scripts():
    # get list of scripts to run
    files = copy_scripts(tdir, testdirs[1])

    for fn in files:
        yield run_tutorial_scripts, fn, testdirs[1]


if __name__ == '__main__':
    # # get list of scripts to run
    # files = copy_scripts(sdir, testdirs[0])
    # for fn in files:
    #     run_scripts(fn, testdirs[0])

    # get list of tutorial scripts to run
    files = copy_scripts(tdir, testdirs[1])
    for fn in files:
        run_tutorial_scripts(fn, testdirs[1])
