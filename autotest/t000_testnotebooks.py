# Remove the temp directory and then create a fresh one
import os
import shutil
import subprocess as sp

nbdir = os.path.join('..', 'examples', 'Notebooks')
testdir = os.path.join('.', 'temp', 'Notebooks')
if os.path.isdir(testdir):
    shutil.rmtree(testdir)
os.mkdir(testdir)


def get_Notebooks():
    return [f for f in os.listdir(nbdir) if f.endswith('.ipynb')]


def run_notebook(fn):
    pth = os.path.join(nbdir, fn)
    cmd = 'jupyter ' + 'nbconvert ' + \
          '--ExecutePreprocessor.kernel_name=python ' + \
          '--ExecutePreprocessor.timeout=600 ' + '--to ' + 'notebook ' + \
          '--execute ' + '{} '.format(pth) + \
          '--output-dir ' + '{} '.format(testdir) + \
          '--output ' + '{}'.format(fn)
    ival = os.system(cmd)
    assert ival == 0, 'could not run {}'.format(fn)


def test_notebooks(silent=False):
    files = get_Notebooks()

    for fn in files:
        yield run_notebook, fn


if __name__ == '__main__':
    files = get_Notebooks()
    for fn in files:
        run_notebook(fn)
