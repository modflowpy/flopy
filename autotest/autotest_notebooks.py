# Remove the temp directory and then create a fresh one
import os
import shutil

nbdir = os.path.join('..', 'examples', 'Notebooks')

# -- make working directories
ddir = os.path.join(nbdir, 'data')
if os.path.isdir(ddir):
    shutil.rmtree(ddir)
os.mkdir(ddir)

tempdir = os.path.join('.', 'temp')
if os.path.isdir(tempdir):
    shutil.rmtree(tempdir)
os.mkdir(tempdir)

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


def test_notebooks():
    files = get_Notebooks()

    for fn in files:
        yield run_notebook, fn


if __name__ == '__main__':
    files = get_Notebooks()
    for fn in files:
        run_notebook(fn)
