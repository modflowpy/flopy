# Remove the temp directory and then create a fresh one
import os
import shutil

nbdir = os.path.join("..", "examples", "Notebooks")
faqdir = os.path.join("..", "examples", "FAQ")
gwdir = os.path.join("..", "examples", "groundwater_paper", "Notebooks")

# -- make working directories
ddir = os.path.join(nbdir, "data")
if os.path.isdir(ddir):
    shutil.rmtree(ddir)
os.mkdir(ddir)


def get_Notebooks(dpth):
    return [f for f in os.listdir(dpth) if f.endswith(".ipynb")]


def run_notebook(dpth, fn):
    # run autotest on each notebook
    src = os.path.join(dpth, fn)
    arg = (
        "jupytext",
        "--from ipynb",
        "--execute",
        src,
    )
    print(" ".join(arg))
    ival = os.system(" ".join(arg))
    assert ival == 0, "could not run {}".format(fn)


def test_notebooks():

    for dpth in [faqdir, nbdir, gwdir]:
        # get list of notebooks to run
        files = get_Notebooks(dpth)

        # run each notebook
        for fn in files:
            yield run_notebook, dpth, fn


if __name__ == "__main__":

    for dpth in [gwdir]:  # faqdir, nbdir, gwpaper]:
        # get list of notebooks to run
        files = get_Notebooks(dpth)

        # run each notebook
        for fn in files:
            run_notebook(dpth, fn)
