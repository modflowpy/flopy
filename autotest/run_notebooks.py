# Remove the temp directory and then create a fresh one
import os

import pytest

nbdir = os.path.join("..", "examples", "Notebooks")
dpth = nbdir
notebook_files = [
    os.path.join(dpth, f) for f in os.listdir(dpth) if f.endswith(".ipynb")
]

faqdir = os.path.join("..", "examples", "FAQ")
dpth = faqdir
notebook_files += [
    os.path.join(dpth, f) for f in os.listdir(dpth) if f.endswith(".ipynb")
]

gwdir = os.path.join("..", "examples", "groundwater_paper", "Notebooks")
dpth = gwdir
notebook_files += [
    os.path.join(dpth, f) for f in os.listdir(dpth) if f.endswith(".ipynb")
]


def run_notebook(src):
    # run autotest on each notebook
    arg = (
        "jupytext",
        "--from ipynb",
        "--execute",
        src,
    )
    print(" ".join(arg))
    ival = os.system(" ".join(arg))
    assert ival == 0, f"could not run {src}"


@pytest.mark.parametrize(
    "fpth",
    notebook_files,
)
def test_notebooks(fpth):
    run_notebook(fpth)


if __name__ == "__main__":
    # run each notebook
    for fpth in notebook_files:
        run_notebook(fpth)
