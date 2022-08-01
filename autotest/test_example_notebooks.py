import os

import pytest

from autotest.conftest import get_project_root_path


def get_notebooks(exclude=None):
    prjroot = get_project_root_path(__file__)
    nbpaths = []
    nbpaths += [str(p) for p in (prjroot / "examples" / "FAQ").glob("*.ipynb")]
    nbpaths += [str(p) for p in (prjroot / "examples" / "Notebooks").glob("*.ipynb")]
    nbpaths += [str(p) for p in (prjroot / "examples" / "groundwater_paper" / "Notebooks").glob("*.ipynb")]
    return sorted([p for p in nbpaths if not exclude or not any(e in p for e in exclude)])


@pytest.mark.slow
@pytest.mark.example
@pytest.mark.parametrize("notebook", get_notebooks(exclude=["mf6_lgr"]))  # TODO: figure out why this one fails
def test_notebooks(notebook):
    arg = ("jupytext", "--from ipynb", "--execute", notebook)
    print(" ".join(arg))
    assert os.system(" ".join(arg)) == 0, f"could not run {notebook}"
