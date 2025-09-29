import re
from platform import system
from pprint import pprint

import pytest
from flaky import flaky
from modflow_devtools.misc import is_in_ci, run_cmd

from autotest.conftest import get_project_root_path

EXCLUDE = [
    "mf6_lgr",
    "mfusg_transport_tutorial07c_ChainDecay",
    "mfusg_transport_tutorial08_Lake",
]
# skip pyvista notebooks on windows/mac in CI due to persistent issues
# first with offscreen rendering, then with finding pyvista even after
# using pyvista/setup-headless-display-action
if is_in_ci() and system() != "Linux":
    EXCLUDE.append("vtk_pathlines")


def get_notebooks(pattern=None, exclude=None):
    prjroot = get_project_root_path()
    nbpaths = [
        str(p)
        for p in (prjroot / ".docs" / "Notebooks").glob("*.py")
        if pattern is None or pattern in p.name
    ]

    # sort for pytest-xdist: workers must collect tests in the same order
    return sorted(
        [p for p in nbpaths if not exclude or not any(e in p for e in exclude)]
    )


@flaky(max_runs=3)
@pytest.mark.slow
@pytest.mark.example
@pytest.mark.parametrize(
    "notebook",
    get_notebooks(pattern="tutorial", exclude=EXCLUDE)
    + get_notebooks(pattern="example", exclude=EXCLUDE),
)
def test_notebooks(notebook):
    args = ["jupytext", "--from", "py", "--to", "ipynb", "--execute", notebook]
    stdout, stderr, returncode = run_cmd(*args, verbose=True)

    # allow notebooks to fail for lack of optional dependencies in local runs,
    # expect all dependencies to be present and notebooks to pass in ci tests.
    if returncode != 0 and not is_in_ci():
        if "Missing optional dependency" in stderr:
            pkg = re.findall(r"Missing optional dependency '(.*)'", stderr)[0]
            pytest.skip(f"notebook requires optional dependency {pkg!r}")
        elif "No module named " in stderr:
            pkg = re.findall(r"No module named '(.*)'", stderr)[0]
            pytest.skip(f"notebook requires package {pkg!r}")

    if returncode != 0:
        pprint(stdout)
        pprint(stderr)
        raise ValueError(f"could not run {notebook}")
