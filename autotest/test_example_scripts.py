import re
from pprint import pprint

import pytest
from flaky import flaky
from modflow_devtools.misc import is_in_ci, run_cmd, run_py_script

from autotest.conftest import get_project_root_path


def get_scripts(pattern=None, exclude=None):
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
    "script",
    get_scripts(pattern="tutorial", exclude=["mf6_lgr"])
    + get_scripts(pattern="example"),
)
def test_scripts(script):
    stdout, stderr, returncode = run_py_script(script, verbose=True)

    # allow notebooks to fail for lack of optional dependencies in local runs,
    # expect all dependencies to be present and notebooks to pass in ci tests.
    if returncode != 0 and not is_in_ci():
        if "Missing optional dependency" in stderr:
            pkg = re.findall("Missing optional dependency '(.*)'", stderr)[0]
            pytest.skip(f"notebook requires optional dependency {pkg!r}")
        elif "No module named " in stderr:
            pkg = re.findall("No module named '(.*)'", stderr)[0]
            pytest.skip(f"notebook requires package {pkg!r}")

    assert returncode == 0, f"could not run {script}"
    pprint(stdout)
    pprint(stderr)
