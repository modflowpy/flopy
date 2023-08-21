import re
from functools import reduce
from pprint import pprint

import pytest
from autotest.conftest import get_project_root_path
from modflow_devtools.misc import run_py_script


def get_example_scripts(exclude=None):
    prjroot = get_project_root_path()

    # sort for pytest-xdist: workers must collect tests in the same order
    return sorted(
        reduce(
            lambda a, b: a + b,
            [
                [
                    str(p)
                    for p in d.rglob("*.py")
                    if (p.name not in exclude if exclude else True)
                ]
                for d in [
                    prjroot / "examples" / "scripts",
                ]
            ],
            [],
        )
    )


@pytest.mark.slow
@pytest.mark.example
@pytest.mark.parametrize("script", get_example_scripts())
def test_scripts(script):
    stdout, stderr, returncode = run_py_script(script, verbose=True)
    if returncode != 0:
        if "Missing optional dependency" in stderr:
            pkg = re.findall("Missing optional dependency '(.*)'", stderr)[0]
            pytest.skip(f"script requires optional dependency {pkg!r}")

    assert returncode == 0
    pprint(stdout)
    pprint(stderr)
