import re
from functools import reduce
from os import linesep

import pytest
from autotest.conftest import get_project_root_path
from modflow_devtools.misc import run_py_script


def get_example_scripts(exclude=None):
    prjroot = get_project_root_path()

    # sort to appease pytest-xdist: all workers must collect identically ordered sets of tests
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
                    prjroot / "examples" / "Tutorials",
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

    allowed_patterns = ["findfont", "warning", "loose", "match_original"]

    assert (
        not stderr
        or
        # trap warnings & non-fatal errors
        all(
            (not line or any(p in line.lower() for p in allowed_patterns))
            for line in stderr.split(linesep)
        )
    )
