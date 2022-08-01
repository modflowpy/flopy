from functools import reduce
from os import linesep
from pathlib import Path
from subprocess import PIPE, Popen

import pytest

from autotest.conftest import get_project_root_path


def get_scripts(exclude=None):
    prjroot = get_project_root_path(__file__)

    # sort to appease pytest-xdist: all workers must collect identically ordered sets of tests
    return sorted(reduce(lambda a, b: a + b,
                         [[str(p) for p in d.rglob('*.py') if (p.name not in exclude if exclude else True)] for d in [
                             prjroot / "examples" / "scripts",
                             prjroot / "examples" / "Tutorials"]],
                         []))


@pytest.mark.slow
@pytest.mark.example
@pytest.mark.parametrize("script", get_scripts())
def test_scripts_and_tutorials(script, benchmark):
    proc = Popen(("python", Path(script).name), stdout=PIPE, stderr=PIPE, cwd=Path(script).parent)
    stdout, stderr = benchmark(lambda: proc.communicate())
    if stdout: print(stdout.decode("utf-8"))

    allowed_patterns = [
        "findfont",
        "warning",
        "loose"
    ]

    assert (not stderr or
            # trap warnings & non-fatal errors
            all((not line or any(p in line.lower() for p in allowed_patterns)) for line in stderr.decode("utf-8").split(linesep)))
