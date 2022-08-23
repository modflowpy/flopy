from itertools import groupby
from os import linesep
from pathlib import Path
from tempfile import gettempdir

import pytest
from filelock import FileLock

__mf6_examples = "mf6_examples"
__mf6_examples_path = Path(gettempdir()) / __mf6_examples
__mf6_examples_lock = FileLock(Path(gettempdir()) / f"{__mf6_examples}.lock")


def get_mf6_examples_path() -> Path:
    pytest.importorskip("pymake")
    import pymake

    # use file lock so mf6 distribution is downloaded once,
    # even when tests are run in parallel with pytest-xdist
    __mf6_examples_lock.acquire()
    try:
        if not __mf6_examples_path.is_dir():
            __mf6_examples_path.mkdir(exist_ok=True)
            pymake.download_and_unzip(
                url="https://github.com/MODFLOW-USGS/modflow6-examples/releases/download/current/modflow6-examples.zip",
                pth=str(__mf6_examples_path),
                verify=True,
            )
        return __mf6_examples_path
    finally:
        __mf6_examples_lock.release()


def is_nested(namfile) -> bool:
    p = Path(namfile)
    if not p.is_file() or not p.name.endswith(".nam"):
        raise ValueError(f"Expected a namfile path, got {p}")

    return p.parent.parent.name != __mf6_examples


def pytest_generate_tests(metafunc):
    # examples to skip:
    #   - ex-gwtgwt-mt3dms-p10: https://github.com/MODFLOW-USGS/modflow6/pull/1008
    exclude = ["ex-gwt-gwtgwt-mt3dms-p10"]
    namfiles = [
        str(p)
        for p in get_mf6_examples_path().rglob("mfsim.nam")
        if not any(e in str(p) for e in exclude)
    ]

    # parametrization by model
    #   - single namfile per test case
    #   - no coupling (only first model in each simulation subdir is used)
    key = "mf6_example_namfile"
    if key in metafunc.fixturenames:
        metafunc.parametrize(key, sorted(namfiles))

    # parametrization by simulation
    #   - each test case gets an ordered list of 1+ namfiles
    #   - models can be coupled (run in order provided, sharing workspace)
    key = "mf6_example_namfiles"
    if key in metafunc.fixturenames:
        simulations = []

        def simulation_name_from_model_path(p):
            p = Path(p)
            return p.parent.parent.name if is_nested(p) else p.parent.name

        for model_name, model_namfiles in groupby(
            namfiles, key=simulation_name_from_model_path
        ):
            models = sorted(
                list(model_namfiles)
            )  # sort in alphabetical order (gwf < gwt)
            simulations.append(models)
            print(
                f"Simulation {model_name} has {len(models)} model(s):\n"
                f"{linesep.join(model_namfiles)}"
            )

        def simulation_name_from_model_namfiles(mnams):
            namfile = next(iter(mnams), None)
            if namfile is None:
                pytest.skip("No namfiles (expected ordered collection)")
            namfile = Path(namfile)
            return (
                namfile.parent.parent if is_nested(namfile) else namfile.parent
            ).name

        metafunc.parametrize(
            key, simulations, ids=simulation_name_from_model_namfiles
        )
