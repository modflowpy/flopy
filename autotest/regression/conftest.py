from itertools import groupby
from os import linesep
from pathlib import Path
from tempfile import gettempdir

import pytest
from filelock import FileLock

__mf6_examples = "mf6_examples"
__mf6_examples_path = Path(gettempdir()) / __mf6_examples
__mf6_examples_lock = FileLock(Path(gettempdir()) / f"{__mf6_examples}.lock")


def is_nested(namfile) -> bool:
    p = Path(namfile)
    if not p.is_file() or not p.name.endswith('.nam'):
        raise ValueError(f"Expected a namfile path, got {p}")

    return p.parent.parent.name != __mf6_examples


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


@pytest.fixture(scope="session")
def temp_mf6_examples_path(tmpdir_factory):
    pytest.importorskip("pymake")
    import pymake

    temp = Path(tmpdir_factory.mktemp(__mf6_examples))
    pymake.download_and_unzip(
        url="https://github.com/MODFLOW-USGS/modflow6-examples/releases/download/current/modflow6-examples.zip",
        pth=str(temp),
        verify=True,
    )
    return temp


def pytest_generate_tests(metafunc):
    def get_namfiles():
        return get_mf6_examples_path().rglob("mfsim.nam")

    key = "mf6_example_namfile"
    if key in metafunc.fixturenames:
        # model parametrization (single namfile, no coupling)
        namfiles = [str(p) for p in get_namfiles()]
        metafunc.parametrize(key, sorted(namfiles))

    key = "mf6_example_namfiles"
    if key in metafunc.fixturenames:
        # simulation parametrization (1+ models in series)
        # ordered list of namfiles representing simulation
        namfiles = sorted([str(p) for p in get_namfiles()])

        def simulation_name_from_model_path(p):
            p = Path(p)
            return p.parent.parent.name if is_nested(p) else p.parent.name

        def simulation_name_from_model_namfiles(mnams):
            namfile = next(iter(mnams), None)
            if namfile is None: pytest.skip("No namfiles (expected ordered collection)")
            namfile = Path(namfile)
            return (namfile.parent.parent if is_nested(namfile) else namfile.parent).name

        simulations = []
        for model_name, model_namfiles in groupby(namfiles, key=simulation_name_from_model_path):
            models = sorted(list(model_namfiles))  # sort in alphabetical order (gwf < gwt)
            simulations.append(models)
            print(f"Simulation {model_name} has {len(models)} model(s):\n"
                  f"{linesep.join(model_namfiles)}")

        metafunc.parametrize(key, simulations, ids=simulation_name_from_model_namfiles)
