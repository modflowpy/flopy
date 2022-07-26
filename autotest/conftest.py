import shutil
import warnings
from pathlib import Path
from typing import List, Dict

import pytest

import flopy


# temporary directory fixtures for each scope

@pytest.fixture(scope="function")
def tmpdir(tmpdir_factory, request) -> Path:
    node = request.node.name.replace("/", "_")
    temp = Path(tmpdir_factory.mktemp(node))
    yield Path(temp)

    keep = request.config.getoption('--keep')
    if keep:
        shutil.copytree(temp, Path(keep) / temp.name)


@pytest.fixture(scope="class")
def class_tmpdir(tmpdir_factory, request) -> Path:
    assert request.cls is not None, "Class-scoped temp dir fixture must be used on class"
    temp = Path(tmpdir_factory.mktemp(request.cls.__name__))
    yield temp

    keep = request.config.getoption('--keep')
    if keep:
        shutil.copytree(temp, Path(keep) / temp.name)


@pytest.fixture(scope="module")
def module_tmpdir(tmpdir_factory, request) -> Path:
    temp = Path(tmpdir_factory.mktemp(request.module.__name__))
    yield temp

    keep = request.config.getoption('--keep')
    if keep:
        shutil.copytree(temp, Path(keep) / temp.name)


@pytest.fixture(scope="session")
def session_tmpdir(tmpdir_factory, request) -> Path:
    temp = Path(tmpdir_factory.mktemp(request.session.name))
    yield temp

    keep = request.config.getoption('--keep')
    if keep:
        shutil.copytree(temp, Path(keep) / temp.name)


# example data paths

@pytest.fixture(scope="session")
def example_data_path(request) -> Path:
    return Path(request.session.path).parent / "examples" / "data"


@pytest.fixture(scope="session")
def usgs_model_reference_path(example_data_path) -> Path:
    return example_data_path / "usgs.model.reference"


MODEL_RELATIVE_PATHS = [
    "mf2005_test",
    "freyberg",
    "freyberg_multilayer_transient",
]


MF6SIMULATION_RELATIVE_PATHS = [
    "test001a_Tharmonic",
    "test003_gwfs_disv",
    "test006_gwf3",
    "test045_lake2tr",
    "test006_2models_mvr",
    "test001e_UZF_3lay",
    "test003_gwftri_disv"
]


@pytest.fixture(scope="session", autouse=True)
def example_model_paths(example_data_path) -> Dict[str, Path]:
    return {name: example_data_path / name for name in MODEL_RELATIVE_PATHS}


@pytest.fixture(scope="session", autouse=True)
def example_mf6simulation_paths(example_data_path) -> Dict[str, Path]:
    return {name: example_data_path / "mf6" / name for name in MF6SIMULATION_RELATIVE_PATHS}


@pytest.fixture(scope="session", params=[MODEL_RELATIVE_PATHS])
def example_model_namfiles(request, example_model_paths) -> List[Path]:
    name = request.param
    path = example_model_paths[name]
    namfiles = [f.resolve() for f in path.glob("*.nam")]

    if not namfiles:
        warnings.warn(f"No name files found for example model path {path}")

    return namfiles


@pytest.fixture(scope="session", params=[MF6SIMULATION_RELATIVE_PATHS])
def example_mf6simulation_namfiles(request, example_mf6simulation_paths):
    name = request.param
    path = example_mf6simulation_paths[name]
    namfiles = [f.resolve() for f in path.glob("*.nam")]

    if not namfiles:
        warnings.warn(f"No name files found for example MF6 simulation path {path}")

    return namfiles


@pytest.fixture(scope="session")
def example_shapefiles(example_data_path) -> List[Path]:
    return [f.resolve() for f in (example_data_path / "prj_test").glob("*")]


# example model factories

@pytest.fixture(scope="session")
def get_example_model(example_model_paths):
    def get_model(name, namfile: str, **kwargs):
        return flopy.modflow.Modflow.load(f=namfile,
                                          model_ws=str(example_model_paths[name]),
                                          **kwargs)

    return get_model


@pytest.fixture(scope="session")
def get_example_mf6simulation(example_mf6simulation_paths):
    def get_simulation(name, **kwargs):
        path = str(example_mf6simulation_paths[name])
        return flopy.mf6.MFSimulation.load(sim_ws=path)

    return get_simulation


# pytest configuration

def pytest_addoption(parser):
    parser.addoption(
        "-K",
        "--keep",
        action="store",
        default=None,
        help="Move the contents of temporary test directories to correspondingly named subdirectories at the KEEP "
             "location after tests complete. This option can be used to exclude test results from automatic cleanup, "
             "e.g. for manual inspection. The provided path is created if it does not already exist. An error is "
             "thrown if any matching files already exist.")

    parser.addoption(
        "-M",
        "--meta",
        action="store",
        metavar="NAME",
        help="Marker indicating a test is only run by other tests (e.g., the test framework testing itself).")


def pytest_configure(config):
    # register an additional marker
    config.addinivalue_line(
        "markers", "meta(name): mark test to run only inside other groups of tests.")


def pytest_runtest_setup(item):
    # apply metatest markers
    metagroups = [mark.args[0] for mark in item.iter_markers(name="meta")]
    metagroup = item.config.getoption("--meta")
    if metagroups and metagroup not in metagroups:
        pytest.skip()
