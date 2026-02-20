from pathlib import Path
from shutil import copytree

import pytest
from modflow_devtools.misc import run_cmd

from flopy.mf6 import MFSimulation
from flopy.modflow import Modflow


def get_examples_path():
    """Get path to examples/data directory."""
    return Path(__file__).parent.parent.parent / "examples" / "data"


def load_mf6_sim(tmpdir, model_key="freyberg", run=True):
    """
    Load and optionally run a MODFLOW 6 simulation from examples/data.

    Parameters
    ----------
    tmpdir : Path
        Temporary directory to copy the model to
    model_key : str
        Model identifier (e.g., "freyberg" for mf6-freyberg)
    run : bool
        Whether to run the simulation before returning

    Returns
    -------
    MFSimulation
        The loaded simulation
    """
    examples_path = get_examples_path()

    # Map model keys to their actual directory names
    model_map = {
        "freyberg": "mf6-freyberg",
        "test003": "mf6/test003_gwfs_disv",
        "test006": "mf6/test006_gwf3",
        "test045": "mf6/test045_lake2tr",
    }

    model_dir = model_map.get(model_key, f"mf6-{model_key}")
    source_path = examples_path / model_dir

    if not source_path.exists():
        raise FileNotFoundError(f"Model directory not found: {source_path}")

    # Copy model files to tmpdir
    copytree(source_path, tmpdir, dirs_exist_ok=True)

    # Load the simulation
    sim = MFSimulation.load(sim_ws=tmpdir)

    # Run if requested
    if run:
        run_cmd("mf6", cwd=tmpdir)

    return sim


def load_mf2005_model(tmpdir, model_key="freyberg", run=False):
    """
    Load and optionally run a MODFLOW-2005 model from examples/data.

    Parameters
    ----------
    tmpdir : Path
        Temporary directory to copy the model to
    model_key : str
        Model identifier (e.g., "freyberg")
    run : bool
        Whether to run the model before returning

    Returns
    -------
    Modflow
        The loaded model
    """
    from modflow_devtools.misc import get_namefile_paths

    examples_path = get_examples_path()

    # Map model keys to their actual directory names
    model_map = {
        "freyberg": "freyberg_multilayer_transient",
        "mf2005_test": "mf2005_test",
    }

    model_dir = model_map.get(model_key, model_key)
    source_path = examples_path / model_dir

    if not source_path.exists():
        raise FileNotFoundError(f"Model directory not found: {source_path}")

    # Copy model files to tmpdir
    copytree(source_path, tmpdir, dirs_exist_ok=True)

    # Find the namefile
    nam_files = get_namefile_paths(tmpdir, namefile="*.nam")
    if not nam_files:
        raise FileNotFoundError(f"No .nam file found in {tmpdir}")

    nam_file = nam_files[0].name

    # Load the model
    model = Modflow.load(nam_file, model_ws=tmpdir, check=False)

    # Run if requested
    if run:
        model.run_model()

    return model


@pytest.fixture(scope="session")
def benchmark_config():
    """Configure pytest-benchmark settings."""
    return {
        "warmup": True,
        "warmup_iterations": 5,
        "min_rounds": 5,
        "disable_gc": True,
    }


@pytest.fixture(scope="session")
def models_path(request) -> list[Path]:
    """
    A directories containing model subdirectories. Use
    the --models-path command line option once or more to specify
    model directories. If at least one --models_path is provided,
    external tests (i.e. those using models from an external repo)
    will run against model input files found in the given location
    on the local filesystem rather than model input files from the
    official model registry. This is useful for testing changes to
    test model input files during MF6 development.
    """
    paths = request.config.getoption("--models-path") or []
    return [Path(p).expanduser().resolve().absolute() for p in paths]


def pytest_addoption(parser):
    parser.addoption(
        "--models-path",
        action="append",
        type=str,
        help="directory containing model subdirectories. set this to run external "
        "tests (i.e. those using models from an external repo) against local model "
        "input files rather than input files from the official model registry.",
    )
    parser.addoption(
        "--namefile-pattern",
        action="store",
        type=str,
        default="mfsim.nam",
        help="namefile pattern to use when indexing models when --models-path is set."
        "does nothing otherwise. default is 'mfsim.nam'.",
    )
