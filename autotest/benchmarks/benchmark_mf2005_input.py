from shutil import copytree

import pytest
from modflow_devtools.misc import get_namefile_paths

from flopy.modflow import Modflow

from .conftest import get_examples_path

# Model directories from examples/data
MODEL_DIRS = [
    "freyberg_multilayer_transient",
    "mf2005_test",
]


def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests with model names."""
    if "model_name" in metafunc.fixturenames:
        examples_path = get_examples_path()
        models = []

        # Only include models that exist
        for model_dir in MODEL_DIRS:
            if (examples_path / model_dir).exists():
                models.append(model_dir)

        metafunc.parametrize("model_name", models, ids=models)


def _load_model(ws, model_name):
    """Load MODFLOW-2005 model from examples."""
    examples_path = get_examples_path()
    source_path = examples_path / model_name
    copytree(source_path, ws, dirs_exist_ok=True)

    nam_files = get_namefile_paths(source_path, namefile="*.nam")
    assert nam_files, f"No .nam file found in {source_path}"
    nam_file = nam_files[0].name

    return Modflow.load(nam_file, model_ws=ws, check=False)


@pytest.mark.benchmark
def test_mf2005_load(benchmark, function_tmpdir, model_name):
    benchmark(lambda: _load_model(function_tmpdir, model_name))


@pytest.mark.benchmark
def test_mf2005_write_freyberg(benchmark, function_tmpdir):
    ml = _load_model(function_tmpdir, "freyberg_multilayer_transient")
    benchmark(ml.write_input)


@pytest.mark.benchmark
def test_mf2005_round_trip_freyberg(benchmark, function_tmpdir):
    def round_trip():
        ml = _load_model(function_tmpdir, "freyberg_multilayer_transient")
        ml.write_input()
        return Modflow.load("freyberg.nam", model_ws=function_tmpdir, check=False)

    benchmark(round_trip)
