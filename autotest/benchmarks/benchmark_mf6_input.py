"""
Benchmarks for MODFLOW 6 I/O operations using models from:
- examples/data (local models)
- modflow-devtools registry (downloaded on-demand)
"""

from pathlib import Path
from shutil import copytree

import pytest
from modflow_devtools.models import DEFAULT_REGISTRY, LocalRegistry

from flopy.mf6 import MFSimulation

from .conftest import get_examples_path

# prefixes into the model registry
PREFIXES = ["mf6/test", "mf6/large", "mf2005"]


def pytest_generate_tests(metafunc):
    # Use the --models-path command line option once or more to specify
    # model directories. If at least one --models_path is provided,
    # external tests (i.e. those using models from an external repo)
    # will run against model input files found in the given location
    # on the local filesystem rather than model input files from the
    # official model registry. This is useful for testing changes to
    # test model input files during MF6 development. See conftest.py
    # for the models_path fixture and CLI argument definitions.
    if "model_name" in metafunc.fixturenames:
        # Try to get the models-path option, default to None if not available
        try:
            models_paths = metafunc.config.getoption("--models-path") or []
        except ValueError:
            models_paths = []

        models_paths = [Path(p).expanduser().resolve().absolute() for p in models_paths]
        registry = LocalRegistry() if any(models_paths) else DEFAULT_REGISTRY
        registry_type = type(registry).__name__.lower().replace("registry", "")
        metafunc.parametrize("registry", [registry], ids=[registry_type])
        models = []
        if "local" in registry_type:
            try:
                namefile_pattern = (
                    metafunc.config.getoption("--namefile-pattern") or "mfsim.nam"
                )
            except ValueError:
                namefile_pattern = "mfsim.nam"
            for path in models_paths:
                registry.index(path, namefile=namefile_pattern)
            models.extend(registry.models.keys())
        else:
            for model_prefix in PREFIXES:
                models.extend(
                    [m for m in registry.models.keys() if m.startswith(model_prefix)]
                )
        models = sorted(models)
        metafunc.parametrize("model_name", models, ids=models)


@pytest.mark.benchmark
@pytest.mark.slow
@pytest.mark.external
@pytest.mark.parametrize("use_pandas", [True, False], ids=["pandas", "nopandas"])
def test_load_simulation(function_tmpdir, benchmark, registry, model_name, use_pandas):
    registry.copy_to(function_tmpdir, model_name)
    benchmark(lambda: MFSimulation.load(sim_ws=function_tmpdir, use_pandas=use_pandas))


@pytest.mark.benchmark
@pytest.mark.external
@pytest.mark.slow
@pytest.mark.parametrize("use_pandas", [True, False], ids=["pandas", "nopandas"])
def test_write_simulation(function_tmpdir, benchmark, registry, model_name, use_pandas):
    registry.copy_to(function_tmpdir, model_name)
    sim = MFSimulation.load(sim_ws=function_tmpdir, use_pandas=use_pandas)
    benchmark(sim.write_simulation)
