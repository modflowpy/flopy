from pathlib import Path
from shutil import copytree, which

import pytest
from modflow_devtools.markers import requires_exe

from flopy import run_model


@pytest.fixture
def mf6_model_path(example_data_path):
    return example_data_path / "mf6" / "test006_gwf3"


def test_run_mf6_model_when_namefile_not_in_model_ws(
    mf6_model_path, example_data_path, function_tmpdir
):
    # copy input files to temp workspace
    ws = function_tmpdir / "ws"
    copytree(mf6_model_path, ws)

    # create dir inside the workspace
    inner_ws = ws / "inner"
    inner_ws.mkdir()

    # move the namefile into the inner dir
    namefile_path = ws / "mfsim.nam"
    namefile_path.rename(inner_ws / "mfsim.nam")

    with pytest.raises(FileNotFoundError):
        run_model(
            exe_name="mf6",
            namefile=namefile_path.name,
            model_ws=ws,
            silent=False,
            report=True,
        )


@pytest.mark.mf6
@requires_exe("mf6")
@pytest.mark.parametrize("use_paths", [True, False])
def test_run_mf6_model(mf6_model_path, function_tmpdir, use_paths):
    # copy input files to temp workspace
    ws = function_tmpdir / "ws"
    copytree(mf6_model_path, ws)

    if use_paths:
        success, buff = run_model(
            exe_name=Path(which("mf6")),
            namefile="mfsim.nam",
            model_ws=ws,
            silent=False,
            report=True,
        )
    else:
        success, buff = run_model(
            exe_name="mf6",
            namefile="mfsim.nam",
            model_ws=str(ws),
            silent=False,
            report=True,
        )

    assert success
    assert any(buff)
    assert any(ws.glob("*.lst"))
