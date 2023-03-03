from pathlib import Path
from shutil import copy, copytree, which

import pytest
from modflow_devtools.markers import requires_exe
from modflow_devtools.misc import set_dir

from flopy import run_model
from flopy.mbase import resolve_exe


@requires_exe("mf6")
def test_resolve_exe(function_tmpdir):
    expected = which("mf6").lower()

    # named executable
    actual = resolve_exe("mf6")
    assert actual.lower() == expected
    assert which(actual)

    # full path to exe
    actual = resolve_exe(which("mf6"))
    assert actual.lower() == expected
    assert which(actual)

    # relative path to exe
    bin_dir = function_tmpdir / "bin"
    bin_dir.mkdir()
    inner_dir = function_tmpdir / "inner"
    inner_dir.mkdir()
    copy(expected, bin_dir / "mf6")
    assert (bin_dir / "mf6").is_file()
    with set_dir(inner_dir):
        expected = which(str(Path(bin_dir / "mf6").absolute())).lower()
        actual = resolve_exe("../bin/mf6")
        assert actual.lower() == expected
        assert which(actual)
        with pytest.raises(FileNotFoundError):
            resolve_exe("../bin/mf2005")


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
