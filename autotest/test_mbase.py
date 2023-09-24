from pathlib import Path
from platform import system
from shutil import copy, copytree, which

import pytest
from modflow_devtools.markers import requires_exe
from modflow_devtools.misc import set_dir

from flopy import run_model
from flopy.mbase import resolve_exe
from flopy.utils.flopy_io import relpath_safe


_system = system()


@pytest.fixture
def mf6_model_path(example_data_path):
    return example_data_path / "mf6" / "test006_gwf3"


@requires_exe("mf6")
@pytest.mark.parametrize("use_ext", [True, False])
def test_resolve_exe_by_name(use_ext):
    ext = ".exe" if use_ext else ""
    expected = which("mf6").lower()
    actual = resolve_exe(f"mf6{ext}")
    assert actual.lower() == expected
    assert which(actual)


@requires_exe("mf6")
@pytest.mark.parametrize("use_ext", [True, False])
def test_resolve_exe_by_abs_path(use_ext):
    abs_path = which("mf6")
    if _system == "Windows" and not use_ext:
        abs_path = abs_path[:-4]
    elif _system != "Windows" and use_ext:
        abs_path = f"{abs_path}.exe"
    expected = which("mf6").lower()
    actual = resolve_exe(abs_path)
    assert actual.lower() == expected
    assert which(actual)


@requires_exe("mf6")
@pytest.mark.parametrize("use_ext", [True, False])
@pytest.mark.parametrize("forgive", [True, False])
def test_resolve_exe_by_rel_path(function_tmpdir, use_ext, forgive):
    ext = ".exe" if use_ext else ""
    expected = which("mf6").lower()

    bin_dir = function_tmpdir / "bin"
    bin_dir.mkdir()
    inner_dir = function_tmpdir / "inner"
    inner_dir.mkdir()

    with set_dir(inner_dir):
        # copy exe to relative dir
        new_exe_path = bin_dir / Path(expected).name
        copy(expected, new_exe_path)
        assert new_exe_path.is_file()

        expected = which(str(new_exe_path.absolute())).lower()
        actual = resolve_exe(f"../bin/mf6{ext}")
        assert actual.lower() == expected
        assert which(actual)

        # check behavior if exe DNE
        with (
            pytest.warns(UserWarning)
            if forgive
            else pytest.raises(FileNotFoundError)
        ):
            assert not resolve_exe("../bin/mf2005", forgive)


def test_run_model_when_namefile_not_in_model_ws(
    mf6_model_path, function_tmpdir
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
@pytest.mark.parametrize(
    "exe",
    [
        "mf6",
        Path(which("mf6") or ""),
        relpath_safe(Path(which("mf6") or "")),
    ],
)
def test_run_model(mf6_model_path, function_tmpdir, use_paths, exe):
    ws = function_tmpdir / "ws"
    copytree(mf6_model_path, ws)

    success, buff = run_model(
        exe_name=exe,
        namefile="mfsim.nam",
        model_ws=ws if use_paths else str(ws),
        silent=False,
        report=True,
    )

    assert success
    assert any(buff)
    assert any(ws.glob("*.lst"))


@requires_exe("mf6")
@pytest.mark.parametrize("use_ext", [True, False])
def test_run_model_exe_rel_path(mf6_model_path, function_tmpdir, use_ext):
    if use_ext and system() != "Windows":
        pytest.skip(".exe extensions are Windows-only")

    ws = function_tmpdir / "ws"
    copytree(mf6_model_path, ws)

    ext = ".exe" if use_ext else ""
    mf6 = which("mf6").lower()

    bin_dir = function_tmpdir / "bin"
    bin_dir.mkdir()
    inner_dir = function_tmpdir / "inner"
    inner_dir.mkdir()

    with set_dir(inner_dir):
        # copy exe to relative dir
        copy(mf6, bin_dir / "mf6")
        assert (bin_dir / "mf6").is_file()

        success, buff = run_model(
            exe_name=f"../bin/mf6{ext}",
            namefile="mfsim.nam",
            model_ws=ws,
            silent=False,
            report=True,
        )

        assert success
        assert any(buff)
        assert any(ws.glob("*.lst"))
