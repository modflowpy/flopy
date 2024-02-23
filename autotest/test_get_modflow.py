"""Test get-modflow utility."""
import os
import platform
import sys
from os.path import expandvars
from pathlib import Path
from platform import system
from urllib.error import HTTPError

import pytest
from autotest.conftest import get_project_root_path
from flaky import flaky
from modflow_devtools.markers import requires_github
from modflow_devtools.misc import run_py_script

from flopy.utils import get_modflow
from flopy.utils.get_modflow import get_release, get_releases, select_bindir

rate_limit_msg = "rate limit exceeded"
flopy_dir = get_project_root_path()
get_modflow_script = flopy_dir / "flopy" / "utils" / "get_modflow.py"
bindir_options = {
    "flopy": Path(expandvars(r"%LOCALAPPDATA%\flopy")) / "bin"
    if system() == "Windows"
    else Path.home() / ".local" / "share" / "flopy" / "bin",
    "python": Path(sys.prefix)
    / ("Scripts" if system() == "Windows" else "bin"),
    "home": Path.home() / ".local" / "bin",
}
owner_options = [
    "MODFLOW-USGS",
]
repo_options = {
    "executables": [
        "crt",
        "gridgen",
        "mf2000",
        "mf2005",
        "mf2005dbl",
        "mf6",
        "mflgr",
        "mflgrdbl",
        "mfnwt",
        "mfnwtdbl",
        "mfusg",
        "mfusgdbl",
        "mp6",
        "mp7",
        "mt3dms",
        "mt3dusgs",
        "sutra",
        "swtv4",
        "triangle",
        "vs2dt",
        "zbud6",
        "zonbud3",
        "zonbudusg",
        "libmf6",
    ],
    "modflow6": ["mf6", "mf5to6", "zbud6", "libmf6"],
    "modflow6-nightly-build": ["mf6", "mf5to6", "zbud6", "libmf6"],
}

if system() == "Windows":
    bindir_options["windowsapps"] = Path(
        expandvars(r"%LOCALAPPDATA%\Microsoft\WindowsApps")
    )
else:
    bindir_options["system"] = Path("/usr") / "local" / "bin"


@pytest.fixture
def downloads_dir(tmp_path_factory):
    downloads_dir = tmp_path_factory.mktemp("Downloads")
    return downloads_dir


@pytest.fixture(autouse=True)
def create_home_local_bin():
    # make sure $HOME/.local/bin exists for :home option
    home_local = Path.home() / ".local" / "bin"
    home_local.mkdir(parents=True, exist_ok=True)


def run_get_modflow_script(*args):
    return run_py_script(get_modflow_script, *args, verbose=True)


def append_ext(path: str):
    if system() == "Windows":
        return f"{path}{'.dll' if 'libmf6' in path else '.exe'}"
    elif system() == "Darwin":
        return f"{path}{'.dylib' if 'libmf6' in path else ''}"
    elif system() == "Linux":
        return f"{path}{'.so' if 'libmf6' in path else ''}"


@pytest.mark.parametrize("per_page", [-1, 0, 101, 1000])
def test_get_releases_bad_page_size(per_page):
    with pytest.raises(ValueError):
        get_releases(repo="executables", per_page=per_page)


@flaky
@requires_github
@pytest.mark.parametrize("repo", repo_options.keys())
def test_get_releases(repo):
    releases = get_releases(repo=repo)
    assert "latest" in releases


@flaky
@requires_github
@pytest.mark.parametrize("repo", repo_options.keys())
def test_get_release(repo):
    tag = "latest"
    release = get_release(repo=repo, tag=tag)
    assets = release["assets"]

    expected_assets = ["linux.zip", "mac.zip", "win64.zip"]
    expected_ostags = [a.replace(".zip", "") for a in expected_assets]
    actual_assets = [asset["name"] for asset in assets]

    if repo == "modflow6":
        # can remove if modflow6 releases follow asset name conventions followed in executables and nightly build repos
        assert {a.rpartition("_")[2] for a in actual_assets} >= {
            a for a in expected_assets if not a.startswith("win")
        }
    elif repo == "modflow6-nightly-build":
        expected_assets.append("macarm.zip")
    else:
        for ostag in expected_ostags:
            assert any(
                ostag in a for a in actual_assets
            ), f"dist not found for {ostag}"


@pytest.mark.parametrize("bindir", bindir_options.keys())
def test_select_bindir(bindir, function_tmpdir):
    expected_path = bindir_options[bindir]
    if not os.access(expected_path, os.W_OK):
        pytest.skip(f"{expected_path} is not writable")
    selected = select_bindir(f":{bindir}")

    if system() != "Darwin":
        assert selected == expected_path
    else:
        # for some reason sys.prefix can return different python
        # installs when invoked here and get_modflow.py on macOS
        #   https://github.com/modflowpy/flopy/actions/runs/3331965840/jobs/5512345032#step:8:1835
        #
        # work around by just comparing the end of the bin path
        # should be .../Python.framework/Versions/<version>/bin
        assert selected.parts[-4:] == expected_path.parts[-4:]


def test_script_help():
    assert get_modflow_script.exists()
    stdout, stderr, returncode = run_get_modflow_script("-h")
    assert "usage" in stdout
    assert len(stderr) == 0
    assert returncode == 0


@flaky
@requires_github
@pytest.mark.slow
def test_script_invalid_options(function_tmpdir, downloads_dir):
    # try with bindir that doesn't exist
    bindir = function_tmpdir / "bin1"
    assert not bindir.exists()
    stdout, stderr, returncode = run_get_modflow_script(bindir)
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert "does not exist" in stderr
    assert returncode == 1

    # attempt to fetch a non-existing release-id
    bindir.mkdir()
    assert bindir.exists()
    stdout, stderr, returncode = run_get_modflow_script(
        bindir, "--release-id", "1.9", "--downloads-dir", downloads_dir
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert "Release 1.9 not found" in stderr
    assert returncode == 1

    # try to select an invalid --subset
    bindir = function_tmpdir / "bin2"
    bindir.mkdir()
    stdout, stderr, returncode = run_get_modflow_script(
        bindir, "--subset", "mfnwt,mpx", "--downloads-dir", downloads_dir
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert "subset item not found: mpx" in stderr
    assert returncode == 1


@flaky
@requires_github
@pytest.mark.slow
def test_script_valid_options(function_tmpdir, downloads_dir):
    # fetch latest
    bindir = function_tmpdir / "bin1"
    bindir.mkdir()
    stdout, stderr, returncode = run_get_modflow_script(
        bindir, "--downloads-dir", downloads_dir
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert len(stderr) == returncode == 0
    files = [item.name for item in bindir.iterdir() if item.is_file()]
    assert len(files) > 20

    # valid subset
    bindir = function_tmpdir / "bin2"
    bindir.mkdir()
    stdout, stderr, returncode = run_get_modflow_script(
        bindir, "--subset", "mfnwt,mp6", "--downloads-dir", downloads_dir
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert len(stderr) == returncode == 0
    files = [item.stem for item in bindir.iterdir() if item.is_file()]
    assert sorted(files) == ["mfnwt", "mfnwtdbl", "mp6"]

    # similar as before, but also specify a ostag
    bindir = function_tmpdir / "bin3"
    bindir.mkdir()
    stdout, stderr, returncode = run_get_modflow_script(
        bindir,
        "--subset",
        "mfnwt",
        "--release-id",
        "2.0",
        "--ostag",
        "win64",
        "--downloads-dir",
        downloads_dir,
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert len(stderr) == returncode == 0
    files = [item.name for item in bindir.iterdir() if item.is_file()]
    assert sorted(files) == ["mfnwt.exe", "mfnwtdbl.exe"]


@flaky
@requires_github
@pytest.mark.slow
@pytest.mark.parametrize("owner", owner_options)
@pytest.mark.parametrize("repo", repo_options.keys())
def test_script(function_tmpdir, owner, repo, downloads_dir):
    bindir = str(function_tmpdir)
    stdout, stderr, returncode = run_get_modflow_script(
        bindir,
        "--owner",
        owner,
        "--repo",
        repo,
        "--downloads-dir",
        downloads_dir,
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")

    paths = list(function_tmpdir.glob("*"))
    names = [p.name for p in paths]
    expected_names = [append_ext(p) for p in repo_options[repo]]
    assert set(names) >= set(expected_names)


@flaky
@requires_github
@pytest.mark.slow
@pytest.mark.parametrize("owner", owner_options)
@pytest.mark.parametrize("repo", repo_options.keys())
def test_python_api(function_tmpdir, owner, repo, downloads_dir):
    bindir = str(function_tmpdir)
    try:
        get_modflow(
            bindir, owner=owner, repo=repo, downloads_dir=downloads_dir
        )
    except HTTPError as err:
        if err.code == 403:
            pytest.skip(f"GitHub {rate_limit_msg}")

    paths = list(function_tmpdir.glob("*"))
    names = [p.name for p in paths]
    expected_names = [append_ext(p) for p in repo_options[repo]]
    assert set(names) >= set(expected_names)
