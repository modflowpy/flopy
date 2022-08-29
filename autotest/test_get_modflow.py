"""Test get-modflow utility."""
import sys
import urllib
from urllib.error import HTTPError

import pytest
from autotest.conftest import (
    get_project_root_path,
    requires_github,
    run_py_script,
)
from flaky import flaky

from flopy.utils import get_modflow

flopy_dir = get_project_root_path(__file__)
get_modflow_script = flopy_dir / "flopy" / "utils" / "get_modflow.py"


@pytest.fixture(scope="session")
def downloads_dir(tmp_path_factory):
    downloads_dir = tmp_path_factory.mktemp("Downloads")
    return downloads_dir


def run_get_modflow_script(*args):
    return run_py_script(get_modflow_script, *args)


def test_script_usage():
    assert get_modflow_script.exists()
    stdout, stderr, returncode = run_get_modflow_script("-h")
    assert "usage" in stdout
    assert len(stderr) == 0
    assert returncode == 0


rate_limit_msg = "rate limit exceeded"


@flaky
@requires_github
def test_get_modflow_script(tmp_path, downloads_dir):
    # exit if extraction directory does not exist
    bindir = tmp_path / "bin1"
    assert not bindir.exists()
    stdout, stderr, returncode = run_get_modflow_script(bindir)
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert "does not exist" in stderr
    assert returncode == 1

    # ensure extraction directory exists
    bindir.mkdir()
    assert bindir.exists()

    # attempt to fetch a non-existing release-id
    stdout, stderr, returncode = run_get_modflow_script(
        bindir, "--release-id", "1.9", "--downloads-dir", downloads_dir
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert "Release '1.9' not found" in stderr
    assert returncode == 1

    # fetch latest
    stdout, stderr, returncode = run_get_modflow_script(
        bindir, "--downloads-dir", downloads_dir
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert len(stderr) == returncode == 0
    files = [item.name for item in bindir.iterdir() if item.is_file()]
    assert len(files) > 20

    # take only a few files using --subset, starting with invalid
    bindir = tmp_path / "bin2"
    bindir.mkdir()
    stdout, stderr, returncode = run_get_modflow_script(
        bindir, "--subset", "mfnwt,mpx", "--downloads-dir", downloads_dir
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert "subset item not found: mpx" in stderr
    assert returncode == 1
    # now valid subset
    stdout, stderr, returncode = run_get_modflow_script(
        bindir, "--subset", "mfnwt,mp6", "--downloads-dir", downloads_dir
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert len(stderr) == returncode == 0
    files = [item.stem for item in bindir.iterdir() if item.is_file()]
    assert sorted(files) == ["mfnwt", "mfnwtdbl", "mp6"]

    # similar as before, but also specify a ostag
    bindir = tmp_path / "bin3"
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
def test_get_nightly_script(tmp_path, downloads_dir):
    bindir = tmp_path / "bin1"
    bindir.mkdir()

    stdout, stderr, returncode = run_get_modflow_script(
        bindir,
        "--repo",
        "modflow6-nightly-build",
        "--downloads-dir",
        downloads_dir,
    )
    if rate_limit_msg in stderr:
        pytest.skip(f"GitHub {rate_limit_msg}")
    assert len(stderr) == returncode == 0
    files = [item.name for item in bindir.iterdir() if item.is_file()]
    assert len(files) >= 4


@flaky
@requires_github
def test_get_modflow(tmpdir):
    try:
        get_modflow(tmpdir)
    except HTTPError as err:
        if err.code == 403:
            pytest.skip(f"GitHub {rate_limit_msg}")

    actual = [p.name for p in tmpdir.glob("*")]
    expected = [
        (exe + ".exe" if sys.platform.startswith("win") else exe)
        for exe in [
            "crt",
            "gridgen",
            "gsflow",
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
        ]
    ]

    assert all(exe in actual for exe in expected)


@flaky
@requires_github
def test_get_nightly(tmpdir):
    try:
        get_modflow(tmpdir, repo="modflow6-nightly-build")
    except urllib.error.HTTPError as err:
        if err.code == 403:
            pytest.skip(f"GitHub {rate_limit_msg}")

    actual = [p.name for p in tmpdir.glob("*")]
    expected = [
        (exe + ".exe" if sys.platform.startswith("win") else exe)
        for exe in ["mf6", "mf5to6", "zbud6"]
    ]

    assert all(exe in actual for exe in expected)
