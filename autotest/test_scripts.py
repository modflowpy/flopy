"""Test scripts."""
import socket
import sys
from pathlib import Path
from subprocess import Popen, PIPE

import pytest


def is_connected(hostname):
    """See https://stackoverflow.com/a/20913928/ to test hostname."""
    try:
        host = socket.gethostbyname(hostname)
        s = socket.create_connection((host, 80), 2)
        s.close()
        return True
    except Exception:
        pass
    return False


requires_github = pytest.mark.skipif(
    not is_connected("github.com"), reason="github.com is required."
)

flopy_dir = Path(__file__).parents[1] / "flopy"
get_modflow_script = flopy_dir / "utils" / "get_modflow.py"


@pytest.fixture(scope="session")
def downloads_dir(tmp_path_factory):
    downloads_dir = tmp_path_factory.mktemp("Downloads")
    return downloads_dir


def run_py_script(script, *args):
    """Run a Python script, return tuple (stdout, stderr, returncode)."""
    args = [sys.executable, str(script)] + [str(g) for g in args]
    print("running: " + " ".join(args))
    p = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p.communicate()
    stdout = stdout.decode()
    stderr = stderr.decode()
    returncode = p.returncode
    return stdout, stderr, returncode


def run_get_modflow_script(*args):
    return run_py_script(get_modflow_script, *args)


def test_script_usage():
    assert get_modflow_script.exists()
    stdout, stderr, returncode = run_get_modflow_script("-h")
    assert "usage" in stdout
    assert len(stderr) == 0
    assert returncode == 0


@requires_github
def test_get_modflow(tmp_path, downloads_dir):
    # exit if extraction directory does not exist
    bindir = tmp_path / "bin1"
    assert not bindir.exists()
    stdout, stderr, returncode = run_get_modflow_script(bindir)
    assert "does not exist" in stderr
    assert returncode == 1

    # ensure extraction directory exists
    bindir.mkdir()
    assert bindir.exists()

    # attempt to fetch a non-existing release-id
    stdout, stderr, returncode = run_get_modflow_script(
        bindir, "--release-id", "1.9", "--downloads-dir", downloads_dir
    )
    assert "Release '1.9' not found" in stderr
    assert returncode == 1

    # fetch latest
    stdout, stderr, returncode = run_get_modflow_script(
        bindir, "--downloads-dir", downloads_dir
    )
    assert len(stderr) == returncode == 0
    files = [item.name for item in bindir.iterdir() if item.is_file()]
    assert len(files) > 20

    # take only a few files using --subset, starting with invalid
    bindir = tmp_path / "bin2"
    bindir.mkdir()
    stdout, stderr, returncode = run_get_modflow_script(
        bindir, "--subset", "mfnwt,mpx", "--downloads-dir", downloads_dir
    )
    assert "subset item not found: mpx" in stderr
    assert returncode == 1
    # now valid subset
    stdout, stderr, returncode = run_get_modflow_script(
        bindir, "--subset", "mfnwt,mp6", "--downloads-dir", downloads_dir
    )
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
    assert len(stderr) == returncode == 0
    files = [item.name for item in bindir.iterdir() if item.is_file()]
    assert sorted(files) == ["mfnwt.exe", "mfnwtdbl.exe"]


@requires_github
def test_get_mf6_nightly(tmp_path, downloads_dir):
    bindir = tmp_path / "bin1"
    bindir.mkdir()
    stdout, stderr, returncode = run_get_modflow_script(
        bindir,
        "--repo",
        "modflow6-nightly-build",
        "--downloads-dir",
        downloads_dir,
    )
    assert len(stderr) == returncode == 0
    files = [item.name for item in bindir.iterdir() if item.is_file()]
    assert len(files) >= 4
