"""Test get-modflow utility."""

import io
import os
import sys
import urllib.request
from os.path import expandvars
from pathlib import Path
from platform import system
from urllib.error import HTTPError, URLError

import pytest
from flaky import flaky
from modflow_devtools.markers import requires_github
from modflow_devtools.misc import run_py_script

from autotest.conftest import get_project_root_path
from flopy.utils import get_modflow_module as get_modflow
from flopy.utils.get_modflow import get_release, get_releases, run_main, select_bindir

rate_limit_msg = "rate limit exceeded"
flopy_dir = get_project_root_path()
get_modflow_script = flopy_dir / "flopy" / "utils" / "get_modflow.py"
bindir_options = {
    "flopy": (
        Path(expandvars(r"%LOCALAPPDATA%\flopy")) / "bin"
        if system() == "Windows"
        else Path.home() / ".local" / "share" / "flopy" / "bin"
    ),
    "python": Path(sys.prefix) / ("Scripts" if system() == "Windows" else "bin"),
    "home": Path.home() / ".local" / "bin",
}
owner_options = ["MODFLOW-ORG"]
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
        "zonbud",
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


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    # make retries instant
    monkeypatch.setattr(get_modflow, "http_retry_delay", 0.0)
    monkeypatch.setenv("GET_MODFLOW_RETRY_DELAY", "0")


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


class FakeResponse(io.BytesIO):
    def __init__(self, data=b"", headers=None):
        super().__init__(data)
        self.headers = headers or {}


def fake_urlopen(fail_times, exc, data=b"{}", headers=None):
    # raises exc the first fail_times calls, then returns a fresh FakeResponse;
    # call count is tracked on .calls
    state = {"n": 0}

    def _fake(request, timeout=10, quiet=False):
        state["n"] += 1
        if state["n"] <= fail_times:
            raise exc
        return FakeResponse(data, headers)

    _fake.calls = state
    return _fake


def reset_error():
    return URLError(ConnectionResetError(104, "Connection reset by peer"))


def test_fetch_returns_body_and_headers(monkeypatch):
    fake = fake_urlopen(
        0, None, data=b'{"a": 1}', headers={"x-ratelimit-remaining": "42"}
    )
    monkeypatch.setattr(get_modflow, "urlopen", fake)
    body, headers = get_modflow.fetch(object())
    assert body == b'{"a": 1}'
    assert headers.get("x-ratelimit-remaining") == "42"


def test_fetch_retries_transient_then_succeeds(monkeypatch):
    fake = fake_urlopen(2, reset_error(), data=b"OK")
    monkeypatch.setattr(get_modflow, "urlopen", fake)

    body, _ = get_modflow.fetch(object(), tries=3, delay=0)

    assert body == b"OK"
    assert fake.calls["n"] == 3


def test_fetch_default_tries_from_module(monkeypatch):
    # no tries= arg: falls back to the module-level max_http_tries
    fake = fake_urlopen(99, reset_error())
    monkeypatch.setattr(get_modflow, "urlopen", fake)
    monkeypatch.setattr(get_modflow, "max_http_tries", 4)

    with pytest.raises(URLError):
        get_modflow.fetch(object(), delay=0)
    assert fake.calls["n"] == 4


def test_fetch_reraises_after_cap(monkeypatch):
    fake = fake_urlopen(99, reset_error())
    monkeypatch.setattr(get_modflow, "urlopen", fake)

    with pytest.raises(URLError):
        get_modflow.fetch(object(), tries=3, delay=0)
    assert fake.calls["n"] == 3


def test_fetch_tries_1_disables_retry(monkeypatch):
    fake = fake_urlopen(99, reset_error())
    monkeypatch.setattr(get_modflow, "urlopen", fake)

    with pytest.raises(URLError):
        get_modflow.fetch(object(), tries=1)
    assert fake.calls["n"] == 1


def test_fetch_non_transient_raises_immediately(monkeypatch):
    slept = []
    monkeypatch.setattr(get_modflow.time, "sleep", slept.append)
    fake = fake_urlopen(99, HTTPError("u", 401, "Unauthorized", {}, None))
    monkeypatch.setattr(get_modflow, "urlopen", fake)

    with pytest.raises(HTTPError):
        get_modflow.fetch(object(), tries=3, delay=0)
    assert fake.calls["n"] == 1
    assert slept == []


def test_fetch_404_not_retried(monkeypatch):
    slept = []
    monkeypatch.setattr(get_modflow.time, "sleep", slept.append)
    fake = fake_urlopen(99, HTTPError("u", 404, "Not Found", {}, None))
    monkeypatch.setattr(get_modflow, "urlopen", fake)

    with pytest.raises(HTTPError) as exc_info:
        get_modflow.fetch(object(), tries=3, delay=0)
    assert exc_info.value.code == 404
    assert fake.calls["n"] == 1
    assert slept == []


def test_sleep_before_retry_backoff_and_cap(monkeypatch):
    slept = []
    monkeypatch.setattr(get_modflow.time, "sleep", slept.append)
    err = reset_error()

    get_modflow._sleep_before_retry(1, 3, 2.0, err, quiet=True)
    get_modflow._sleep_before_retry(2, 3, 2.0, err, quiet=True)
    get_modflow._sleep_before_retry(3, 3, 2.0, err, quiet=True)
    assert slept == [2.0, 4.0, 8.0]  # exponential

    # ceiling
    get_modflow._sleep_before_retry(1, 3, 100.0, err, quiet=True)
    assert slept[-1] == get_modflow._max_retry_delay

    # Retry-After header wins when larger
    retry_after = HTTPError("u", 503, "err", {"Retry-After": "30"}, None)
    get_modflow._sleep_before_retry(1, 3, 2.0, retry_after, quiet=True)
    assert slept[-1] == 30.0

    # zero delay stays zero
    get_modflow._sleep_before_retry(5, 3, 0.0, err, quiet=True)
    assert slept[-1] == 0.0


def test_download_atomic_success(tmp_path, monkeypatch):
    monkeypatch.setattr(
        get_modflow, "urlopen", fake_urlopen(1, reset_error(), data=b"PK\x03\x04zip")
    )
    dest = tmp_path / "asset.zip"

    request = urllib.request.Request("http://example/asset.zip")
    get_modflow.download(request, dest, quiet=True, tries=3, delay=0)

    assert dest.read_bytes() == b"PK\x03\x04zip"
    assert not (tmp_path / "asset.zip.part").exists()


def test_download_failure_preserves_existing(tmp_path, monkeypatch):
    monkeypatch.setattr(get_modflow, "urlopen", fake_urlopen(99, reset_error()))
    dest = tmp_path / "asset.zip"
    dest.write_bytes(b"OLD-GOOD-CACHE")

    request = urllib.request.Request("http://example/asset.zip")
    with pytest.raises(URLError):
        get_modflow.download(request, dest, quiet=True, tries=3, delay=0)

    assert dest.read_bytes() == b"OLD-GOOD-CACHE"
    assert not (tmp_path / "asset.zip.part").exists()


def test_cli_forwards_retry_flags_to_run_main(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        get_modflow, "run_main", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["get_modflow.py", str(Path.home()), "--retries", "9", "--retry-delay", "0.5"],
    )
    tries_before = get_modflow.max_http_tries

    get_modflow.cli_main()

    assert captured["retries"] == 9
    assert captured["retry_delay"] == 0.5
    assert get_modflow.max_http_tries == tries_before


def test_cli_omitted_retry_flags_are_none(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        get_modflow, "run_main", lambda **kwargs: captured.update(kwargs)
    )
    monkeypatch.setattr(sys, "argv", ["get_modflow.py", str(Path.home())])

    get_modflow.cli_main()

    assert captured["retries"] is None
    assert captured["retry_delay"] is None


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
    expected_assets = ["linux.zip", "macarm.zip", "win64.zip"]
    expected_ostags = [a.replace(".zip", "") for a in expected_assets]
    actual_assets = [asset["name"] for asset in assets]

    if repo == "modflow6":
        # can remove if modflow6 releases follow the same asset name
        # convention used in the executables and nightly build repos
        assert {a.rpartition("_")[2] for a in actual_assets} >= {
            a for a in expected_assets if not a.startswith("win")
        }
    else:
        for ostag in expected_ostags:
            assert any(ostag in a for a in actual_assets), f"dist not found for {ostag}"


@pytest.mark.parametrize("bindir", bindir_options.keys())
def test_select_bindir(bindir, function_tmpdir):
    expected_path = bindir_options[bindir]
    if not os.access(expected_path, os.W_OK):
        pytest.skip(f"{expected_path} is not writable")
    selected = select_bindir(f":{bindir}")

    # For some reason sys.prefix can return different python
    # installs when invoked here and get_modflow.py on macOS.
    # Work around by just comparing the end of the bin path,
    # should be .../Python.framework/Versions/<version>/bin
    if system() != "Darwin":
        assert selected == expected_path
    else:
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
        bindir, "--owner", owner, "--repo", repo, "--downloads-dir", downloads_dir
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
        run_main(bindir, owner=owner, repo=repo, downloads_dir=downloads_dir)
    except HTTPError as err:
        if err.code == 403:
            pytest.skip(f"GitHub {rate_limit_msg}")

    paths = list(function_tmpdir.glob("*"))
    names = [p.name for p in paths]
    expected_names = [append_ext(p) for p in repo_options[repo]]
    assert set(names) >= set(expected_names)
