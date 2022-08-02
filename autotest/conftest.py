import os
import socket
import subprocess
from os import environ
from os.path import basename, normpath
from pathlib import Path
from platform import system
from shutil import copytree, which
from typing import List, Optional
from urllib import request
from warnings import warn

import pytest

# constants

SHAPEFILE_EXTENSIONS = ["prj", "shx", "dbf"]


# misc utilities


def get_project_root_path(path=None) -> Path:
    """
    Infers the path to the project root given the path to the current working directory.
    The current working location must be somewhere in the project, below the project root.

    This function aims to work whether invoked from the autotests directory, the examples
    directory, the flopy module directory, or any subdirectories of these. GitHub Actions
    CI runners, local `act` runners for GitHub CI, and local environments are supported.
    This function can be modified to support other flopy testing environments if needed.

    Parameters
    ----------
    path : the path to the current working directory

    Returns
    -------
        The absolute path to the project root
    """

    cwd = Path(path) if path is not None else Path(os.getcwd())

    def backtrack_or_raise():
        tries = [1]
        if is_in_ci():
            tries.append(2)
        for t in tries:
            parts = cwd.parts[0: cwd.parts.index("flopy") + t]
            pth = Path(*parts)
            if next(iter([p for p in pth.glob("setup.cfg")]), None) is not None:
                return pth
        raise Exception(
            f"Can't infer location of project root from {cwd} "
            f"(run from project root, flopy module, examples, or autotest)"
        )

    if cwd.name == "autotest":
        # we're in top-level autotest folder
        return cwd.parent
    elif "autotest" in cwd.parts and cwd.parts.index("autotest") > cwd.parts.index("flopy"):
        # we're somewhere inside autotests
        parts = cwd.parts[0: cwd.parts.index("autotest")]
        return Path(*parts)
    elif "examples" in cwd.parts and cwd.parts.index("examples") > cwd.parts.index("flopy"):
        # we're somewhere inside examples folder
        parts = cwd.parts[0: cwd.parts.index("examples")]
        return Path(*parts)
    elif "flopy" in cwd.parts:
        if cwd.parts.count("flopy") >= 2:
            # we're somewhere inside the project or flopy module
            return backtrack_or_raise()
        elif cwd.parts.count("flopy") == 1:
            if cwd.name == "flopy":
                # we're in project root
                return cwd
            elif cwd.name == ".working":
                # we're in local `act` github actions runner
                return backtrack_or_raise()
            else:
                raise Exception(
                    f"Can't infer location of project root from {cwd}"
                    f"(run from project root, flopy module, examples, or autotest)"
                )
    else:
        raise Exception(
            f"Can't infer location of project root from {cwd}"
            f"(run from project root, flopy module, examples, or autotest)"
        )


def get_example_data_path(path=None) -> Path:
    """
    Gets the absolute path to example models and data.
    The path argument is a hint, interpreted as
    the current working location.
    """
    return get_project_root_path(path) / "examples" / "data"


def get_flopy_data_path(path=None) -> Path:
    """
    Gets the absolute path to flopy module data.
    The path argument is a hint, interpreted as
    the current working location.
    """
    return get_project_root_path(path) / "flopy" / "data"


def get_current_branch() -> str:
    # check if on GitHub Actions CI
    ref = environ.get("GITHUB_REF")
    if ref is not None:
        return basename(normpath(ref)).lower()

    # otherwise ask git about it
    try:
        b = subprocess.Popen(
            ("git", "status"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        ).communicate()[0]

        if isinstance(b, bytes):
            b = b.decode("utf-8")

        for line in b.splitlines():
            if "On branch" in line:
                return line.replace("On branch ", "").rstrip().lower()
    except:
        raise ValueError(
            "Could not determine current branch. Is git installed?"
        )


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


def is_in_ci():
    return "CI" in os.environ


def is_github_rate_limited() -> Optional[bool]:
    """
    Determines if a GitHub API rate limit is applied to the current IP.
    Note that running this function will consume an API request!

    Returns
    -------
        True if rate-limiting is applied, otherwise False (or None if the connection fails).
    """
    try:
        with request.urlopen(
            "https://api.github.com/users/octocat"
        ) as response:
            remaining = int(response.headers["x-ratelimit-remaining"])
            if remaining < 10:
                warn(
                    f"Only {remaining} GitHub API requests remaining before rate-limiting"
                )
            return remaining > 0
    except:
        return None


def requires_exes(exes):
    return pytest.mark.skipif(
        any(which(exe) is None for exe in exes),
        reason=f"requires executables: {', '.join(exes)}",
    )


def requires_exe(exe):
    return requires_exes([exe])


def requires_platform(platform, ci_only=False):
    return pytest.mark.skipif(
        system().lower() != platform.lower() and (is_in_ci() if ci_only else True),
        reason=f"only compatible with platform: {platform.lower()}",
    )


def excludes_platform(platform, ci_only=False):
    return pytest.mark.skipif(
        system().lower() == platform.lower() and (is_in_ci() if ci_only else True),
        reason=f"not compatible with platform: {platform.lower()}",
    )


def requires_branch(branch):
    current = get_current_branch()
    return pytest.mark.skipif(
        current != branch, reason=f"must run on branch: {branch}"
    )


def excludes_branch(branch):
    current = get_current_branch()
    return pytest.mark.skipif(
        current == branch, reason=f"can't run on branch: {branch}"
    )


requires_github = pytest.mark.skipif(
    not is_connected("github.com"),
    reason="github.com is required.")


# example data fixtures

@pytest.fixture(scope="session")
def project_root_path(request) -> Path:
    return get_project_root_path(request.session.path)


@pytest.fixture(scope="session")
def example_data_path(request) -> Path:
    return get_example_data_path(request.session.path)


@pytest.fixture(scope="session")
def flopy_data_path(request) -> Path:
    return get_flopy_data_path(request.session.path)


@pytest.fixture(scope="session")
def example_shapefiles(example_data_path) -> List[Path]:
    return [f.resolve() for f in (example_data_path / "prj_test").glob("*")]


# keepable temporary directory fixtures for various scopes

@pytest.fixture(scope="function")
def tmpdir(tmpdir_factory, request) -> Path:
    node = (
        request.node.name.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
    )
    temp = Path(tmpdir_factory.mktemp(node))
    yield Path(temp)

    keep = request.config.getoption("--keep")
    if keep:
        copytree(temp, Path(keep) / temp.name)


@pytest.fixture(scope="class")
def class_tmpdir(tmpdir_factory, request) -> Path:
    assert (
        request.cls is not None
    ), "Class-scoped temp dir fixture must be used on class"
    temp = Path(tmpdir_factory.mktemp(request.cls.__name__))
    yield temp

    keep = request.config.getoption("--keep")
    if keep:
        copytree(temp, Path(keep) / temp.name)


@pytest.fixture(scope="module")
def module_tmpdir(tmpdir_factory, request) -> Path:
    temp = Path(tmpdir_factory.mktemp(request.module.__name__))
    yield temp

    keep = request.config.getoption("--keep")
    if keep:
        copytree(temp, Path(keep) / temp.name)


@pytest.fixture(scope="session")
def session_tmpdir(tmpdir_factory, request) -> Path:
    temp = Path(tmpdir_factory.mktemp(request.session.name))
    yield temp

    keep = request.config.getoption("--keep")
    if keep:
        copytree(temp, Path(keep) / temp.name)


# pytest configuration hooks


def pytest_addoption(parser):
    parser.addoption(
        "-K",
        "--keep",
        action="store",
        default=None,
        help="Move the contents of temporary test directories to correspondingly named subdirectories at the KEEP "
        "location after tests complete. This option can be used to exclude test results from automatic cleanup, "
        "e.g. for manual inspection. The provided path is created if it does not already exist. An error is "
        "thrown if any matching files already exist.",
    )

    parser.addoption(
        "-M",
        "--meta",
        action="store",
        metavar="NAME",
        help="Marker indicating a test is only run by other tests (e.g., the test framework testing itself).",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "meta(name): mark test to run only inside other groups of tests.",
    )


def pytest_runtest_setup(item):
    # apply meta-test marker
    metagroups = [mark.args[0] for mark in item.iter_markers(name="meta")]
    if metagroups and item.config.getoption("--meta") not in metagroups:
        pytest.skip()
