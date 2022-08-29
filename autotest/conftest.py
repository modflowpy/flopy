import importlib
import os
import socket
import sys
from os import environ
from os.path import basename, normpath
from pathlib import Path
from platform import system
from shutil import copytree, which
from subprocess import PIPE, Popen
from typing import List, Optional
from urllib import request
from warnings import warn

import matplotlib.pyplot as plt
import pkg_resources
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
            parts = cwd.parts[0 : cwd.parts.index("flopy") + t]
            pth = Path(*parts)
            if (
                next(iter([p for p in pth.glob("setup.cfg")]), None)
                is not None
            ):
                return pth
        raise Exception(
            f"Can't infer location of project root from {cwd} "
            f"(run from project root, flopy module, examples, or autotest)"
        )

    if cwd.name == "autotest":
        # we're in top-level autotest folder
        return cwd.parent
    elif "autotest" in cwd.parts and cwd.parts.index(
        "autotest"
    ) > cwd.parts.index("flopy"):
        # we're somewhere inside autotests
        parts = cwd.parts[0 : cwd.parts.index("autotest")]
        return Path(*parts)
    elif "examples" in cwd.parts and cwd.parts.index(
        "examples"
    ) > cwd.parts.index("flopy"):
        # we're somewhere inside examples folder
        parts = cwd.parts[0 : cwd.parts.index("examples")]
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
    if not which("git"):
        raise RuntimeError("'git' required to determine current branch")
    stdout, stderr, code = run_cmd("git", "rev-parse", "--abbrev-ref", "HEAD")
    if code == 0 and stdout:
        return stdout.strip().lower()
    raise ValueError(f"Could not determine current branch: {stderr}")


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
    # if running in GitHub Actions CI, "CI" variable always set to true
    # https://docs.github.com/en/actions/learn-github-actions/environment-variables#default-environment-variables
    return bool(os.environ.get("CI", None))


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


_has_exe_cache = {}
_has_pkg_cache = {}


def has_exe(exe):
    if exe not in _has_exe_cache:
        _has_exe_cache[exe] = bool(which(exe))
    return _has_exe_cache[exe]


def has_pkg(pkg):
    if pkg not in _has_pkg_cache:

        # for some dependencies, package name and import name are different
        # (e.g. pyshp/shapefile, mfpymake/pymake, python-dateutil/dateutil)
        # pkg_resources expects package name, importlib expects import name
        try:
            _has_pkg_cache[pkg] = bool(importlib.import_module(pkg))
        except ModuleNotFoundError:
            try:
                _has_pkg_cache[pkg] = bool(pkg_resources.get_distribution(pkg))
            except pkg_resources.DistributionNotFound:
                _has_pkg_cache[pkg] = False

    return _has_pkg_cache[pkg]


def requires_exe(*exes):
    missing = {exe for exe in exes if not has_exe(exe)}
    return pytest.mark.skipif(
        missing,
        reason=f"missing executable{'s' if len(missing) != 1 else ''}: "
        + ", ".join(missing),
    )


def requires_pkg(*pkgs):
    missing = {pkg for pkg in pkgs if not has_pkg(pkg)}
    return pytest.mark.skipif(
        missing,
        reason=f"missing package{'s' if len(missing) != 1 else ''}: "
        + ", ".join(missing),
    )


def requires_platform(platform, ci_only=False):
    return pytest.mark.skipif(
        system().lower() != platform.lower()
        and (is_in_ci() if ci_only else True),
        reason=f"only compatible with platform: {platform.lower()}",
    )


def excludes_platform(platform, ci_only=False):
    return pytest.mark.skipif(
        system().lower() == platform.lower()
        and (is_in_ci() if ci_only else True),
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
    not is_connected("github.com"), reason="github.com is required."
)


requires_spatial_reference = pytest.mark.skipif(
    not is_connected("spatialreference.org"),
    reason="spatialreference.org is required.",
)


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

    keep_failed = request.config.getoption("--keep-failed")
    if keep_failed and request.node.rep_call.failed:
        copytree(temp, Path(keep_failed) / temp.name)


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


# fixture to automatically close any plots (or optionally show them)


@pytest.fixture(autouse=True)
def close_plot(request):
    yield

    # plots only shown if requested via CLI flag,
    # figures are available, and we're not in CI
    show = request.config.getoption("--show-plots")
    if len(plt.get_fignums()) > 0 and not is_in_ci() and show:
        plt.show()
    else:
        plt.close("all")


# pytest configuration hooks


@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    # this is necessary so temp dir fixtures can
    # inspect test results and check for failure
    # (see https://doc.pytest.org/en/latest/example/simple.html#making-test-result-information-available-in-fixtures)

    outcome = yield
    rep = outcome.get_result()

    # report attribute for each phase (setup, call, teardown)
    # we're only interested in result of the function call
    setattr(item, "rep_" + rep.when, rep)


def pytest_addoption(parser):
    parser.addoption(
        "-K",
        "--keep",
        action="store",
        default=None,
        help="Move the contents of temporary test directories to correspondingly named subdirectories at the given "
        "location after tests complete. This option can be used to exclude test results from automatic cleanup, "
        "e.g. for manual inspection. The provided path is created if it does not already exist. An error is "
        "thrown if any matching files already exist.",
    )

    parser.addoption(
        "--keep-failed",
        action="store",
        default=None,
        help="Move the contents of temporary test directories to correspondingly named subdirectories at the given "
        "location if the test case fails. This option automatically saves the outputs of failed tests in the "
        "given location. The path is created if it doesn't already exist. An error is thrown if files with the "
        "same names already exist in the given location.",
    )

    parser.addoption(
        "-M",
        "--meta",
        action="store",
        metavar="NAME",
        help="Marker indicating a test is only run by other tests (e.g., the test framework testing itself).",
    )

    parser.addoption(
        "-S",
        "--smoke",
        action="store_true",
        default=False,
        help="Run only smoke tests (should complete in <1 minute).",
    )

    parser.addoption(
        "--show-plots",
        action="store_true",
        default=False,
        help="Show any figure windows created by test cases. (Useful to display plots for visual inspection, "
        "but automated tests should probably also check patch collections or figure & axis properties.)",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "meta(name): mark test to run only inside other groups of tests.",
    )


def pytest_runtest_setup(item):
    # apply meta-test option
    meta = item.config.getoption("--meta")
    metagroups = [mark.args[0] for mark in item.iter_markers(name="meta")]
    if metagroups and meta not in metagroups:
        pytest.skip()

    # smoke tests are \ {slow U example U regression}
    smoke = item.config.getoption("--smoke")
    slow = list(item.iter_markers(name="slow"))
    example = list(item.iter_markers(name="example"))
    regression = list(item.iter_markers(name="regression"))
    if smoke and (slow or example or regression):
        pytest.skip()


def pytest_report_header(config):
    """Header for pytest to show versions of packages."""

    # if we ever drop support for python 3.7, could use importlib.metadata instead?
    # or importlib_metadata backport: https://importlib-metadata.readthedocs.io/en/latest/
    # pkg_resources discouraged: https://setuptools.pypa.io/en/latest/pkg_resources.html

    processed = set()
    flopy_pkg = pkg_resources.get_distribution("flopy")
    lines = []
    items = []
    for pkg in flopy_pkg.requires():
        name = pkg.name
        processed.add(name)
        try:
            version = pkg_resources.get_distribution(name).version
            items.append(f"{name}-{version}")
        except pkg_resources.DistributionNotFound:
            items.append(f"{name} (not found)")
    lines.append("required packages: " + ", ".join(items))
    installed = []
    not_found = []
    for pkg in flopy_pkg.requires(["optional"]):
        name = pkg.name
        if name in processed:
            continue
        processed.add(name)
        try:
            version = pkg_resources.get_distribution(name).version
            installed.append(f"{name}-{version}")
        except pkg_resources.DistributionNotFound:
            not_found.append(name)
    if installed:
        lines.append("optional packages: " + ", ".join(installed))
    if not_found:
        lines.append("optional packages not found: " + ", ".join(not_found))
    return "\n".join(lines)


# functions to run commands and scripts


def run_cmd(*args, verbose=False, **kwargs):
    """Run any command, return tuple (stdout, stderr, returncode)."""
    args = [str(g) for g in args]
    if verbose:
        print("running: " + " ".join(args))
    p = Popen(args, stdout=PIPE, stderr=PIPE, **kwargs)
    stdout, stderr = p.communicate()
    stdout = stdout.decode()
    stderr = stderr.decode()
    returncode = p.returncode
    if verbose:
        print(f"stdout:\n{stdout}")
        print(f"stderr:\n{stderr}")
        print(f"returncode: {returncode}")
    return stdout, stderr, returncode


def run_py_script(script, *args, verbose=False):
    """Run a Python script, return tuple (stdout, stderr, returncode)."""
    return run_cmd(
        sys.executable, script, *args, verbose=verbose, cwd=Path(script).parent
    )


# use noninteractive matplotlib backend if in Mac OS CI to avoid pytest-xdist node failure
# e.g. https://github.com/modflowpy/flopy/runs/7748574375?check_suite_focus=true#step:9:57
@pytest.fixture(scope="session", autouse=True)
def patch_macos_ci_matplotlib():
    if is_in_ci() and system().lower() == "darwin":
        import matplotlib

        matplotlib.use("agg")
